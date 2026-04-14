"""
Mask Generator - Interactive OpenCV mask playground
Features:
  - Gaussian blur
  - Global grayscale threshold (6 modes)
  - Morphology cleanup
  - Double mask (ROI)
  - Color mask (HSV / RGB / LAB range, click-to-pick)
  - Light adaptation (CLAHE, Retinex, Gamma, Gray World, Hist-EQ, LAB-norm)
  - Contours
  - Alpha blend
  - Export / copy Python code
"""
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clamp_odd(v, lo, hi):
    v = max(lo, min(int(v), hi))
    return v if v % 2 == 1 else (v - 1 if v > lo else v + 1)


THRESH_MODES  = ["IN_RANGE", "BINARY", "BINARY_INV", "OTSU", "ADAPTIVE_MEAN", "ADAPTIVE_GAUSSIAN"]
COMBINE_MODES = ["REPLACE", "AND", "OR", "XOR"]
SHOW_MODES    = ["Blended", "Mask only", "Masked image"]
LIGHT_MODES   = ["None", "CLAHE", "Hist-EQ", "Retinex-SSR", "Gamma", "Gray World", "LAB-norm"]
COLOR_SPACES  = ["HSV", "RGB", "LAB"]
COLOR_COMBINE = ["AND", "OR", "REPLACE gray", "REPLACE color"]

# Channel labels per color space
CH_LABELS = {
    "HSV": ("H  0–179", "S  0–255", "V  0–255"),
    "RGB": ("R  0–255", "G  0–255", "B  0–255"),
    "LAB": ("L  0–255", "A  0–255", "B  0–255"),
}
CH_MAX = {
    "HSV": (179, 255, 255),
    "RGB": (255, 255, 255),
    "LAB": (255, 255, 255),
}


# ---------------------------------------------------------------------------
# Grayscale threshold
# ---------------------------------------------------------------------------

def apply_threshold(gray, mode, low, high, block_size=11):
    """Return a binary mask (uint8, 0/255)."""
    if mode == "IN_RANGE":
        return cv2.inRange(gray, int(low), int(high))
    elif mode == "BINARY":
        _, m = cv2.threshold(gray, int(low), int(high), cv2.THRESH_BINARY)
        return m
    elif mode == "BINARY_INV":
        _, m = cv2.threshold(gray, int(low), int(high), cv2.THRESH_BINARY_INV)
        return m
    elif mode == "OTSU":
        _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return m
    elif mode == "ADAPTIVE_MEAN":
        bs = clamp_odd(max(3, block_size), 3, 255)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, bs, int(low) // 10)
    elif mode == "ADAPTIVE_GAUSSIAN":
        bs = clamp_odd(max(3, block_size), 3, 255)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, bs, int(low) // 10)
    return np.zeros_like(gray)


# ---------------------------------------------------------------------------
# Light / shade adaptation  (returns corrected RGB uint8)
# ---------------------------------------------------------------------------

def apply_light_adaptation(rgb, mode, clahe_clip=2.0, clahe_tile=8,
                            gamma=1.0, retinex_sigma=80):
    if mode == "None":
        return rgb

    if mode == "Gamma":
        lut = (255.0 * (np.arange(256) / 255.0) ** (1.0 / max(gamma, 0.05))).astype(np.uint8)
        return cv2.LUT(rgb, lut)

    if mode == "Gray World":
        # Rescale each channel so its mean equals the overall mean brightness
        result = rgb.astype(np.float32)
        means = result.mean(axis=(0, 1))          # (3,)
        target = means.mean()
        for c in range(3):
            if means[c] > 0:
                result[:, :, c] *= target / means[c]
        return result.clip(0, 255).astype(np.uint8)

    # --- LAB-based methods ---
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch = lab[:, :, 0]

    if mode == "Hist-EQ":
        lab[:, :, 0] = cv2.equalizeHist(l_ch)

    elif mode == "CLAHE":
        tile = max(1, int(clahe_tile))
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip),
                                 tileGridSize=(tile, tile))
        lab[:, :, 0] = clahe.apply(l_ch)

    elif mode == "LAB-norm":
        # Stretch L channel to full 0-255 range
        mn, mx = l_ch.min(), l_ch.max()
        if mx > mn:
            lab[:, :, 0] = ((l_ch.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)

    elif mode == "Retinex-SSR":
        # Single-Scale Retinex on L channel
        sigma = max(1.0, float(retinex_sigma))
        l_f = l_ch.astype(np.float32) + 1.0
        blurred = cv2.GaussianBlur(l_f, (0, 0), sigma)
        retinex = np.log(l_f) - np.log(blurred + 1e-6)
        # Normalise back to 0-255
        retinex -= retinex.min()
        mx = retinex.max()
        if mx > 0:
            retinex = (retinex / mx * 255).astype(np.uint8)
        else:
            retinex = np.zeros_like(l_ch)
        lab[:, :, 0] = retinex

    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ---------------------------------------------------------------------------
# Color-space mask
# ---------------------------------------------------------------------------

def apply_color_mask(rgb, space,
                     c1_lo, c1_hi, c2_lo, c2_hi, c3_lo, c3_hi,
                     density_weight=False):
    """
    Returns binary mask (uint8 0/255).
    density_weight: weight the mask by the saturation (HSV) or chroma (LAB)
    so that more saturated / vivid regions contribute more.
    """
    if space == "HSV":
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    elif space == "LAB":
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    else:  # RGB
        img = rgb.copy()

    lo = np.array([int(c1_lo), int(c2_lo), int(c3_lo)], dtype=np.uint8)
    hi = np.array([int(c1_hi), int(c2_hi), int(c3_hi)], dtype=np.uint8)

    # Handle HSV hue wrap-around (e.g. red: 170-10)
    if space == "HSV" and int(c1_lo) > int(c1_hi):
        m1 = cv2.inRange(img, np.array([int(c1_lo), int(c2_lo), int(c3_lo)], np.uint8),
                              np.array([179,         int(c2_hi), int(c3_hi)], np.uint8))
        m2 = cv2.inRange(img, np.array([0,           int(c2_lo), int(c3_lo)], np.uint8),
                              np.array([int(c1_hi),  int(c2_hi), int(c3_hi)], np.uint8))
        mask = cv2.bitwise_or(m1, m2)
    else:
        mask = cv2.inRange(img, lo, hi)

    if density_weight:
        # Weight by saturation (HSV S) or chroma (LAB: sqrt(a^2+b^2))
        if space == "HSV":
            weight = img[:, :, 1].astype(np.float32) / 255.0
        elif space == "LAB":
            a = img[:, :, 1].astype(np.float32) - 128
            b = img[:, :, 2].astype(np.float32) - 128
            chroma = np.sqrt(a**2 + b**2)
            weight = chroma / (chroma.max() + 1e-6)
        else:  # RGB — use distance from gray
            r = img[:, :, 0].astype(np.float32)
            g = img[:, :, 1].astype(np.float32)
            b_ch = img[:, :, 2].astype(np.float32)
            mean = (r + g + b_ch) / 3.0
            chroma = np.sqrt((r-mean)**2 + (g-mean)**2 + (b_ch-mean)**2)
            weight = chroma / (chroma.max() + 1e-6)

        # Blend: keep mask shape but dim low-density regions
        mask_f = mask.astype(np.float32) / 255.0
        weighted = (mask_f * weight * 255).clip(0, 255).astype(np.uint8)
        _, mask = cv2.threshold(weighted, 50, 255, cv2.THRESH_BINARY)

    return mask


# ---------------------------------------------------------------------------
# Widget helper
# ---------------------------------------------------------------------------

class LabeledScale(tk.Frame):
    def __init__(self, parent, label, from_, to, resolution, init, command=None, **kw):
        super().__init__(parent, bg="#2b2b2b")
        self._cmd = command
        self.lbl_widget = tk.Label(self, text=label, bg="#2b2b2b", fg="#cccccc",
                                   width=20, anchor="w", font=("Consolas", 8))
        self.lbl_widget.pack(side="left")
        self.var = tk.DoubleVar(value=init)
        self._val_lbl = tk.Label(self, text=f"{init:.2g}", bg="#2b2b2b", fg="#ffcc55",
                                 width=5, font=("Consolas", 8))
        self._val_lbl.pack(side="right")
        self.scale = tk.Scale(self, variable=self.var, from_=from_, to=to,
                              resolution=resolution, orient="horizontal",
                              command=self._on_change, showvalue=False,
                              bg="#2b2b2b", fg="#cccccc", troughcolor="#555",
                              highlightthickness=0, **kw)
        self.scale.pack(side="left", fill="x", expand=True)

    def _on_change(self, val):
        v = float(val)
        self._val_lbl.config(text=f"{v:.3g}")
        if self._cmd:
            self._cmd(val)

    def get(self):       return self.var.get()
    def set(self, v):    self.var.set(v)
    def set_label(self, t): self.lbl_widget.config(text=t)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

class MaskGeneratorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mask Generator")
        self.geometry("1500x900")
        self.configure(bg="#1a1a1a")

        self.original   = None     # RGB numpy array
        self.roi        = None     # (x1,y1,x2,y2) image pixels
        self._roi_start = None
        self._disp_scale = 1.0
        self._disp_ox    = 0
        self._disp_oy    = 0
        self._click_mode = "roi"   # "roi" | "pick"

        self._build_ui()
        self.bind("<Configure>", lambda e: self._process_and_display())

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.left_frame = tk.Frame(self, bg="#1a1a1a")
        self.left_frame.pack(side="left", fill="both", expand=True)

        right_frame = tk.Frame(self, bg="#2b2b2b", width=370)
        right_frame.pack(side="right", fill="y")
        right_frame.pack_propagate(False)

        self.orig_canvas   = tk.Canvas(self.left_frame, bg="#111",
                                       cursor="crosshair", highlightthickness=0)
        self.result_canvas = tk.Canvas(self.left_frame, bg="#111", highlightthickness=0)
        self.orig_canvas.pack(side="left",  fill="both", expand=True)
        self.result_canvas.pack(side="right", fill="both", expand=True)

        self.orig_canvas.bind("<ButtonPress-1>",   self._left_press)
        self.orig_canvas.bind("<B1-Motion>",        self._left_drag)
        self.orig_canvas.bind("<ButtonRelease-1>",  self._left_release)
        self.orig_canvas.bind("<ButtonPress-3>",    lambda e: self._clear_roi())

        self._build_controls(right_frame)

    def _build_controls(self, parent):
        outer = tk.Frame(parent, bg="#2b2b2b")
        outer.pack(fill="both", expand=True)
        cv = tk.Canvas(outer, bg="#2b2b2b", highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        cv.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(cv, bg="#2b2b2b")
        inner_win = cv.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>", lambda e: cv.itemconfig(inner_win, width=e.width))
        cv.bind_all("<MouseWheel>", lambda e: cv.yview_scroll(-1*(e.delta//120), "units"))

        p  = {"padx": 6, "pady": 2,    "fill": "x"}
        ps = {"padx": 6, "pady": (8,2),"fill": "x"}

        def section(txt):
            tk.Label(inner, text=f"  {txt}", bg="#3a3a5a", fg="#aaaaff",
                     font=("Consolas", 9, "bold"), anchor="w").pack(**ps)

        def row_check(label, init):
            var = tk.BooleanVar(value=init)
            tk.Checkbutton(inner, text=label, variable=var, command=self._on_change,
                           bg="#2b2b2b", fg="#cccccc", selectcolor="#555",
                           activebackground="#2b2b2b", activeforeground="#fff").pack(**p)
            return var

        def row_combo(label, values, init):
            f = tk.Frame(inner, bg="#2b2b2b")
            f.pack(**p)
            tk.Label(f, text=label, bg="#2b2b2b", fg="#cccccc",
                     width=20, anchor="w", font=("Consolas", 8)).pack(side="left")
            var = tk.StringVar(value=init)
            cb = ttk.Combobox(f, textvariable=var, values=values, state="readonly", width=16)
            cb.pack(side="left")
            cb.bind("<<ComboboxSelected>>", self._on_change)
            return var

        def sldr(label, lo, hi, res, init):
            s = LabeledScale(inner, label, lo, hi, res, init, command=self._on_change)
            s.pack(**p)
            return s

        # ── File ──
        section("FILE")
        tk.Button(inner, text="Load Image", command=self.load_image,
                  bg="#4a7eba", fg="white", relief="flat").pack(**p)
        self._export_btn = tk.Button(inner, text="Export result as PNG",
                                     command=self._export_png,
                                     bg="#7a5a00", fg="white", relief="flat")
        self._export_btn.pack(**p)

        # ── Light Adaptation ──
        section("LIGHT ADAPTATION")
        self.light_mode     = row_combo("Mode", LIGHT_MODES, "None")
        self.light_mode_var = self.light_mode  # alias for pipeline
        self.clahe_clip     = sldr("CLAHE clip limit", 0.5, 20.0, 0.5, 2.0)
        self.clahe_tile     = sldr("CLAHE tile size", 2, 32, 1, 8)
        self.gamma          = sldr("Gamma  (>1 = darker)", 0.1, 5.0, 0.05, 1.0)
        self.retinex_sigma  = sldr("Retinex sigma", 10, 300, 5, 80)

        # ── Blur ──
        section("BLUR")
        self.blur_k = sldr("Blur kernel (odd 1–51)", 1, 51, 2, 5)

        # ── Global Mask ──
        section("GLOBAL MASK  (grayscale)")
        self.thresh_mode = row_combo("Mode", THRESH_MODES, "IN_RANGE")
        self.thresh_low  = sldr("Low threshold", 0, 255, 1, 60)
        self.thresh_high = sldr("High threshold / maxval", 0, 255, 1, 200)
        self.block_size  = sldr("Adaptive block size", 3, 99, 2, 11)

        # ── Morphology ──
        section("MORPHOLOGY (mask cleanup)")
        self.morph_mode = row_combo("Op", ["None","Erode","Dilate","Open","Close"], "None")
        self.morph_k    = sldr("Kernel size", 1, 21, 2, 3)

        # ── Color Mask ──
        section("COLOR MASK  (click image to pick)")
        self.color_mask_var = row_check("Enable color mask", False)
        self.color_space    = row_combo("Color space", COLOR_SPACES, "HSV")
        self.color_space_cb = None  # keep reference for binding
        self.color_combine  = row_combo("Combine w/ gray mask", COLOR_COMBINE, "AND")
        self.density_weight = row_check("Density weight (saturation/chroma)", False)

        # Pick-color button
        self._pick_btn = tk.Button(inner, text="Pick color from image  [OFF]",
                                   command=self._toggle_pick_mode,
                                   bg="#5a3a7a", fg="white", relief="flat")
        self._pick_btn.pack(**p)
        self.pick_tolerance = sldr("Pick tolerance ±", 0, 60, 1, 20)

        # 3 × (lo, hi) sliders — labels updated dynamically
        self.c1_lo = sldr("H/R/L  low",  0, 255, 1, 0)
        self.c1_hi = sldr("H/R/L  high", 0, 255, 1, 179)
        self.c2_lo = sldr("S/G/a  low",  0, 255, 1, 50)
        self.c2_hi = sldr("S/G/a  high", 0, 255, 1, 255)
        self.c3_lo = sldr("V/B/b  low",  0, 255, 1, 50)
        self.c3_hi = sldr("V/B/b  high", 0, 255, 1, 255)

        # ── Color range visualizer ──
        tk.Label(inner, text="  Color range preview", bg="#2b2b2b", fg="#888888",
                 font=("Consolas", 8), anchor="w").pack(padx=6, pady=(4,0), fill="x")
        self.color_viz_canvas = tk.Canvas(inner, height=110, bg="#111",
                                          highlightthickness=1,
                                          highlightbackground="#444")
        self.color_viz_canvas.pack(padx=6, pady=(0,4), fill="x")

        # Update channel labels when color space changes
        def _on_space_change(e=None):
            sp = self.color_space.get()
            lbls = CH_LABELS[sp]
            mx   = CH_MAX[sp]
            pairs = [(self.c1_lo, self.c1_hi, lbls[0], mx[0]),
                     (self.c2_lo, self.c2_hi, lbls[1], mx[1]),
                     (self.c3_lo, self.c3_hi, lbls[2], mx[2])]
            for lo_w, hi_w, lbl, m in pairs:
                lo_w.set_label(f"{lbl}  lo")
                hi_w.set_label(f"{lbl}  hi")
                lo_w.scale.config(to=m)
                hi_w.scale.config(to=m)
            self._on_change()

        # Find the combobox widget inside the row_combo frame and re-bind
        for child in inner.winfo_children():
            for sub in (child.winfo_children() if hasattr(child, "winfo_children") else []):
                if isinstance(sub, ttk.Combobox) and sub.cget("textvariable") == str(self.color_space):
                    sub.bind("<<ComboboxSelected>>", lambda e: (_on_space_change(), None))
        # Simpler: just call on every change via _on_change and also bind directly
        self._space_change_cb = _on_space_change

        # ── Double Mask (ROI) ──
        section("DOUBLE MASK  (draw ROI on left)")
        self.double_var   = row_check("Enable double mask", False)
        self.combine_mode = row_combo("Combine", COMBINE_MODES, "REPLACE")
        self.thresh_mode2 = row_combo("ROI mode", THRESH_MODES, "BINARY_INV")
        self.blur_k2      = sldr("ROI blur kernel", 1, 51, 2, 5)
        self.thresh_low2  = sldr("ROI low threshold", 0, 255, 1, 100)
        self.thresh_high2 = sldr("ROI high / maxval", 0, 255, 1, 200)
        self.block_size2  = sldr("ROI adaptive block", 3, 99, 2, 11)

        # ── Contours ──
        section("CONTOURS")
        self.contours_var  = row_check("Draw contours", False)
        self.contour_thick = sldr("Thickness", 1, 10, 1, 2)
        self.contour_min   = sldr("Min area", 0, 10000, 50, 200)

        # ── Blend & Display ──
        section("BLEND & DISPLAY")
        self.alpha_s   = sldr("Alpha (original weight)", 0.0, 1.0, 0.01, 0.5)
        self.show_mode = row_combo("Show as", SHOW_MODES, "Blended")

        # ── ROI status ──
        section("ROI  (right-click canvas to clear)")
        self.roi_lbl = tk.Label(inner, text="No ROI – draw on left image",
                                bg="#2b2b2b", fg="#888888", justify="left",
                                font=("Consolas", 8))
        self.roi_lbl.pack(**p)
        tk.Button(inner, text="Clear ROI", command=self._clear_roi,
                  bg="#7a3030", fg="white", relief="flat").pack(**p)

        # ── Picked colour swatch ──
        self._swatch = tk.Label(inner, text="  picked colour  ",
                                bg="#2b2b2b", fg="#888", font=("Consolas", 8))
        self._swatch.pack(**p)

        # ── Export code ──
        section("EXPORT CODE")
        tk.Button(inner, text="Generate Python Code", command=self._generate_code,
                  bg="#3a7a3a", fg="white", relief="flat").pack(**p)
        tk.Button(inner, text="Copy to Clipboard", command=self._copy_code,
                  bg="#5a5a2a", fg="white", relief="flat").pack(**p)
        self.code_text = tk.Text(inner, height=12, bg="#111", fg="#88ff88",
                                 font=("Consolas", 7), wrap="none",
                                 insertbackground="#fff", relief="flat",
                                 selectbackground="#334")
        self.code_text.pack(padx=6, pady=(2, 4), fill="x")

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp")])
        if not path:
            return
        bgr = cv2.imread(path)
        if bgr is None:
            return
        self.original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._clear_roi()
        self._process_and_display()

    # ------------------------------------------------------------------
    # Click mode & ROI
    # ------------------------------------------------------------------

    def _toggle_pick_mode(self):
        if self._click_mode == "roi":
            self._click_mode = "pick"
            self._pick_btn.config(text="Pick color from image  [ON]", bg="#aa55cc")
            self.orig_canvas.config(cursor="plus")
        else:
            self._click_mode = "roi"
            self._pick_btn.config(text="Pick color from image  [OFF]", bg="#5a3a7a")
            self.orig_canvas.config(cursor="crosshair")

    def _left_press(self, e):
        if self._click_mode == "pick":
            self._pick_color(e)
        else:
            self._roi_press(e)

    def _left_drag(self, e):
        if self._click_mode == "roi":
            self._roi_drag(e)

    def _left_release(self, e):
        if self._click_mode == "roi":
            self._roi_release(e)

    def _pick_color(self, e):
        if self.original is None:
            return
        ix, iy = self._canvas_to_img(e.x, e.y)
        rgb_px = self.original[iy, ix].astype(int)   # (R,G,B)
        tol = int(self.pick_tolerance.get())

        sp = self.color_space.get()
        if sp == "HSV":
            px_hsv = cv2.cvtColor(np.uint8([[rgb_px]]), cv2.COLOR_RGB2HSV)[0, 0].astype(int)
            vals = px_hsv
        elif sp == "LAB":
            px_lab = cv2.cvtColor(np.uint8([[rgb_px]]), cv2.COLOR_RGB2LAB)[0, 0].astype(int)
            vals = px_lab
        else:
            vals = rgb_px

        maxs = CH_MAX[sp]
        pairs = [(self.c1_lo, self.c1_hi, vals[0], maxs[0]),
                 (self.c2_lo, self.c2_hi, vals[1], maxs[1]),
                 (self.c3_lo, self.c3_hi, vals[2], maxs[2])]
        for lo_w, hi_w, v, mx in pairs:
            lo_w.set(max(0, v - tol))
            hi_w.set(min(mx, v + tol))

        # Update swatch
        r, g, b = int(rgb_px[0]), int(rgb_px[1]), int(rgb_px[2])
        hex_col = f"#{r:02x}{g:02x}{b:02x}"
        self._swatch.config(
            text=f"  RGB({r},{g},{b})  {sp}({vals[0]},{vals[1]},{vals[2]})  ",
            bg=hex_col, fg="white" if (r+g+b) < 384 else "black")

        self.color_mask_var.set(True)
        self._on_change()

    def _roi_press(self, e):
        if self.original is None:
            return
        self._roi_start = (e.x, e.y)
        self.orig_canvas.delete("roi_rect")

    def _roi_drag(self, e):
        if self._roi_start is None:
            return
        self.orig_canvas.delete("roi_rect")
        x0, y0 = self._roi_start
        self.orig_canvas.create_rectangle(x0, y0, e.x, e.y,
                                          outline="#00ff88", width=2,
                                          dash=(4, 3), tags="roi_rect")

    def _roi_release(self, e):
        if self._roi_start is None:
            return
        p1 = self._canvas_to_img(*self._roi_start)
        p2 = self._canvas_to_img(e.x, e.y)
        self._roi_start = None
        x1, y1 = p1
        x2, y2 = p2
        if abs(x2-x1) > 4 and abs(y2-y1) > 4:
            self.roi = (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))
            self.roi_lbl.config(
                text=f"ROI: ({self.roi[0]},{self.roi[1]}) → ({self.roi[2]},{self.roi[3]})\n"
                     f"Size: {self.roi[2]-self.roi[0]} × {self.roi[3]-self.roi[1]} px",
                fg="#88ff88")
            self._process_and_display()

    def _clear_roi(self):
        self.roi = None
        self._roi_start = None
        self.orig_canvas.delete("roi_rect")
        self.roi_lbl.config(text="No ROI – draw on left image", fg="#888888")
        self._process_and_display()

    def _canvas_to_img(self, cx, cy):
        if self.original is None:
            return (0, 0)
        h, w = self.original.shape[:2]
        x = int((cx - self._disp_ox) / self._disp_scale)
        y = int((cy - self._disp_oy) / self._disp_scale)
        return (max(0, min(x, w-1)), max(0, min(y, h-1)))

    def _img_to_canvas(self, ix, iy):
        return (int(ix * self._disp_scale + self._disp_ox),
                int(iy * self._disp_scale + self._disp_oy))

    # ------------------------------------------------------------------
    # Processing pipeline
    # ------------------------------------------------------------------

    def _on_change(self, _=None):
        self._draw_color_viz()
        self._process_and_display()

    # ------------------------------------------------------------------
    # Color range visualizer
    # ------------------------------------------------------------------

    def _draw_color_viz(self):
        c = self.color_viz_canvas
        c.update_idletasks()
        W = c.winfo_width()
        if W < 10:
            W = 300
        H = 110
        sp = self.color_space.get()

        # ── Top strip: channel-1 gradient (Hue / R / L) ──────────────
        # Full gradient across channel-1 range, with selection highlighted
        STRIP_H = 18
        c1_lo = int(self.c1_lo.get())
        c1_hi = int(self.c1_hi.get())
        c2_mid = int((self.c2_lo.get() + self.c2_hi.get()) / 2)
        c3_mid = int((self.c3_lo.get() + self.c3_hi.get()) / 2)
        c1_max = CH_MAX[sp][0]

        strip = np.zeros((STRIP_H, W, 3), dtype=np.uint8)
        for x in range(W):
            v1 = int(x / W * c1_max)
            if sp == "HSV":
                pixel = cv2.cvtColor(np.uint8([[[v1, c2_mid, c3_mid]]]),
                                     cv2.COLOR_HSV2RGB)[0, 0]
            elif sp == "LAB":
                pixel = cv2.cvtColor(np.uint8([[[v1, c2_mid, c3_mid]]]),
                                     cv2.COLOR_LAB2RGB)[0, 0]
            else:
                pixel = np.array([v1, c2_mid, c3_mid], dtype=np.uint8)
            # dim if outside selected range
            in_range = c1_lo <= v1 <= c1_hi if c1_lo <= c1_hi else (v1 >= c1_lo or v1 <= c1_hi)
            strip[:, x] = pixel if in_range else (pixel * 0.25).astype(np.uint8)

        # Draw range bracket on strip
        x_lo = int(c1_lo / c1_max * (W - 1))
        x_hi = int(c1_hi / c1_max * (W - 1))
        strip[0, :] = 60
        strip[-1, :] = 60
        strip[:, x_lo] = [255, 255, 0]
        strip[:, min(x_hi, W-1)] = [255, 255, 0]

        # ── Bottom 2D view: ch1 × ch2 grid ───────────────────────────
        VIZ_H = H - STRIP_H - 4
        VIZ_W = W

        # Build colour grid
        xs = np.linspace(0, c1_max, VIZ_W).astype(int)       # ch1 → X
        ys = np.linspace(CH_MAX[sp][1], 0, VIZ_H).astype(int)  # ch2 → Y (inverted)

        grid_c1 = np.tile(xs[np.newaxis, :], (VIZ_H, 1))   # (H,W)
        grid_c2 = np.tile(ys[:, np.newaxis], (1, VIZ_W))   # (H,W)
        grid_c3 = np.full((VIZ_H, VIZ_W), c3_mid, dtype=np.uint8)

        src = np.stack([grid_c1, grid_c2, grid_c3], axis=2).astype(np.uint8)

        if sp == "HSV":
            grid_rgb = cv2.cvtColor(src, cv2.COLOR_HSV2RGB)
        elif sp == "LAB":
            grid_rgb = cv2.cvtColor(src, cv2.COLOR_LAB2RGB)
        else:
            grid_rgb = src.copy()

        # Build in-range boolean mask for the grid
        c2_lo = int(self.c2_lo.get())
        c2_hi = int(self.c2_hi.get())
        in_c1 = (grid_c1 >= c1_lo) & (grid_c1 <= c1_hi) if c1_lo <= c1_hi else \
                (grid_c1 >= c1_lo) | (grid_c1 <= c1_hi)
        in_c2 = (grid_c2 >= c2_lo) & (grid_c2 <= c2_hi)
        in_range_2d = in_c1 & in_c2   # (ignoring c3 for spatial layout)

        # Dim out-of-range pixels
        dim = grid_rgb.astype(np.float32)
        dim[~in_range_2d] = dim[~in_range_2d] * 0.18
        grid_rgb = dim.clip(0, 255).astype(np.uint8)

        # Draw selection rectangle border
        y_lo_px = int((1.0 - c2_hi / CH_MAX[sp][1]) * (VIZ_H - 1))
        y_hi_px = int((1.0 - c2_lo / CH_MAX[sp][1]) * (VIZ_H - 1))
        x_lo_px = int(c1_lo / c1_max * (VIZ_W - 1))
        x_hi_px = int(min(c1_hi, c1_max) / c1_max * (VIZ_W - 1))
        cv2.rectangle(grid_rgb, (x_lo_px, y_lo_px), (x_hi_px, y_hi_px), (255, 255, 0), 1)

        # Channel axis labels
        lbl1, lbl2 = CH_LABELS[sp][0].split()[0], CH_LABELS[sp][1].split()[0]
        cv2.putText(grid_rgb, f"{lbl2}^", (2, 12), cv2.FONT_HERSHEY_PLAIN, 0.8, (200,200,200), 1)
        cv2.putText(grid_rgb, f"{lbl1}>", (VIZ_W - 28, VIZ_H - 4), cv2.FONT_HERSHEY_PLAIN, 0.8, (200,200,200), 1)

        # ── Combine strip + 2D grid into one image ───────────────────
        gap = np.zeros((4, W, 3), dtype=np.uint8)
        viz = np.vstack([strip, gap, grid_rgb])

        pil_img = Image.fromarray(viz)
        tk_img = ImageTk.PhotoImage(pil_img)
        c.config(height=viz.shape[0])
        c.delete("all")
        c.create_image(0, 0, anchor="nw", image=tk_img)
        c.img_ref = tk_img   # prevent GC

    def _run_pipeline(self):
        """Run full pipeline and return the result image (RGB, original resolution)."""
        if self.original is None:
            return None

        # ── 1. Light adaptation ──
        adapted = apply_light_adaptation(
            self.original,
            self.light_mode.get(),
            clahe_clip    = float(self.clahe_clip.get()),
            clahe_tile    = int(self.clahe_tile.get()),
            gamma         = float(self.gamma.get()),
            retinex_sigma = float(self.retinex_sigma.get()),
        )

        gray = cv2.cvtColor(adapted, cv2.COLOR_RGB2GRAY)

        # ── 2. Blur ──
        k = clamp_odd(self.blur_k.get(), 1, 51)
        blurred = cv2.GaussianBlur(gray, (k, k), 0) if k > 1 else gray.copy()

        # ── 3. Grayscale mask ──
        gray_mask = apply_threshold(
            blurred,
            self.thresh_mode.get(),
            self.thresh_low.get(),
            self.thresh_high.get(),
            int(self.block_size.get()),
        )

        # ── 4. Morphology ──
        mop = self.morph_mode.get()
        mk  = clamp_odd(self.morph_k.get(), 1, 21)
        if mop != "None" and mk > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
            ops = {"Erode": cv2.MORPH_ERODE, "Dilate": cv2.MORPH_DILATE,
                   "Open":  cv2.MORPH_OPEN,  "Close":  cv2.MORPH_CLOSE}
            gray_mask = cv2.morphologyEx(gray_mask, ops[mop], kernel)

        # ── 5. Color mask ──
        if self.color_mask_var.get():
            color_mask = apply_color_mask(
                adapted,
                self.color_space.get(),
                self.c1_lo.get(), self.c1_hi.get(),
                self.c2_lo.get(), self.c2_hi.get(),
                self.c3_lo.get(), self.c3_hi.get(),
                density_weight=bool(self.density_weight.get()),
            )
            cc = self.color_combine.get()
            if cc == "AND":
                mask = cv2.bitwise_and(gray_mask, color_mask)
            elif cc == "OR":
                mask = cv2.bitwise_or(gray_mask, color_mask)
            elif cc == "REPLACE gray":
                mask = color_mask
            else:
                mask = cv2.bitwise_and(color_mask, gray_mask)
        else:
            mask = gray_mask

        # ── 6. Double mask (ROI) ──
        if self.double_var.get() and self.roi is not None:
            x1, y1, x2, y2 = self.roi
            roi_gray = gray[y1:y2, x1:x2]
            k2 = clamp_odd(self.blur_k2.get(), 1, 51)
            roi_blurred = cv2.GaussianBlur(roi_gray, (k2, k2), 0) if k2 > 1 else roi_gray.copy()
            roi_mask = apply_threshold(
                roi_blurred,
                self.thresh_mode2.get(),
                self.thresh_low2.get(),
                self.thresh_high2.get(),
                int(self.block_size2.get()),
            )
            g = mask[y1:y2, x1:x2]
            combine = self.combine_mode.get()
            if combine == "REPLACE":
                mask[y1:y2, x1:x2] = roi_mask
            elif combine == "AND":
                mask[y1:y2, x1:x2] = cv2.bitwise_and(g, roi_mask)
            elif combine == "OR":
                mask[y1:y2, x1:x2] = cv2.bitwise_or(g, roi_mask)
            elif combine == "XOR":
                mask[y1:y2, x1:x2] = cv2.bitwise_xor(g, roi_mask)

        # ── 7. Contours ──
        contour_img = None
        if self.contours_var.get():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [c for c in contours if cv2.contourArea(c) >= float(self.contour_min.get())]
            contour_img = self.original.copy()
            cv2.drawContours(contour_img, contours, -1, (0, 255, 100), int(self.contour_thick.get()))

        # ── 8. Build result ──
        show     = self.show_mode.get()
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        if show == "Mask only":
            result = mask_rgb
        elif show == "Masked image":
            result = cv2.bitwise_and(self.original, mask_rgb)
        else:  # Blended
            alpha  = float(self.alpha_s.get())
            orig_f = self.original.astype(np.float32) / 255.0
            mask_f = mask_rgb.astype(np.float32) / 255.0
            result = (orig_f * alpha + mask_f * (1.0 - alpha)).clip(0, 1)
            result = (result * 255).astype(np.uint8)

        if contour_img is not None:
            result = cv2.addWeighted(result, 0.65, contour_img, 0.35, 0)

        if self.roi and self.double_var.get():
            x1, y1, x2, y2 = self.roi
            result = result.copy()
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 200, 255), 2)

        return result

    def _process_and_display(self, _=None):
        result = self._run_pipeline()
        if result is None:
            return
        self._show_on_canvas(self.original, self.orig_canvas,
                             label="Original  (draw ROI | pick colour)", draw_roi=True)
        self._show_on_canvas(result, self.result_canvas, label="Result")

    def _export_png(self):
        if self.original is None:
            return
        result = self._run_pipeline()
        if result is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
            title="Save result image",
        )
        if not path:
            return
        bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
        self._export_btn.config(text=f"Saved  {path.split('/')[-1].split(chr(92))[-1]}")

    def _show_on_canvas(self, img, canvas, label="", draw_roi=False):
        canvas.update_idletasks()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        h, w = img.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2

        if draw_roi:
            self._disp_scale = scale
            self._disp_ox = ox
            self._disp_oy = oy

        pil    = Image.fromarray(img).resize((nw, nh), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil)
        canvas.delete("img")
        canvas.create_image(ox, oy, anchor="nw", image=tk_img, tags="img")
        canvas.img_ref = tk_img

        canvas.delete("label")
        canvas.create_text(6, 6, anchor="nw", text=label,
                           fill="#888888", font=("Consolas", 8), tags="label")

        if draw_roi and self.roi:
            x1, y1, x2, y2 = self.roi
            cx1, cy1 = self._img_to_canvas(x1, y1)
            cx2, cy2 = self._img_to_canvas(x2, y2)
            canvas.delete("roi_rect")
            canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="#00ff88",
                                    width=2, dash=(4,3), tags="roi_rect")
            canvas.create_text(cx1+3, cy1+3, anchor="nw", text="ROI",
                               fill="#00ff88", font=("Consolas", 8), tags="roi_rect")

    # ------------------------------------------------------------------
    # Code export
    # ------------------------------------------------------------------

    def _collect_params(self):
        return dict(
            light_mode     = self.light_mode.get(),
            clahe_clip     = float(self.clahe_clip.get()),
            clahe_tile     = int(self.clahe_tile.get()),
            gamma          = float(self.gamma.get()),
            retinex_sigma  = float(self.retinex_sigma.get()),
            blur_k         = clamp_odd(self.blur_k.get(), 1, 51),
            thresh_mode    = self.thresh_mode.get(),
            thresh_low     = int(self.thresh_low.get()),
            thresh_high    = int(self.thresh_high.get()),
            block_size     = int(self.block_size.get()),
            morph_mode     = self.morph_mode.get(),
            morph_k        = clamp_odd(self.morph_k.get(), 1, 21),
            color_mask     = bool(self.color_mask_var.get()),
            color_space    = self.color_space.get(),
            color_combine  = self.color_combine.get(),
            density_weight = bool(self.density_weight.get()),
            c1_lo=int(self.c1_lo.get()), c1_hi=int(self.c1_hi.get()),
            c2_lo=int(self.c2_lo.get()), c2_hi=int(self.c2_hi.get()),
            c3_lo=int(self.c3_lo.get()), c3_hi=int(self.c3_hi.get()),
            double_mask    = bool(self.double_var.get()),
            roi            = self.roi,
            combine_mode   = self.combine_mode.get(),
            thresh_mode2   = self.thresh_mode2.get(),
            blur_k2        = clamp_odd(self.blur_k2.get(), 1, 51),
            thresh_low2    = int(self.thresh_low2.get()),
            thresh_high2   = int(self.thresh_high2.get()),
            block_size2    = int(self.block_size2.get()),
            contours       = bool(self.contours_var.get()),
            contour_thick  = int(self.contour_thick.get()),
            contour_min    = float(self.contour_min.get()),
            alpha          = float(self.alpha_s.get()),
            show_mode      = self.show_mode.get(),
        )

    def _build_code(self, p):
        # light block
        light_block = ""
        if p['light_mode'] != "None":
            light_block = f"""
    adapted = apply_light_adaptation(img_rgb, "{p['light_mode']}",
        clahe_clip={p['clahe_clip']}, clahe_tile={p['clahe_tile']},
        gamma={p['gamma']}, retinex_sigma={p['retinex_sigma']})"""
        else:
            light_block = "\n    adapted = img_rgb"

        # morph block
        morph_block = ""
        if p['morph_mode'] != "None" and p['morph_k'] > 1:
            morph_block = f"""
    ops = {{"Erode": cv2.MORPH_ERODE, "Dilate": cv2.MORPH_DILATE,
            "Open": cv2.MORPH_OPEN,   "Close": cv2.MORPH_CLOSE}}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ({p['morph_k']}, {p['morph_k']}))
    gray_mask = cv2.morphologyEx(gray_mask, ops["{p['morph_mode']}"], kernel)"""

        # color mask block
        color_block = ""
        if p['color_mask']:
            cc = p['color_combine']
            op = {"AND": "cv2.bitwise_and(gray_mask, color_mask)",
                  "OR":  "cv2.bitwise_or(gray_mask, color_mask)",
                  "REPLACE gray": "color_mask",
                  "REPLACE color": "cv2.bitwise_and(color_mask, gray_mask)"}[cc]
            color_block = f"""
    color_mask = apply_color_mask(adapted, "{p['color_space']}",
        {p['c1_lo']}, {p['c1_hi']}, {p['c2_lo']}, {p['c2_hi']},
        {p['c3_lo']}, {p['c3_hi']}, density_weight={p['density_weight']})
    mask = {op}"""
        else:
            color_block = "\n    mask = gray_mask"

        # double mask block
        double_block = ""
        if p['double_mask'] and p['roi']:
            x1, y1, x2, y2 = p['roi']
            combine = p['combine_mode']
            op2 = {"REPLACE": "roi_mask",
                   "AND":     "cv2.bitwise_and(g, roi_mask)",
                   "OR":      "cv2.bitwise_or(g, roi_mask)",
                   "XOR":     "cv2.bitwise_xor(g, roi_mask)"}[combine]
            double_block = f"""
    x1, y1, x2, y2 = {x1}, {y1}, {x2}, {y2}
    roi_gray = gray[y1:y2, x1:x2]
    k2 = {p['blur_k2']}
    roi_blurred = cv2.GaussianBlur(roi_gray, (k2,k2), 0) if k2 > 1 else roi_gray.copy()
    roi_mask = apply_threshold(roi_blurred, "{p['thresh_mode2']}", {p['thresh_low2']}, {p['thresh_high2']}, {p['block_size2']})
    g = mask[y1:y2, x1:x2]
    mask[y1:y2, x1:x2] = {op2}"""

        # result block
        if p['show_mode'] == "Mask only":
            result_block = "result = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)"
        elif p['show_mode'] == "Masked image":
            result_block = "result = cv2.bitwise_and(img_rgb, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))"
        else:
            result_block = (f"mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)\n"
                            f"    orig_f = img_rgb.astype(np.float32)/255.0\n"
                            f"    mask_f = mask_rgb.astype(np.float32)/255.0\n"
                            f"    result = ((orig_f*{p['alpha']} + mask_f*{1-p['alpha']:.4f}).clip(0,1)*255).astype(np.uint8)")

        code = f'''"""Auto-generated mask pipeline — Mask Generator snapshot"""
import cv2
import numpy as np


def clamp_odd(v, lo, hi):
    v = max(lo, min(int(v), hi))
    return v if v % 2 == 1 else (v - 1 if v > lo else v + 1)


def apply_threshold(gray, mode, low, high, block_size=11):
    if mode == "IN_RANGE":   return cv2.inRange(gray, int(low), int(high))
    if mode == "BINARY":
        _, m = cv2.threshold(gray, int(low), int(high), cv2.THRESH_BINARY); return m
    if mode == "BINARY_INV":
        _, m = cv2.threshold(gray, int(low), int(high), cv2.THRESH_BINARY_INV); return m
    if mode == "OTSU":
        _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU); return m
    bs = clamp_odd(max(3, block_size), 3, 255)
    method = cv2.ADAPTIVE_THRESH_MEAN_C if mode=="ADAPTIVE_MEAN" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    return cv2.adaptiveThreshold(gray, 255, method, cv2.THRESH_BINARY, bs, int(low)//10)


def apply_light_adaptation(rgb, mode, clahe_clip=2.0, clahe_tile=8, gamma=1.0, retinex_sigma=80):
    if mode == "None": return rgb
    if mode == "Gamma":
        lut = (255.*(np.arange(256)/255.)**(1./max(gamma,.05))).astype(np.uint8)
        return cv2.LUT(rgb, lut)
    if mode == "Gray World":
        r = rgb.astype(np.float32); means = r.mean(axis=(0,1)); t = means.mean()
        for c in range(3):
            if means[c]>0: r[:,:,c] *= t/means[c]
        return r.clip(0,255).astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB); l = lab[:,:,0]
    if mode == "Hist-EQ":   lab[:,:,0] = cv2.equalizeHist(l)
    elif mode == "CLAHE":
        cl = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(int(clahe_tile),int(clahe_tile)))
        lab[:,:,0] = cl.apply(l)
    elif mode == "LAB-norm":
        mn,mx = l.min(),l.max()
        if mx>mn: lab[:,:,0] = ((l.astype(np.float32)-mn)/(mx-mn)*255).astype(np.uint8)
    elif mode == "Retinex-SSR":
        lf = l.astype(np.float32)+1.; bl = cv2.GaussianBlur(lf,(0,0),retinex_sigma)
        r2 = np.log(lf)-np.log(bl+1e-6); r2 -= r2.min()
        mx = r2.max()
        lab[:,:,0] = (r2/mx*255).astype(np.uint8) if mx>0 else np.zeros_like(l)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def apply_color_mask(rgb, space, c1_lo, c1_hi, c2_lo, c2_hi, c3_lo, c3_hi, density_weight=False):
    if space=="HSV":  img = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    elif space=="LAB": img = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    else: img = rgb.copy()
    lo = np.array([int(c1_lo),int(c2_lo),int(c3_lo)], np.uint8)
    hi = np.array([int(c1_hi),int(c2_hi),int(c3_hi)], np.uint8)
    if space=="HSV" and int(c1_lo)>int(c1_hi):
        m1 = cv2.inRange(img, np.array([int(c1_lo),int(c2_lo),int(c3_lo)],np.uint8),
                              np.array([179,int(c2_hi),int(c3_hi)],np.uint8))
        m2 = cv2.inRange(img, np.array([0,int(c2_lo),int(c3_lo)],np.uint8),
                              np.array([int(c1_hi),int(c2_hi),int(c3_hi)],np.uint8))
        mask = cv2.bitwise_or(m1,m2)
    else:
        mask = cv2.inRange(img, lo, hi)
    if density_weight:
        if space=="HSV": w = img[:,:,1].astype(np.float32)/255.
        elif space=="LAB":
            a=img[:,:,1].astype(np.float32)-128; b=img[:,:,2].astype(np.float32)-128
            ch=np.sqrt(a**2+b**2); w=ch/(ch.max()+1e-6)
        else:
            r,g,b2 = img[:,:,0].astype(np.float32),img[:,:,1].astype(np.float32),img[:,:,2].astype(np.float32)
            m=(r+g+b2)/3; ch=np.sqrt((r-m)**2+(g-m)**2+(b2-m)**2); w=ch/(ch.max()+1e-6)
        weighted=(mask.astype(np.float32)/255.*w*255).clip(0,255).astype(np.uint8)
        _,mask=cv2.threshold(weighted,50,255,cv2.THRESH_BINARY)
    return mask


def process(image_path: str) -> np.ndarray:
    bgr = cv2.imread(image_path)
    if bgr is None: raise FileNotFoundError(image_path)
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
{light_block}
    gray = cv2.cvtColor(adapted, cv2.COLOR_RGB2GRAY)
    blur_k = {p['blur_k']}
    blurred = cv2.GaussianBlur(gray,(blur_k,blur_k),0) if blur_k>1 else gray.copy()
    gray_mask = apply_threshold(blurred, "{p['thresh_mode']}", {p['thresh_low']}, {p['thresh_high']}, {p['block_size']}){morph_block}{color_block}{double_block}
    {result_block}
    return result


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv)>1 else "image.png"
    out = process(path)
    cv2.imwrite("result.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print("Saved result.png")
'''
        return code

    def _generate_code(self):
        code = self._build_code(self._collect_params())
        self.code_text.delete("1.0", "end")
        self.code_text.insert("1.0", code)

    def _copy_code(self):
        code = self.code_text.get("1.0", "end").strip()
        if not code:
            self._generate_code()
            code = self.code_text.get("1.0", "end").strip()
        self.clipboard_clear()
        self.clipboard_append(code)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = MaskGeneratorApp()
    app.mainloop()
