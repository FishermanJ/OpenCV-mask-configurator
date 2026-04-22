# Mask Generator (OpenCV)

Interactive image masking playground built with Tkinter + OpenCV.
Tune every parameter live and see the result instantly.

---

## Quick start

```bash
pip install opencv-python pillow numpy
python mask_generator_opencv.py
```

---

## Window layout

```
┌──────────────────────────────────────────┬──────────────────────────┐
│                                          │  FILE                    │
│   Original image                         │  ▸ Load Image            │
│   (click + drag → draw ROI)              │  ▸ Export result as PNG  │
│   (right-click  → clear ROI)             ├──────────────────────────┤
│                                          │  LIGHT ADAPTATION        │
│  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─┐                   │  BLUR                    │
│    ROI selection      │ ←cyan dashed box │  GLOBAL MASK (grayscale) │
│  └ ─ ─ ─ ─ ─ ─ ─ ─ ─┘                   │  MORPHOLOGY              │
│                                          │  COLOR MASK              │
├──────────────────────────────────────────│    color range preview   │
│                                          │  DOUBLE MASK (ROI)       │
│   Result image                           │  CONTOURS                │
│   (blended / mask only / masked)         │  BLEND & DISPLAY         │
│                                          │  EXPORT CODE             │
│                                          │   └ text box (copyable)  │
└──────────────────────────────────────────┴──────────────────────────┘
```

---

## Pipeline (applied in order)

```
Original image
    │
    ▼
[1] Light adaptation      ← fix uneven lighting / shadows
    │
    ▼
[2] Gaussian blur         ← smooth before thresholding
    │
    ▼
[3] Grayscale mask        ← IN_RANGE / BINARY / OTSU / ADAPTIVE
    │
    ▼
[4] Morphology            ← Erode / Dilate / Open / Close
    │
    ▼
[5] Color mask  ──────────── HSV / RGB / LAB range filter
    │               (AND / OR / REPLACE with grayscale mask)
    ▼
[6] Double mask (ROI)     ← second threshold inside a drawn region
    │
    ▼
[7] Contours              ← find + draw edges, filter by min area
    │
    ▼
[8] Result                ← Blended α | Mask only | Masked image
```

---

## Light adaptation modes

| Mode | What it does |
|---|---|
| **None** | Pass-through |
| **CLAHE** | Local contrast boost (great for shadows). Tune *clip limit* and *tile size* |
| **Hist-EQ** | Global histogram equalisation |
| **Retinex-SSR** | Single-scale Retinex — subtracts smooth illumination component |
| **Gamma** | Power-law correction. `< 1` brightens, `> 1` darkens |
| **Gray World** | White-balance by forcing per-channel mean to be equal |
| **LAB-norm** | Stretches the L channel to full 0–255 range |

Visual effect of light adaptation on a shadowed image:

```
Before CLAHE                     After CLAHE
┌────────────────┐               ┌────────────────┐
│ ░░░▓▓▓▓▓▓████ │               │ ░░▒▒▓▓▓▓████▓▓ │
│ ░░░░░▓▓▓▓████ │  ──────────►  │ ░░░▒▒▒▓▓▓████▓ │
│ ░░░░░░░░▓▓███ │               │ ░░░░░▒▒▒▒▓▓▓▓▓ │
│ ░░░░░░░░░░░░░ │               │ ░░░▒▒▒▒▒▒▒▒▒▒▒ │
└────────────────┘               └────────────────┘
  shadow clips detail              details recovered
```

---

## Global mask — threshold modes

| Mode | Use when |
|---|---|
| **IN_RANGE** | Keep pixels whose intensity is *between* Low and High |
| **BINARY** | Pixels *above* Low → set to High value |
| **BINARY_INV** | Pixels *below* Low → set to High value |
| **OTSU** | Auto-pick threshold (ignores Low/High sliders) |
| **ADAPTIVE_MEAN** | Local threshold per tile — handles patchy lighting |
| **ADAPTIVE_GAUSSIAN** | Same but Gaussian-weighted tiles — smoother edges |

---

## Color mask

Filter by color instead of brightness. Supports three color spaces:

### HSV  *(recommended for isolating specific colors)*
```
H  0 ──────────────────────────── 179
   red  orange  yellow  green  cyan  blue  magenta  red
S  0 ──────── 255        (0 = grey, 255 = fully saturated)
V  0 ──────── 255        (0 = black, 255 = full brightness)
```

### Color range visualizer (live preview in the panel)

```
Top strip — channel 1 gradient (Hue shown here):
┌──┬──┬──┬──┬──┬────────────────────┬──┬──┬──┬──┬──┐
│░░│░░│██│██│██│  selected range    │██│██│░░│░░│░░│
└──┴──┴──┴──┴──┴────────────────────┴──┴──┴──┴──┴──┘
                 ↑ yellow markers                    

2D grid — Hue × Saturation (V fixed at mid-V):
┌───────────────────────────────────────┐
│S                                      │  bright = in-range
│▲ ░░░░░░░████████████░░░░░░░░░░░░░░░  │  dim   = out-of-range
││ ░░░░░░░████████████░░░░░░░░░░░░░░░  │
││ ░░░░░░░████████████░░░░░░░░░░░░░░░  │  yellow box = selection
││ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│└─────────────────────────────────── H►│
└───────────────────────────────────────┘
```

### Click-to-pick color
1. Click **"Pick color from image [OFF]"** → button turns purple and shows **[ON]**
2. Click any pixel on the original image
3. Sliders snap to that color ± *Pick tolerance* value
4. A coloured swatch shows the picked RGB + converted value
5. Color mask is auto-enabled

---

## Double mask (ROI)

Apply a **second, independent threshold** inside a hand-drawn region:

```
Step 1 — draw ROI on left canvas:          Step 2 — result:
┌─────────────────────────┐                ┌─────────────────────────┐
│                         │                │  global mask everywhere │
│   ┌──────────────┐      │                │                         │
│   │  ROI region  │      │   ────────►    │  ┌──────────────┐       │
│   │  (cyan dash) │      │                │  │  local mask  │       │
│   └──────────────┘      │                │  └──────────────┘       │
│                         │                │       (cyan border)     │
└─────────────────────────┘                └─────────────────────────┘

Combine modes:
  REPLACE  — ROI area uses only the local mask
  AND      — pixel must pass BOTH global AND local mask
  OR       — pixel passes if EITHER mask accepts it
  XOR      — pixel passes if exactly ONE mask accepts it
```

---

## Export

| Action | How |
|---|---|
| **Save PNG / JPG / BMP** | Click *Export result as PNG* → file dialog → saves at **original image resolution** (no downscaling) |
| **Copy pipeline code** | Click *Generate Python Code*, then *Copy to Clipboard* — pastes a self-contained `process(image_path)` function with all current settings baked in |

---

## Requirements

| Package | Version tested |
|---|---|
| Python | 3.9 + |
| opencv-python | 4.x |
| Pillow | 10.x |
| numpy | 1.24 + |
