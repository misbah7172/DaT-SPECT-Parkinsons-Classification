"""V24 CNN inference: rotation alignment + translation+rotation TTA.

Changes from v23:
- Coarse rotation search (±5° in 2° steps) before NCC translation matching
- Rotation TTA: base + 6 translation rolls + 6 rotation views = 13 views/model
- Supports 3/3 architecture split (3 Net3dR + 3 Net3dBig) for diversity
- Input: (1, 32, 40, 48) at 2.0mm isotropic
"""
import os
import numpy as np
import scipy.ndimage as ndi

TARGET = 2.0
GRID = 80
CENTER = (GRID - 1) / 2.0
BX = (0, 80, 0, 80, 0, 80)  # Full 80^3 grid (training used full 80^3)

# Rotation search parameters
ROT_STEP = 2.0  # degrees
ROT_MAX = 5.0   # ±5 degrees
ROT_ANGLES = np.arange(-ROT_MAX, ROT_MAX + ROT_STEP, ROT_STEP)


def _reorient_ras(data, affine):
    from nibabel.orientations import apply_orientation, axcodes2ornt, io_orientation, ornt_transform
    current = io_orientation(affine)
    target = axcodes2ornt("RAS")
    transform = ornt_transform(current, target)
    return apply_orientation(data, transform)


def load_aligned(nii_path):
    """Load NIfTI, RAS reorient, resample to 2.0mm isotropic, COM-center on 80³ grid."""
    import nibabel as nib
    img = nib.load(str(nii_path))
    data = _reorient_ras(img.get_fdata().astype(np.float32), img.affine)
    zooms = np.array(img.header.get_zooms()[:3], dtype=np.float64)
    factor = zooms / TARGET
    shape = tuple(max(1, int(round(s * f))) for s, f in zip(data.shape, factor))
    zf = [n / s for s, n in zip(data.shape, shape)]
    data = ndi.zoom(data, zoom=zf, order=1)
    head = data > np.percentile(data, 5)
    com = np.array(ndi.center_of_mass(head)) if head.any() else np.array(data.shape) / 2.0
    offset = com - CENTER
    return ndi.affine_transform(
        data, matrix=np.eye(3), offset=offset, output_shape=(GRID, GRID, GRID),
        order=1, mode="constant", cval=0.0)


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


def _rotate_volume(volume, angle_deg, axes=(1, 2)):
    """Rotate volume around specified axes by angle_degrees."""
    return ndi.rotate(volume, angle_deg, axes=axes, reshape=False, order=1, mode='constant', cval=0.0)


class TemplateCache:
    """Precompute shifted+rotated template windows for alignment."""

    def __init__(self, template):
        sm_t = template[BX[0]:BX[1], BX[2]:BX[3], BX[4]:BX[5]].astype(np.float64)
        self.windows = []
        self.shifts = []

        # Translation windows (±4 voxels)
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                for dz in range(-4, 5):
                    tw = ndi.shift(sm_t, (-dx, -dy, -dz), order=1)
                    if tw.sum():
                        self.windows.append(tw)
                        self.shifts.append((dx, dy, dz))

        # Precompute rotated template windows for rotation search
        self.rot_windows = []
        for angle in ROT_ANGLES:
            for axes in [(1, 2), (0, 2), (0, 1)]:  # sagittal, coronal, axial
                rw = _rotate_volume(sm_t, angle, axes=axes)
                self.rot_windows.append((angle, axes, rw))

    def best_shift(self, aligned):
        """Find best translation alignment via NCC."""
        sm = ndi.gaussian_filter(aligned, sigma=1.0)
        W = sm[BX[0]:BX[1], BX[2]:BX[3], BX[4]:BX[5]]
        m = W > 0
        if m.sum() < 100:
            return 0, 0, 0
        best = ((0, 0, 0), -1)
        for tw, sh in zip(self.windows, self.shifts):
            if tw[m].size == 0:
                continue
            c = _ncc(W[m], tw[m])
            if c > best[1]:
                best = (sh, c)
        return best[0]

    def best_rotation(self, aligned):
        """Find best rotation alignment via NCC (coarse search)."""
        sm = ndi.gaussian_filter(aligned, sigma=1.0)
        W = sm[BX[0]:BX[1], BX[2]:BX[3], BX[4]:BX[5]]
        m = W > 0
        if m.sum() < 100:
            return 0, (1, 2)

        best = (0, (1, 2), -1)
        for angle, axes, rw in self.rot_windows:
            if rw[m].size == 0:
                continue
            c = _ncc(W[m], rw[m])
            if c > best[2]:
                best = (angle, axes, c)
        return best[0], best[1]


def to_model_input(nii_path, template, cache=None):
    """Full alignment pipeline: RAS → resample → COM-center → rotation → translation → crop."""
    aligned = load_aligned(nii_path)
    if cache is None:
        cache = TemplateCache(template)

    # Step 1: Coarse rotation alignment
    angle, axes = cache.best_rotation(aligned)
    if abs(angle) > 0:
        aligned = _rotate_volume(aligned, angle, axes=axes)

    # Step 2: Translation alignment via NCC
    dx, dy, dz = cache.best_shift(aligned)
    moved = ndi.shift(aligned, (dx, dy, dz), order=1)

    # Step 3: Crop and normalize
    w = moved[BX[0]:BX[1], BX[2]:BX[3], BX[4]:BX[5]]
    v = w[w > 0]
    if v.size:
        lo, hi = np.percentile(v, 1), np.percentile(v, 99)
        if hi > lo:
            w = (w - lo) / (hi - lo)
    return w.astype(np.float32)


def to_model_input_rotation(nii_path, template, angle_deg, axes, cache=None):
    """Extract crop with a specific rotation applied (for rotation TTA)."""
    aligned = load_aligned(nii_path)

    # Apply rotation
    if abs(angle_deg) > 0:
        aligned = _rotate_volume(aligned, angle_deg, axes=axes)

    if cache is None:
        cache = TemplateCache(template)

    dx, dy, dz = cache.best_shift(aligned)
    moved = ndi.shift(aligned, (dx, dy, dz), order=1)

    w = moved[BX[0]:BX[1], BX[2]:BX[3], BX[4]:BX[5]]
    v = w[w > 0]
    if v.size:
        lo, hi = np.percentile(v, 1), np.percentile(v, 99)
        if hi > lo:
            w = (w - lo) / (hi - lo)
    return w.astype(np.float32)


import torch
import torch.nn as tnn


class _ResBlock3(tnn.Module):
    def __init__(self, c, T=None):
        super().__init__()
        self.b1 = tnn.Sequential(tnn.Conv3d(c, c, 3, padding=1), tnn.BatchNorm3d(c), tnn.ReLU())
        self.b2 = tnn.Sequential(tnn.Conv3d(c, c, 3, padding=1), tnn.BatchNorm3d(c))
        self.relu = tnn.ReLU()

    def forward(self, x):
        return self.relu(x + self.b2(self.b1(x)))


class _DeepNet(tnn.Module):
    def __init__(self, net, head):
        super().__init__()
        self.net = net
        self.head = head

    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)


class DeepEnsemble:
    def __init__(self, weights_dir, device="cpu", allow=None):
        import torch
        self.torch = torch
        T = torch.nn
        self.T = T
        self.device = device
        self.models = []
        for f in sorted(os.listdir(weights_dir)):
            if f.startswith("deep_") and f.endswith(".pt"):
                if allow is not None and f not in allow:
                    continue
                ck = torch.load(os.path.join(weights_dir, f), map_location=device, weights_only=False)
                m = self._build(ck["arch"])
                m.load_state_dict(ck["state"])
                m.eval().to(device)
                self.models.append((ck["arch"], m))
        if not self.models:
            raise RuntimeError("no deep models found in " + weights_dir)

    def _block(self, c):
        T = self.T
        return _ResBlock3(c, T)

    def _build(self, arch):
        T = self.T
        if arch == "r":
            net = T.Sequential(
                T.Conv3d(1, 16, 3, padding=1), T.BatchNorm3d(16), T.ReLU(), T.MaxPool3d(2),
                T.Conv3d(16, 32, 3, padding=1), T.BatchNorm3d(32), T.ReLU(), T.MaxPool3d(2),
                _ResBlock3(32, T), _ResBlock3(32, T),
                T.Conv3d(32, 64, 3, padding=1), T.BatchNorm3d(64), T.ReLU(),
                T.AdaptiveAvgPool3d((2, 2, 2)))
            head = T.Sequential(T.Flatten(), T.Linear(64 * 8, 192), T.ReLU(), T.Dropout(0.4), T.Linear(192, 1))
        else:
            net = T.Sequential(
                T.Conv3d(1, 20, 3, padding=1), T.BatchNorm3d(20), T.ReLU(), T.MaxPool3d(2),
                T.Conv3d(20, 36, 3, padding=1), T.BatchNorm3d(36), T.ReLU(), T.MaxPool3d(2),
                _ResBlock3(36, T), _ResBlock3(36, T),
                T.Conv3d(36, 48, 3, padding=1), T.BatchNorm3d(48), T.ReLU(),
                T.AdaptiveAvgPool3d((2, 2, 2)))
            head = T.Sequential(T.Flatten(), T.Linear(48 * 8, 224), T.ReLU(),
                                T.Dropout(0.4), T.Linear(224, 1))
        return _DeepNet(net, head)

    def predict_one(self, x):
        """x: (32,40,48) float32 crop. Returns mean sigmoid across models (translation + rotation TTA)."""
        T = self.torch
        with T.no_grad():
            xin = T.from_numpy(x[None, None]).to(self.device)
            total = None
            for _arch, m in self.models:
                # Base view
                p = T.sigmoid(m(xin)).float()
                if total is None:
                    total = p
                else:
                    total = total + p

                # Translation TTA: ±1 voxel on X, Y, Z (6 views)
                for dx in (-1, 1):
                    xv = T.from_numpy(np.roll(x, dx, axis=0)[None, None]).to(self.device)
                    total = total + T.sigmoid(m(xv)).float()
                for dy in (-1, 1):
                    xv = T.from_numpy(np.roll(x, dy, axis=1)[None, None]).to(self.device)
                    total = total + T.sigmoid(m(xv)).float()
                for dz in (-1, 1):
                    xv = T.from_numpy(np.roll(x, dz, axis=2)[None, None]).to(self.device)
                    total = total + T.sigmoid(m(xv)).float()

                # Rotation TTA: ±2°, ±4° on 2 axes (8 views)
                for angle in (-2.0, 2.0, -4.0, 4.0):
                    for axes in [(1, 2), (0, 2)]:
                        xr = _rotate_volume(x, angle, axes=axes).astype(np.float32)
                        xrv = T.from_numpy(xr[None, None]).to(self.device)
                        total = total + T.sigmoid(m(xrv)).float()

            # 1 base + 6 translation + 8 rotation = 15 views per model
            n_views = 15
            return float((total / (len(self.models) * n_views)).cpu().numpy()[0])
