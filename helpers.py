"""Shared code for the robustness notebook (mvtec_ad_robustness.ipynb).

Copied and simplified from mvtec_ad_comparison.ipynb so that the first notebook stays untouched. Only the
nearest-neighbour method (Wide ResNet50 layer 2, border margin) and the evaluation helpers are here.
Images are kept per category, as uint8 at RESOLUTION x RESOLUTION; the feature functions accept uint8 or float images.
"""
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

RESOLUTION = 512  # images are stored at this size (original is 1024x1024), as in the first notebook
CATEGORIES = ["transistor", "pill", "metal_nut", "leather", "carpet", "grid"]
GRID = 28         # layer 2 of the Wide ResNet50 gives a 28 x 28 grid of patch descriptions at 224 px input
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# --- data ---------------------------------------------------------------------------------------------------------

def data_root():
    """Folder with one subfolder per category; set MVTEC_AD_ROOT or edit the default."""
    return Path(os.environ.get("MVTEC_AD_ROOT", "datasets/mvtec_ad_dataset"))


@dataclass
class Category:
    name: str
    X: np.ndarray       # (N, RESOLUTION, RESOLUTION, 3) uint8, RGB
    labels: np.ndarray  # "good" or the defect type
    splits: np.ndarray  # "train" or "test"


def load_category(name, root=None):
    """All train and test images of one category (the ground-truth masks are skipped)."""
    root = Path(root) if root else data_root()
    X, labels, splits = [], [], []
    for img_path in (root / name).rglob("*.png"):
        label, split = img_path.parent.name, img_path.parent.parent.name
        if split == "ground_truth":
            continue
        img = cv2.imread(str(img_path))  # BGR
        if img is None:
            continue  # skip unreadable files
        X.append(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (RESOLUTION, RESOLUTION)))
        labels.append(label)
        splits.append(split)
    if not X:
        raise FileNotFoundError(f"no images found under {(root / name).resolve()}; set MVTEC_AD_ROOT")
    return Category(name, np.array(X), np.array(labels), np.array(splits))


def test_labels(name, root=None):
    """Labels ("good" or the defect type) of the test images of one category, in the order of select(cat, "test"),
    read from the folder names without loading the images."""
    root = Path(root) if root else data_root()
    return np.array([p.parent.name for p in (root / name).rglob("*.png") if p.parent.parent.name == "test"])


def select(cat, split, good=None):
    """Indices into cat.X for one split. good=True/False keeps only good/defective images."""
    m = cat.splits == split
    if good is not None:
        m &= (cat.labels == "good") == good
    return np.where(m)[0]


def held_out_split(cat, seed=0):
    """A fifth of the good training images is held out (threshold and border margin), the rest fits the model.
    The same split as in the first notebook."""
    tr = select(cat, "train", good=True)
    perm = np.random.default_rng(seed).permutation(len(tr))
    n_val = len(tr) // 5
    return tr[perm[:n_val]], tr[perm[n_val:]]  # val_idx, fit_idx


# --- nearest-neighbour method -------------------------------------------------------------------------------------

_backbone = None


def backbone():
    global _backbone
    if _backbone is None:
        _backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).eval()
    return _backbone


def to_float(im):
    return im.astype("float32") / 255.0 if im.dtype == np.uint8 else im.astype("float32")


@torch.inference_mode()
def cnn_features(imgs, bs=16):
    """RGB images (uint8 or float in 0..1) -> layer-2 feature map per image, shape (N, 28, 28, 512)."""
    b = backbone()
    out = []
    for s in range(0, len(imgs), bs):
        x = np.stack([cv2.resize(to_float(im), (224, 224), interpolation=cv2.INTER_AREA) for im in imgs[s:s + bs]])
        x = (torch.from_numpy(x).permute(0, 3, 1, 2) - IMNET_MEAN) / IMNET_STD
        x = b.layer2(b.layer1(b.maxpool(b.relu(b.bn1(b.conv1(x))))))
        out.append(x.permute(0, 2, 3, 1).numpy())
    return np.concatenate(out)


def build_knn_memory(cat, idx=None, max_patches=40000, seed=0):
    """Memory bank: patch features of good training images (a random subset of the patches, to keep the search fast).
    idx: which images; default all good training images."""
    idx = select(cat, "train", good=True) if idx is None else idx
    f = cnn_features(cat.X[idx]).reshape(-1, 512)
    keep = np.random.default_rng(seed).choice(len(f), size=min(max_patches, len(f)), replace=False)
    return torch.from_numpy(f[keep])


@torch.inference_mode()
def knn_maps(memory, imgs):
    """Distance of every patch to its nearest good patch, shape (N, 28, 28)."""
    feats = cnn_features(imgs)
    return np.array([torch.cdist(torch.from_numpy(f.reshape(-1, 512)), memory).min(dim=1).values.reshape(GRID, GRID).numpy()
                     for f in feats])


def border_scores(maps, margin):
    """Image score = worst patch, ignoring `margin` rings of patches at the border."""
    g = maps.shape[1]
    return maps[:, margin:g - margin, margin:g - margin].reshape(len(maps), -1).max(1)


def ring_profile(maps):
    """Mean distance of the held-out good images per ring from the border (0 = outermost ring)."""
    g = maps.shape[1]
    prof = []
    for r in range(g // 2):
        mask = np.zeros((g, g), bool)
        mask[r:g - r, r:g - r] = True
        if r + 1 < g // 2:
            mask[r + 1:g - r - 1, r + 1:g - r - 1] = False
        prof.append(maps[:, mask].mean())
    return np.array(prof)


@dataclass
class Reference:
    """The fitted nearest-neighbour model of one category, with everything fixed on good training images only."""
    memory: torch.Tensor   # fitted on the fit images
    margin: int            # border margin in patches
    val_scores: np.ndarray  # scores of the held-out good images (with the margin), the basis of every threshold

    def scores(self, imgs):
        return border_scores(knn_maps(self.memory, imgs), self.margin)

    def threshold(self, target_frr):
        return np.quantile(self.val_scores, 1 - target_frr, method="higher")


def fit_reference(cat, tol=1.05, max_frac=0.1, max_patches=40000, seed=0):
    """Memory from 80% of the good training images; border margin and threshold basis from the other 20%.
    The margin is the first ring within `tol` of the interior level, at most `max_frac` of the grid size."""
    val_idx, fit_idx = held_out_split(cat, seed)
    memory = build_knn_memory(cat, fit_idx, max_patches, seed)
    val_maps = knn_maps(memory, cat.X[val_idx])
    prof = ring_profile(val_maps)
    interior = np.median(prof[len(prof) // 2:])
    margin = min(int(np.argmax(prof <= tol * interior)), int(max_frac * GRID))
    return Reference(memory, margin, border_scores(val_maps, margin))


# --- evaluation ---------------------------------------------------------------------------------------------------

def operating_point(scores_val, scores_good, scores_defect, target_frr):
    """Threshold from held-out good images only. Returns the share of defects caught and the false-reject rate on the good test images."""
    thr = np.quantile(scores_val, 1 - target_frr, method="higher")
    return (scores_defect > thr).mean(), (scores_good > thr).mean()


def bootstrap_auroc(y, s, n=1000, seed=0):
    """95% interval of the AUROC: resample the good and the defective images separately, with replacement."""
    rng = np.random.default_rng(seed)
    good, defect = s[y == 0], s[y == 1]
    labels_b = np.r_[np.zeros(len(good)), np.ones(len(defect))]
    out = [roc_auc_score(labels_b, np.r_[rng.choice(good, len(good)), rng.choice(defect, len(defect))]) for _ in range(n)]
    return np.percentile(out, [2.5, 97.5])


# --- simulated variations -----------------------------------------------------------------------------------------
# Each variation is a function (float RGB image in 0..1, rng) -> changed image. The rng is seeded per image (see
# score_augmented), so the same image gets the same random draw at every strength. These are simulations, not real lots.

def _clip(im):
    return np.clip(im, 0.0, 1.0).astype("float32")


def _ramp(a):
    """Uneven lighting: brightness changes linearly across the image, in a random direction, by +-a at the corners."""
    def f(im, rng):
        theta = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[:im.shape[0], :im.shape[1]].astype("float32")
        t = ((xx / im.shape[1] - 0.5) * np.cos(theta) + (yy / im.shape[0] - 0.5) * np.sin(theta)) * 2
        t /= abs(np.cos(theta)) + abs(np.sin(theta))
        return _clip(im * (1 + a * t)[..., None])
    return f


def _colour(d):
    """Colour cast: red up and blue down by a share d (negative d: blue up and red down)."""
    return lambda im, rng: _clip(im * np.array([1 + d, 1, 1 - d], dtype="float32"))


def _contrast(c):
    """Contrast: the values are stretched (c > 1) or squeezed (c < 1) around the mean of the image."""
    def f(im, rng):
        m = im.mean()
        return _clip(m + c * (im - m))
    return f


def _blur(sigma):
    return lambda im, rng: cv2.GaussianBlur(im, (0, 0), sigma)


def _noise(sigma):
    return lambda im, rng: _clip(im + rng.normal(0, sigma, im.shape).astype("float32"))


def _flicker(a):
    """A different gain for every image, between 1 - a and 1 + a."""
    return lambda im, rng: _clip(im * rng.uniform(1 - a, 1 + a))


def _blobs(centres, radii, depth, shape):
    """Product of soft dark spots: a multiplier map with values <= 1."""
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    mult = np.ones(shape[:2], "float32")
    for (cy, cx), r in zip(centres, radii):
        mult *= 1 - depth * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (r / 2) ** 2)).astype("float32")
    return mult


_LENS_SPOTS = None


def _lens_spots(n):
    """Dirt on the lens: the same n soft dark spots in every image (the first n of a fixed set of eight)."""
    global _LENS_SPOTS
    if _LENS_SPOTS is None:
        rng = np.random.default_rng(12345)
        _LENS_SPOTS = (rng.uniform(0.1, 0.9, (8, 2)) * RESOLUTION, rng.uniform(10, 25, 8))
    mult = _blobs(_LENS_SPOTS[0][:n], _LENS_SPOTS[1][:n], 0.4, (RESOLUTION, RESOLUTION))[..., None]
    return lambda im, rng: _clip(im * mult)


def _contamination(n):
    """Harmless dust or oil: n soft dark spots at random places, different in every image. Not a defect."""
    def f(im, rng):
        centres = rng.uniform(0.05, 0.95, (n, 2)) * im.shape[0]
        radii = rng.uniform(8, 20, n)
        return _clip(im * _blobs(centres, radii, rng.uniform(0.25, 0.4), im.shape)[..., None])
    return f


# family -> (title, [(level label, parameter)], factory)
FAMILIES = {
    "darker":        ("Lighting: darker (gain)", [("0.85", 0.85), ("0.7", 0.7), ("0.5", 0.5)], lambda p: (lambda im, rng: _clip(im * p))),
    "brighter":      ("Lighting: brighter (gain)", [("1.15", 1.15), ("1.3", 1.3), ("1.5", 1.5)], lambda p: (lambda im, rng: _clip(im * p))),
    "contrast_down": ("Contrast: lower", [("0.8", 0.8), ("0.6", 0.6), ("0.4", 0.4)], _contrast),
    "contrast_up":   ("Contrast: higher", [("1.25", 1.25), ("1.5", 1.5), ("2.0", 2.0)], _contrast),
    "uneven":        ("Uneven lighting (ramp)", [("10%", 0.1), ("20%", 0.2), ("30%", 0.3)], _ramp),
    "colour":        ("Colour cast: warm (red up, blue down)", [("5%", 0.05), ("10%", 0.1), ("20%", 0.2)], _colour),
    "colour_cool":   ("Colour cast: cool (blue up, red down)", [("5%", -0.05), ("10%", -0.1), ("20%", -0.2)], _colour),
    "blur":          ("Blur (sigma in px at 512)", [("1", 1), ("2", 2), ("4", 4)], _blur),
    "noise":         ("Sensor noise (sigma)", [("0.01", 0.01), ("0.02", 0.02), ("0.04", 0.04)], _noise),
    "flicker":       ("Flicker (random gain per image)", [("+-10%", 0.1), ("+-20%", 0.2), ("+-40%", 0.4)], _flicker),
    "lens_spots":    ("Dirt on the lens (same spots in every image)", [("2", 2), ("4", 4), ("8", 8)], _lens_spots),
    "contamination": ("Harmless dust or oil (random spots)", [("1", 1), ("2", 2), ("4", 4)], _contamination),
}


def _shift(px):
    """The part sits a little off: a shift by px pixels (at RESOLUTION) in a random direction, the border mirrored."""
    def f(im, rng):
        a = rng.uniform(0, 2 * np.pi)
        m = np.float32([[1, 0, px * np.cos(a)], [0, 1, px * np.sin(a)]])
        return cv2.warpAffine(im, m, im.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return f


def _rotate(deg):
    """The part sits a little turned: a rotation by deg degrees about the centre, to the left or right, the border mirrored."""
    def f(im, rng):
        m = cv2.getRotationMatrix2D((im.shape[1] / 2, im.shape[0] / 2), deg * rng.choice([-1, 1]), 1.0)
        return cv2.warpAffine(im, m, im.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return f


# geometry changes of the part, kept apart from FAMILIES (a separate experiment)
GEOMETRY = {
    "shift":  ("Shift (px at 512, random direction)", [("4", 4), ("12", 12), ("32", 32)], _shift),
    "rotate": ("Rotation (degrees, random sign)", [("1", 1), ("3", 3), ("8", 8)], _rotate),
}


def family(name):
    """(title, [(level label, parameter)], factory) of a variation family from FAMILIES or GEOMETRY."""
    return FAMILIES[name] if name in FAMILIES else GEOMETRY[name]


def make_variation(name, param):
    return family(name)[2](param)


def score_augmented(ref, X, idx, fn, seed=0, chunk=32):
    """Scores of the images X[idx] after the variation fn, with the unchanged model. Random draws are seeded per image index."""
    out = []
    for s in range(0, len(idx), chunk):
        imgs = [fn(to_float(X[i]), np.random.default_rng([seed, int(i)])) for i in idx[s:s + chunk]]
        out.append(ref.scores(imgs))
    return np.concatenate(out)


# --- classical golden template ------------------------------------------------------------------------------------
# The template method of the first notebook (per-pixel mean and std of the aligned good images, ECC alignment, 512 px).
# It has the same interface as Reference, so the batch runner below works for both.

BLUR, PCTL, STD_FLOOR = 7, 99.5, 1e-3  # as in the first notebook


def _gray(im):
    return cv2.cvtColor(to_float(im), cv2.COLOR_RGB2GRAY)


def _align_ecc(gray, template, n_iter=100, eps=1e-5):
    """Rotation + translation alignment of `gray` onto `template`."""
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, eps)
    try:
        _, warp = cv2.findTransformECC(template, gray, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5)
    except cv2.error:
        return gray
    return cv2.warpAffine(gray, warp, template.shape[::-1], flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT)


@dataclass
class TemplateReference:
    mean: np.ndarray
    std: np.ndarray
    tmpl: np.ndarray
    val_scores: np.ndarray = None  # scores of the held-out good images

    def scores(self, imgs):
        out = []
        for im in imgs:
            g = _align_ecc(_gray(im), self.tmpl)
            z = cv2.GaussianBlur(np.abs(g - self.mean) / self.std, (BLUR, BLUR), 0)
            out.append(np.percentile(z, PCTL))
        return np.array(out)

    def threshold(self, target_frr):
        return np.quantile(self.val_scores, 1 - target_frr, method="higher")


def fit_template_reference(cat, seed=0):
    """Golden template from 80% of the good training images, the other 20% give the threshold (same split as fit_reference)."""
    val_idx, fit_idx = held_out_split(cat, seed)
    train = [_gray(cat.X[i]) for i in fit_idx]
    tmpl = train[0]  # a sharp real image is a more stable alignment target than the blurry mean
    train = np.stack([_align_ecc(g, tmpl) for g in train])
    ref = TemplateReference(train.mean(0), train.std(0) + STD_FLOOR, tmpl)
    ref.val_scores = ref.scores(cat.X[val_idx])
    return ref


# --- batch runner -------------------------------------------------------------------------------------------------

def conditions(families=None, strongest_only=False):
    """The unchanged images and every variation at every strength: (family, level, function). families: FAMILIES (default),
    GEOMETRY or a merge of both. strongest_only: only the strongest level of every family."""
    families = FAMILIES if families is None else families
    return [("none", "-", lambda im, rng: im)] + [
        (fam, lab, make_variation(fam, param)) for fam, (_, levels, _) in families.items()
        for lab, param in (levels[-1:] if strongest_only else levels)]


def run_batch(refs, csv_path, scores_dir, targets=(0.05, 0.01), log=print, families=None, strongest_only=False):
    """Score all test images and the held-out good images under every condition, with the unchanged model in refs[category].
    Finished (condition, category) pairs are read from csv_path, so an interrupted run continues. Returns the results table."""
    csv_path, scores_dir = Path(csv_path), Path(scores_dir)
    scores_dir.mkdir(exist_ok=True)
    done = pd.read_csv(csv_path, dtype={"level": str}) if csv_path.exists() else pd.DataFrame(columns=["family", "level", "category"])
    done_keys = set(zip(done["family"], done["level"], done["category"]))
    rows = done.to_dict("records")
    conds = conditions(families, strongest_only)
    for ctg, ref in refs.items():
        todo = [c for c in conds if (c[0], c[1], ctg) not in done_keys]
        if not todo:
            continue
        cat = load_category(ctg)
        val_idx, _ = held_out_split(cat)
        idx = select(cat, "test")
        is_def = cat.labels[idx] != "good"
        store_path = scores_dir / f"{ctg}.npz"
        store = dict(np.load(store_path)) if store_path.exists() else {}
        store["is_defective"], store["ref_val"] = is_def, ref.val_scores
        t0 = time.perf_counter()
        for fam, lab, fn in todo:
            s = score_augmented(ref, cat.X, idx, fn)
            s_val = score_augmented(ref, cat.X, val_idx, fn)  # held-out good images changed the same way
            row = {"family": fam, "level": lab, "category": ctg, "auroc": roc_auc_score(is_def, s),
                   "good_median": float(np.median(s[~is_def]))}
            for t in targets:
                row[f"caught@{t:.0%}"], row[f"frr@{t:.0%}"] = operating_point(ref.val_scores, s[~is_def], s[is_def], t)
                row[f"recal_caught@{t:.0%}"], row[f"recal_frr@{t:.0%}"] = operating_point(s_val, s[~is_def], s[is_def], t)
            rows.append(row)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            store[f"{fam}__{lab}__test"], store[f"{fam}__{lab}__val"] = s, s_val
            np.savez(store_path, **store)
        log(f"{ctg:11s} {len(todo)} conditions in {time.perf_counter() - t0:.0f} s")
    return pd.DataFrame(rows)


# --- candidates for a more robust method --------------------------------------------------------------------------
# 1. the template with a per-image brightness normalisation (gain, offset and a linear shading fitted to the template),
# 2. the same with a coarse position estimate (phase correlation) before the ECC alignment,
# 3. the neural method on images aligned to the template first.

def _ecc_warp(gray, template, phase_init=False, n_iter=100, eps=1e-5):
    """2x3 warp (template coordinates -> image coordinates) from ECC. With phase_init, ECC is also started at the shift found by
    phase correlation, and the result with the higher correlation is kept (the phase start alone misleads ECC on images that are
    not shifted)."""
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, eps)

    def run(start):
        try:
            cc, warp = cv2.findTransformECC(template, gray, start.copy(), cv2.MOTION_EUCLIDEAN, criteria, None, 5)
            return cc, warp
        except cv2.error:
            return -1.0, start

    starts = [np.eye(2, 3, dtype=np.float32)]
    if phase_init:
        win = cv2.createHanningWindow(gray.shape[::-1], cv2.CV_32F)
        (dx, dy), _ = cv2.phaseCorrelate(template, gray, win)
        shifted = np.eye(2, 3, dtype=np.float32)
        shifted[0, 2], shifted[1, 2] = dx, dy
        starts.append(shifted)
    return max((run(s) for s in starts), key=lambda r: r[0])[1]


def _unwarp(im, warp):
    """Bring an image (grey or RGB) onto the template with a warp from _ecc_warp."""
    return cv2.warpAffine(im, warp, im.shape[1::-1], flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)


class _Photometric:
    """Fits a grey image to a template: image = a * template + b + c * x + d * y, and returns (image - b - c x - d y) / a."""

    def __init__(self, mean_img):
        hh, ww = mean_img.shape
        yy, xx = np.mgrid[:hh, :ww].astype("float32")
        self.xx, self.yy = xx / ww - 0.5, yy / hh - 0.5
        A = np.stack([mean_img.ravel(), np.ones(hh * ww, "float32"), self.xx.ravel(), self.yy.ravel()], 1).astype("float64")
        self.pinv = np.linalg.pinv(A)  # (4, pixels)

    def __call__(self, g):
        a, b, c, d = self.pinv @ g.ravel().astype("float64")
        if a < 0.05:
            a = 1.0  # a nearly black or flat image: do not divide by (almost) nothing
        return ((g - b - c * self.xx - d * self.yy) / a).astype("float32")


@dataclass
class NormalisedTemplateReference:
    mean: np.ndarray
    std: np.ndarray
    tmpl: np.ndarray
    phase_init: bool
    photo: _Photometric
    val_scores: np.ndarray = None

    def scores(self, imgs):
        out = []
        for im in imgs:
            g = _gray(im)
            g = self.photo(_unwarp(g, _ecc_warp(g, self.tmpl, self.phase_init)))
            z = cv2.GaussianBlur(np.abs(g - self.mean) / self.std, (BLUR, BLUR), 0)
            out.append(np.percentile(z, PCTL))
        return np.array(out)

    def threshold(self, target_frr):
        return np.quantile(self.val_scores, 1 - target_frr, method="higher")


def fit_normalised_template(cat, phase_init=False, seed=0):
    """Like fit_template_reference, but every image (also in training) is fitted to the mean image in brightness before the z-map."""
    val_idx, fit_idx = held_out_split(cat, seed)
    grays = [_gray(cat.X[i]) for i in fit_idx]
    tmpl = grays[0]
    aligned = np.stack([_unwarp(g, _ecc_warp(g, tmpl, phase_init)) for g in grays])
    photo = _Photometric(aligned.mean(0))
    normed = np.stack([photo(g) for g in aligned])
    ref = NormalisedTemplateReference(normed.mean(0), normed.std(0) + STD_FLOOR, tmpl, phase_init, photo)
    ref.val_scores = ref.scores(cat.X[val_idx])
    return ref


@dataclass
class AlignedReference:
    memory: torch.Tensor
    margin: int
    tmpl: np.ndarray
    phase_init: bool
    val_scores: np.ndarray = None

    def align(self, im):
        im = to_float(im)
        return _unwarp(im, _ecc_warp(_gray(im), self.tmpl, self.phase_init))

    def scores(self, imgs):
        return border_scores(knn_maps(self.memory, [self.align(im) for im in imgs]), self.margin)

    def threshold(self, target_frr):
        return np.quantile(self.val_scores, 1 - target_frr, method="higher")


def fit_aligned_reference(cat, phase_init=True, tol=1.05, max_frac=0.1, max_patches=40000, seed=0):
    """The neural method (fit_reference) with every image, in training too, aligned to the template first."""
    val_idx, fit_idx = held_out_split(cat, seed)
    ref = AlignedReference(None, 0, _gray(cat.X[fit_idx[0]]), phase_init)
    f = cnn_features([ref.align(cat.X[i]) for i in fit_idx]).reshape(-1, 512)
    keep = np.random.default_rng(seed).choice(len(f), size=min(max_patches, len(f)), replace=False)
    ref.memory = torch.from_numpy(f[keep])
    val_maps = knn_maps(ref.memory, [ref.align(cat.X[i]) for i in val_idx])
    prof = ring_profile(val_maps)
    ref.margin = min(int(np.argmax(prof <= tol * np.median(prof[len(prof) // 2:]))), int(max_frac * GRID))
    ref.val_scores = border_scores(val_maps, ref.margin)
    return ref


# --- other feature networks for the nearest-neighbour method -----------------------------------------------------
# Same method as fit_reference (memory of good patches, distance of every patch to its nearest good patch, border margin from
# held-out good images), with the features of another pretrained network.

class FeatureNet:
    """Patch features of a pretrained network: __call__(images) -> (N, g, g, C). Subclasses set `name` and implement `_forward`."""
    name = ""
    size = 224

    def __call__(self, imgs, bs=16):
        out = []
        with torch.inference_mode():
            for s in range(0, len(imgs), bs):
                x = np.stack([cv2.resize(to_float(im), (self.size, self.size), interpolation=cv2.INTER_AREA) for im in imgs[s:s + bs]])
                x = (torch.from_numpy(x).permute(0, 3, 1, 2) - IMNET_MEAN) / IMNET_STD
                out.append(self._forward(x).numpy())
        return np.concatenate(out)


class ConvNeXtNet(FeatureNet):
    """ConvNeXt-tiny after stage 2 (28 x 28 grid, 192 channels), the same grid as the Wide ResNet50 layer 2."""
    name = "convnext_tiny"

    def __init__(self, stage=3):
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        self.stage = stage
        self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).eval()

    def _forward(self, x):
        for k in range(self.stage + 1):
            x = self.model.features[k](x)
        return x.permute(0, 2, 3, 1)


class DinoNet(FeatureNet):
    """DINOv2 ViT-S/14 patch tokens (via timm), size / 14 patches per side."""
    name = "dinov2_small"

    def __init__(self, size=336):
        import timm
        self.size = size
        self.model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=size).eval()

    def _forward(self, x):
        t = self.model.forward_features(x)[:, 1:]  # drop the class token
        g = int(round(t.shape[1] ** 0.5))
        return t.reshape(t.shape[0], g, g, -1)


@torch.inference_mode()
def net_knn_maps(net, memory, imgs):
    """Distance of every patch to its nearest good patch, shape (N, g, g), with the features of `net`."""
    feats = net(imgs)
    g, c = feats.shape[1], feats.shape[-1]
    return np.array([torch.cdist(torch.from_numpy(f.reshape(-1, c)), memory).min(dim=1).values.reshape(g, g).numpy() for f in feats])


@dataclass
class NetReference:
    net: FeatureNet
    memory: torch.Tensor
    margin: int
    val_scores: np.ndarray = None

    def scores(self, imgs):
        return border_scores(net_knn_maps(self.net, self.memory, imgs), self.margin)

    def threshold(self, target_frr):
        return np.quantile(self.val_scores, 1 - target_frr, method="higher")


def fit_net_reference(cat, net, tol=1.05, max_frac=0.1, max_patches=40000, seed=0):
    """fit_reference with the features of `net` (memory from 80% of the good training images, margin and threshold from the rest)."""
    val_idx, fit_idx = held_out_split(cat, seed)
    f = net(cat.X[fit_idx])
    f = f.reshape(-1, f.shape[-1])
    keep = np.random.default_rng(seed).choice(len(f), size=min(max_patches, len(f)), replace=False)
    memory = torch.from_numpy(f[keep])
    val_maps = net_knn_maps(net, memory, cat.X[val_idx])
    prof = ring_profile(val_maps)
    margin = min(int(np.argmax(prof <= tol * np.median(prof[len(prof) // 2:]))), int(max_frac * val_maps.shape[1]))
    return NetReference(net, memory, margin, border_scores(val_maps, margin))
