# app/preprocess.py
import io
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

logger = logging.getLogger(__name__)

# ---------- Konfiguration ----------

class DocKind(Enum):
    SCANNED = "scanned"
    PHOTO = "photographed"
    MIXED = "mixed"

@dataclass
class Cfg:
    target_width: int = 2400        # für kleine Schriftzeichen
    target_dpi: int = 300           # tesseract-freundlich
    angle_limit: float = 10.0       # maximale Deskew-Korrektur

    # Entrauschen
    bilateral_d: int = 5
    bilateral_sigma_color: int = 50
    bilateral_sigma_space: int = 50

    # CLAHE (sanft für Fotos)
    clahe_clip: float = 2.0
    clahe_grid: Tuple[int, int] = (8, 8)

    # Adaptive Threshold
    sauvola_win: int = 31           # ungerade
    sauvola_k: float = 0.2          # 0.2–0.34 üblich
    sauvola_r: float = 128.0

    # Morphologie
    open_size: int = 1
    close_size: int = 1

    # Schärfen
    unsharp_radius: float = 0.8
    unsharp_percent: int = 140
    unsharp_threshold: int = 2

# ---------- Hilfen ----------

def _pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def _cv_to_pil(arr: np.ndarray) -> Image.Image:
    if len(arr.shape) == 2:
        return Image.fromarray(arr)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def _ensure_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1

def _scale_to_width(img: Image.Image, target_w: int) -> Image.Image:
    if img.width >= target_w:
        return img
    s = target_w / img.width
    return img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)

def _local_contrast_std(gray: np.ndarray, win: int = 31) -> float:
    m = cv2.boxFilter(gray.astype(np.float32), ddepth=-1, ksize=(win, win),
                      normalize=True, borderType=cv2.BORDER_REPLICATE)
    m2 = cv2.boxFilter((gray.astype(np.float32) ** 2), ddepth=-1, ksize=(win, win),
                       normalize=True, borderType=cv2.BORDER_REPLICATE)
    var = np.maximum(m2 - m * m, 0.0)
    return float(np.mean(np.sqrt(var)))

def _illumination_correct(gray: np.ndarray) -> np.ndarray:
    # morphologisches Opening für Hintergrundschätzung + Division
    k = max(min(gray.shape) // 30, 15)
    k = _ensure_odd(k)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    bg = np.clip(bg, 1, 255)
    norm = cv2.divide(gray, bg, scale=255)
    return norm.astype(np.uint8)

def _auto_gamma(gray: np.ndarray, p_low=2, p_high=98) -> np.ndarray:
    lo = np.percentile(gray, p_low)
    hi = np.percentile(gray, p_high)
    if hi <= lo + 1:
        return gray
    out = np.clip((gray - lo) * (255.0 / (hi - lo)), 0, 255)
    return out.astype(np.uint8)

def _fast_denoise(gray: np.ndarray) -> np.ndarray:
    # bilateral ist robust, fastNlMeans ist optional (gleiches Ergebnis für beide Fälle)
    try:
        den = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        return den
    except Exception:
        return cv2.bilateralFilter(gray, 5, 50, 50)

# ---------- Sauvola / Niblack (ohne Zusatz-Module) ----------

def _mean_std(gray: np.ndarray, win: int) -> Tuple[np.ndarray, np.ndarray]:
    f32 = gray.astype(np.float32)
    ksize = (_ensure_odd(win), _ensure_odd(win))
    mean = cv2.boxFilter(f32, -1, ksize, normalize=True, borderType=cv2.BORDER_REPLICATE)
    mean_sq = cv2.boxFilter(f32 * f32, -1, ksize, normalize=True, borderType=cv2.BORDER_REPLICATE)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(var)
    return mean, std

def sauvola_binarize(gray: np.ndarray, w: int = 31, k: float = 0.2, r: float = 128.0) -> np.ndarray:
    mean, std = _mean_std(gray, w)
    thresh = mean * (1.0 + k * ((std / r) - 1.0))
    out = (gray.astype(np.float32) > thresh).astype(np.uint8) * 255
    return out

# ---------- Deskew + Orientierung ----------

def _deskew_angle(gray: np.ndarray) -> float:
    # Kanten → HoughLinesP → Winkel-Median
    v = np.median(gray)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, lower, upper)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(50, gray.shape[1] // 15),
                            minLineLength=gray.shape[1] // 8, maxLineGap=gray.shape[1] // 30)
    if lines is None:
        return 0.0

    angs = []
    for l in lines[:150]:
        x1, y1, x2, y2 = l[0]
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if a > 45:   a -= 90
        if a < -45:  a += 90
        if -20 <= a <= 20:
            angs.append(a)
    if not angs:
        return 0.0
    return float(np.median(angs))

def _rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)

def _best_90deg_orientation(bin_img: np.ndarray) -> np.ndarray:
    # wähle Rotation (0,90,180,270) mit größter Varianz der horizontalen Projektion
    candidates = [
        bin_img,
        cv2.rotate(bin_img, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(bin_img, cv2.ROTATE_180),
        cv2.rotate(bin_img, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    scores = []
    for b in candidates:
        # Zähle schwarze Pixel pro Zeile → Varianz hoch, wenn Textzeilen horizontal liegen
        proj = np.sum(b == 0, axis=1).astype(np.float32)
        scores.append(float(np.var(proj)))
    return candidates[int(np.argmax(scores))]

# ---------- Binarisierung (mit Auto-Fallback) ----------

def _adaptive_binarize(gray: np.ndarray, cfg: Cfg, photo_like: bool) -> np.ndarray:
    if photo_like:
        # sanfte Korrekturen vor adaptive Threshold
        gray2 = _illumination_correct(gray)
        gray2 = _auto_gamma(gray2, 2, 98)
        # Sauvola (stabil bei ungleichmäßiger Beleuchtung)
        binimg = sauvola_binarize(gray2, w=cfg.sauvola_win, k=cfg.sauvola_k, r=cfg.sauvola_r)
    else:
        # Scans: meist homogen → Otsu mit leichtem CLAHE
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=cfg.clahe_grid)
        g = clahe.apply(gray)
        _, binimg = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Überfüllung prüfen → Parameter nachjustieren
    white_ratio = np.mean(binimg == 255)
    if white_ratio > 0.97 or white_ratio < 0.03:
        # Fallback-Mix: 60% adaptive, 40% Otsu
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binimg = cv2.addWeighted(binimg, 0.6, otsu, 0.4, 0)
        binimg = np.where(binimg > 127, 255, 0).astype(np.uint8)
    return binimg

# ---------- Hauptpipeline ----------

def _detect_kind(gray: np.ndarray) -> DocKind:
    # grobe Heuristik: hohe lokale Std + Helligkeitsschwankungen → Foto
    lc = _local_contrast_std(gray, 31)
    # Helligkeitsschwankung über große Skala
    big = cv2.GaussianBlur(gray, (0, 0), sigmaX=21, sigmaY=21)
    fluct = float(np.std(big))
    if lc > 18 or fluct > 18:   # konservative Schwellwerte
        return DocKind.PHOTO
    # sehr homogen
    if lc < 10 and fluct < 10:
        return DocKind.SCANNED
    return DocKind.MIXED

def _morphology_smooth(binimg: np.ndarray, cfg: Cfg) -> np.ndarray:
    # nur sehr kleine Kerne, sonst „weißwaschen“ wir das Bild
    if cfg.open_size > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_size, cfg.open_size))
        binimg = cv2.morphologyEx(binimg, cv2.MORPH_OPEN, k, iterations=1)
    if cfg.close_size > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.close_size, cfg.close_size))
        binimg = cv2.morphologyEx(binimg, cv2.MORPH_CLOSE, k, iterations=1)
    return binimg

def _gentle_sharpen(binimg: np.ndarray, cfg: Cfg) -> Image.Image:
    pil = Image.fromarray(binimg)
    pil = pil.filter(ImageFilter.UnsharpMask(
        radius=cfg.unsharp_radius, percent=cfg.unsharp_percent, threshold=cfg.unsharp_threshold
    ))
    pil = ImageEnhance.Contrast(pil).enhance(1.03)  # ganz leicht
    return pil

def preprocess_image(file_bytes: bytes) -> bytes:
    cfg = Cfg()
    try:
        # 1) Laden + EXIF-Orientation
        with Image.open(io.BytesIO(file_bytes)) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")

        # 2) OCR-freundliche Größe
        im = _scale_to_width(im, cfg.target_width)

        # 3) nach CV2
        gray = cv2.cvtColor(_pil_to_cv(im), cv2.COLOR_BGR2GRAY)

        # 4) sanft entrauschen
        gray = _fast_denoise(gray)

        # 5) Dokument-Charakteristik
        kind = _detect_kind(gray)
        photo_like = (kind == DocKind.PHOTO or kind == DocKind.MIXED)

        # 6) Deskew (kleiner Winkel)
        angle = np.clip(_deskew_angle(gray), -cfg.angle_limit, cfg.angle_limit)
        if abs(angle) > 0.2:
            gray = _rotate(gray, angle)

        # 7) Adaptive Binarisierung mit Auto-Fallback
        binimg = _adaptive_binarize(gray, cfg, photo_like=photo_like)

        # 8) Orientierung in 90°-Schritten prüfen (fix für „um 90° gedreht“)
        binimg = _best_90deg_orientation(binimg)

        # 9) Leichte Morphologie + Schärfen (nur falls nicht „fast nur weiß“)
        if np.mean(binimg == 255) < 0.97:
            binimg = _morphology_smooth(binimg, cfg)
            out_pil = _gentle_sharpen(binimg, cfg)
        else:
            out_pil = Image.fromarray(binimg)

        # 10) PNG mit DPI
        buf = io.BytesIO()
        out_pil.save(buf, format="PNG", optimize=True, compress_level=6,
                     dpi=(cfg.target_dpi, cfg.target_dpi))
        return buf.getvalue()

    except Exception as e:
        logger.exception(f"Preprocessing error: {e}")
        # Fallback: Graustufen-PNG
        try:
            with Image.open(io.BytesIO(file_bytes)) as im:
                im = ImageOps.exif_transpose(im).convert("L")
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                return buf.getvalue()
        except Exception:
            raise
