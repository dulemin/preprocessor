# app/preprocess.py
"""
Robuste Bildvorverarbeitung für OCR (nur OpenCV + Pillow).
- Vermeidet Überkontrast bei Handyfotos (sanftere Beleuchtungskorrektur)
- Deskew nur in kleinem Winkelbereich (kein "45°-Kippen")
- Adaptive Binarisierung mit Fallback, falls zu "weiß"
- Sehr vorsichtige Morphologie & Schärfung
"""

from __future__ import annotations
import io
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List

import cv2  # OpenCV
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance  # Pillow

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# -----------------------------
#   Konfiguration / Typen
# -----------------------------

class DocumentType(Enum):
    SCANNED = "scanned"
    PHOTOGRAPHED = "photographed"
    PRINTED = "printed"
    HANDWRITTEN = "handwritten"
    MIXED = "mixed"


@dataclass
class ProcessingConfig:
    # Auflösung & DPI (Tesseract fühlt sich bei 300dpi wohl)
    target_width: int = 2400
    target_dpi: int = 300

    # Schräglage: wir korrigieren nur kleine Winkel (kein 90°-Turn)
    angle_limit: float = 10.0  # Grad

    # Rauschen
    bilateral_d: int = 5
    bilateral_sigma_color: float = 50
    bilateral_sigma_space: float = 50
    noise_reduction_extra: bool = False  # optionaler zusätzlicher Blur

    # Beleuchtung / Kontrast
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    gamma: float = 1.0  # 1.0 = aus

    # Morphologie
    open_size: int = 1
    close_size: int = 2

    # Schärfen
    unsharp_radius: float = 0.8
    unsharp_percent: int = 140
    unsharp_threshold: int = 2

    # Feature-Flags
    use_shadow_removal: bool = True
    use_deblurring: bool = False
    use_text_enhancement: bool = True


def _cfg_for(doc: DocumentType) -> ProcessingConfig:
    if doc == DocumentType.SCANNED:
        return ProcessingConfig(
            target_width=2200, use_shadow_removal=False,
            clahe_clip_limit=1.4, gamma=1.0, use_text_enhancement=True
        )
    if doc == DocumentType.PHOTOGRAPHED:
        return ProcessingConfig(
            target_width=2600, use_shadow_removal=True,
            clahe_clip_limit=2.2, gamma=1.1, use_text_enhancement=True
        )
    if doc == DocumentType.PRINTED:
        return ProcessingConfig(
            target_width=2400, use_shadow_removal=True,
            clahe_clip_limit=1.8, gamma=1.05, use_text_enhancement=True
        )
    if doc == DocumentType.HANDWRITTEN:
        return ProcessingConfig(
            target_width=2800, use_shadow_removal=True,
            close_size=3, unsharp_percent=160, unsharp_threshold=1,
            clahe_clip_limit=1.8, use_text_enhancement=True
        )
    return ProcessingConfig()  # MIXED / Fallback


# -----------------------------
#   Hilfsfunktionen
# -----------------------------

def _pil_to_gray(img_rgb: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2GRAY)


def _resize_for_ocr(img: Image.Image, target_width: int) -> Image.Image:
    if img.width < target_width:
        s = target_width / img.width
        img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    return img


def _bilateral_denoise(img: np.ndarray, cfg: ProcessingConfig) -> np.ndarray:
    out = cv2.bilateralFilter(img, cfg.bilateral_d, cfg.bilateral_sigma_color, cfg.bilateral_sigma_space)
    if cfg.noise_reduction_extra:
        k = 3 | 1  # ungerade
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out


def _shadow_removal(img: np.ndarray, cfg: ProcessingConfig) -> np.ndarray:
    """
    Sanfte Hintergrundschätzung: morph. Öffnen + Division + leichte CLAHE.
    (Vermeidet harte Weißflächen bei Fotos.)
    """
    try:
        h, w = img.shape[:2]
        k = max(15, min(h, w) // 30)  # kleiner Kernel als früher
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bg = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        norm = cv2.divide(img, bg, scale=255)
        mixed = cv2.addWeighted(norm, 0.7, img, 0.3, 0)

        clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=cfg.clahe_tile_grid)
        return clahe.apply(mixed.astype(np.uint8))
    except Exception as e:
        logger.warning(f"Shadow removal failed: {e}")
        return img


def _apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    if gamma == 1.0:
        return img
    inv = 1.0 / gamma
    tbl = (np.arange(256) / 255.0) ** inv * 255.0
    return cv2.LUT(img, tbl.astype(np.uint8))


def _deskew_angle_hough(gray: np.ndarray) -> Optional[float]:
    # adaptive Canny
    v = np.median(gray); sigma = 0.33
    lo = int(max(0, (1.0 - sigma) * v)); hi = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(gray, lo, hi)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=max(50, gray.shape[1] // 15),
                            minLineLength=gray.shape[1] // 8, maxLineGap=gray.shape[1] // 30)
    if lines is None:
        return None
    angles: List[float] = []
    for (x1, y1, x2, y2) in lines[:, 0, :]:
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if ang > 45: ang -= 90
        if ang < -45: ang += 90
        if -15 <= ang <= 15:
            angles.append(ang)
    if len(angles) < 3:
        return None
    return float(np.median(angles))


def _deskew_angle_contours(gray: np.ndarray) -> Optional[float]:
    try:
        _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(binv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) > 100]
        if not cnts:
            return None
        angles = []
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:10]:
            angle = cv2.minAreaRect(c)[-1]
            if angle < -45: angle = 90 + angle
            if -15 <= angle <= 15:
                angles.append(angle)
        if len(angles) < 2:
            return None
        return float(np.median(angles))
    except Exception:
        return None


def _rotate_keep_size(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def _adaptive_binary(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive Threshold mit Fenstergröße abhängig von der Auflösung +
    Fallback, wenn Ergebnis zu weiß/schwarz ist.
    """
    block = max(31, min(gray.shape) // 15)
    if block % 2 == 0:
        block += 1
    result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block, 8)
    white = (result == 255).mean()
    if white > 0.95 or white < 0.05:
        # Mischung mit Otsu, wenn zu extrem
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = cv2.addWeighted(result, 0.6, otsu, 0.4, 0)
        result = (result > 127).astype(np.uint8) * 255
    return result


def _morph_cleanup(binimg: np.ndarray, cfg: ProcessingConfig) -> np.ndarray:
    white = (binimg == 255).mean()
    if white > 0.90:
        return binimg  # schon sehr weiß -> nichts tun

    out = binimg
    if cfg.open_size > 0:
        k = min(cfg.open_size, 2)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)
    if cfg.close_size > 0 and (out == 255).mean() < 0.90:
        k = min(cfg.close_size, 2)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)), iterations=1)
    return out


def _unsharp_pil(binimg: np.ndarray, cfg: ProcessingConfig) -> Image.Image:
    pil = Image.fromarray(binimg)
    sharp = pil.filter(ImageFilter.UnsharpMask(
        radius=cfg.unsharp_radius, percent=cfg.unsharp_percent, threshold=cfg.unsharp_threshold
    ))
    # leichte Kantanhebung + minimale Kontrastanhebung
    edge = sharp.filter(ImageFilter.EDGE_ENHANCE)
    mixed = cv2.addWeighted(np.array(sharp), 0.85, np.array(edge), 0.15, 0)
    res = Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8))
    return ImageEnhance.Contrast(res).enhance(1.03)


def _doc_type(gray: np.ndarray) -> DocumentType:
    """
    Sehr einfache Heuristiken:
    - Kantendichte (Canny)
    - Varianz des Laplacian (Schärfemaß)
    - mittlere Helligkeit
    """
    edges = cv2.Canny(gray, 50, 150)
    edge_density = (edges > 0).mean()
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_b = gray.mean()

    if edge_density < 0.02 and lap_var > 900 and mean_b > 200:
        return DocumentType.SCANNED
    if edge_density > 0.05 and lap_var < 600:
        return DocumentType.PHOTOGRAPHED
    if edge_density > 0.025 and lap_var > 800:
        return DocumentType.PRINTED
    return DocumentType.MIXED


# -----------------------------
#   Haupt-Pipeline
# -----------------------------

def preprocess_image(file_bytes: bytes) -> bytes:
    """
    Lädt Bildbytes → RGB → Graustufen → Vorverarbeitung → PNG (300dpi).
    Gibt PNG-Bytes zurück (für n8n HTTP Response Binary).
    """
    try:
        with Image.open(io.BytesIO(file_bytes)) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")

        # 1) für OCR hochskalieren (falls nötig)
        im = _resize_for_ocr(im, target_width=2400)

        # 2) Graustufen
        gray = _pil_to_gray(im)

        # 3) Dokumenttyp + passende Config
        dtype = _doc_type(gray)
        cfg = _cfg_for(dtype)
        logger.info(f"Document type detected: {dtype.value}")

        # 4) Rauschen
        gray = _bilateral_denoise(gray, cfg)

        # 5) Beleuchtung (für Fotos sanft, für Scans meist aus)
        if cfg.use_shadow_removal:
            gray = _shadow_removal(gray, cfg)

        # 6) Gamma (sanft, um „ausgefressene“ Bereiche zu vermeiden)
        gray = _apply_gamma(gray, cfg.gamma)

        # 7) Deskew (nur kleiner Winkel; verhindert „45°“-Effekt)
        a1 = _deskew_angle_hough(gray)
        a2 = _deskew_angle_contours(gray)
        angles = [a for a in [a1, a2] if a is not None]
        if angles:
            ang = float(np.median(angles))
            ang = float(np.clip(ang, -cfg.angle_limit, cfg.angle_limit))
            if abs(ang) > 0.2:
                gray = _rotate_keep_size(gray, ang)
                logger.info(f"Deskew: {ang:.2f}°")

        # 8) Binarisierung (adaptive + Fallback)
        binimg = _adaptive_binary(gray)
        white_ratio = (binimg == 255).mean()
        logger.info(f"Binary white ratio: {white_ratio:.2%}")

        # 9) (optional) Text-Enhancement als sehr sanfte Morphologie
        if cfg.use_text_enhancement and white_ratio < 0.95:
            binimg = _morph_cleanup(binimg, cfg)

        # 10) sanftes Schärfen
        if white_ratio < 0.95:
            out_img = _unsharp_pil(binimg, cfg)
        else:
            # Wenn zu weiß geworden ist, nutze „milderes“ statisches Schwellwerten
            t = np.percentile(gray, 20)
            _, tmp = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
            out_img = Image.fromarray(tmp)

        # 11) PNG mit DPI speichern
        buf = io.BytesIO()
        out_img.save(buf, format="PNG", optimize=True, compress_level=6, dpi=(cfg.target_dpi, cfg.target_dpi))
        return buf.getvalue()

    except Exception as e:
        logger.error(f"Preprocess failed: {e}")
        # Minimaler Fallback: nur nach L konvertieren
        try:
            with Image.open(io.BytesIO(file_bytes)) as im:
                im = ImageOps.exif_transpose(im).convert("L")
                b = io.BytesIO()
                im.save(b, format="PNG")
                return b.getvalue()
        except Exception:
            raise
