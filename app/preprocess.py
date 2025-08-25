# app/preprocess.py
# Sanfte, robuste Vorverarbeitung für OCR (ohne „übertreiben“)

import io
import cv2
import numpy as np
from PIL import Image, ImageOps


# -----------------------------
# kleine Hilfsfunktionen
# -----------------------------
def _pil_to_gray_np(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

def _resize_for_ocr(img: Image.Image, target_w: int = 1800) -> Image.Image:
    if img.width < target_w:
        s = target_w / img.width
        return img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    return img

def _auto_gamma(gray: np.ndarray) -> np.ndarray:
    # leichte automatische Gamma-Korrektur, nur wenn nötig
    mean = float(gray.mean()) / 255.0
    if mean < 0.35:
        gamma = 1.25  # etwas aufhellen (sanft)
    elif mean > 0.75:
        gamma = 0.85  # etwas abdunkeln (sanft)
    else:
        return gray
    inv = 1.0 / gamma
    table = (np.power(np.linspace(0, 1, 256), inv) * 255).astype(np.uint8)
    return cv2.LUT(gray, table)

def _clahe(gray: np.ndarray, clip: float = 1.6, tile=(8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    return clahe.apply(gray)

def _light_illumination_fix(gray: np.ndarray) -> np.ndarray:
    # milde Beleuchtungskorrektur via „division“ mit stark geglättetem Hintergrund
    h, w = gray.shape
    # Sigma abhängig von Bildgröße – eher sanft
    sigma = max(10, min(h, w) // 30)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    div = cv2.divide(gray, bg, scale=255)
    return np.clip(div, 0, 255).astype(np.uint8)

def _is_photographed(gray: np.ndarray) -> bool:
    # sehr einfache Heuristik: Fotos haben i.d.R. mehr Helligkeits-Varianz & Rauschen
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    std = gray.std()
    return (std > 40) or (lap < 150)  # eher Foto

def _deskew(gray: np.ndarray, max_abs_angle: float = 8.0) -> np.ndarray:
    # vorsichtige Schräglagenkorrektur; wenn nicht sicher -> lass es
    try:
        thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        inv = 255 - thr
        pts = cv2.findNonZero(inv)
        if pts is None:
            return gray
        rect = cv2.minAreaRect(pts)
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) < 0.2 or abs(angle) > max_abs_angle:
            return gray
        (h, w) = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    except Exception:
        return gray

def _mild_denoise(gray: np.ndarray, photo: bool) -> np.ndarray:
    # Bilateral ist schonend; bei Scans noch milder
    d = 5 if photo else 3
    sc = 45 if photo else 30
    ss = 45 if photo else 30
    return cv2.bilateralFilter(gray, d, sc, ss)

def _binarize(gray: np.ndarray, photo: bool) -> np.ndarray:
    """
    Sanfte, adaptive Binarisierung:
    - Für Fotos: Adaptive MEAN, eher größere Fenster, mildes C
    - Für Scans: Otsu (ggf. mit leichter Weichzeichnung)
    + Fallbacks, falls zu extrem
    """
    h, w = gray.shape
    bs = max(21, min(61, (min(h, w) // 40) | 1))  # 21..61, ungerade
    if photo:
        # adaptive, eher milde Parameter
        result = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY,
                                       bs, 8)
        # leicht mit Otsu mischen, um Ausreißer zu glätten
        otsu = cv2.threshold(gray, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        result = cv2.addWeighted(result, 0.75, otsu, 0.25, 0)
    else:
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        result = cv2.threshold(blur, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    white = float((result == 255).mean())
    black = 1.0 - white

    # Fallbacks, wenn es kippt
    if white > 0.98:  # zu weiß -> sanftere adaptive Variante
        result = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY,
                                       max(21, bs - 10) | 1, 6)
    elif black > 0.6:  # zu schwarz -> etwas anheben
        mean = gray.mean()
        thr = max(100, min(180, int(mean + 10)))
        _, result = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)

    # minimale Morphologie (nicht übertreiben!)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)
    return result


# -----------------------------
# Hauptfunktion (API)
# -----------------------------
def preprocess_image(file_bytes: bytes) -> bytes:
    """
    Sanfte, robuste Pipeline:
    1) Laden & ggf. leicht vergrößern
    2) leichte CLAHE + (bei Fotos) milde Beleuchtungskorrektur
    3) milde Rauschminderung
    4) vorsichtiges Deskew
    5) milde Binarisierung mit Fallbacks
    """
    # 1) Laden
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = _resize_for_ocr(img, target_w=1800)

    # 2) zu Graustufen + leichte Kontrast/Gamma
    gray = _pil_to_gray_np(img)
    gray = _clahe(gray, clip=1.4, tile=(8, 8))
    gray = _auto_gamma(gray)

    # Foto vs. Scan-Heuristik
    is_photo = _is_photographed(gray)

    # milde Beleuchtungskorrektur nur bei Fotos
    if is_photo:
        gray = _light_illumination_fix(gray)

    # 3) sanftes Denoise
    gray = _mild_denoise(gray, photo=is_photo)

    # 4) vorsichtige Schräglagenkorrektur
    gray = _deskew(gray, max_abs_angle=8.0)

    # 5) milde Binarisierung
    bin_img = _binarize(gray, photo=is_photo)

    # PNG ausgeben (mit DPI, hilft Tesseract)
    out = io.BytesIO()
    Image.fromarray(bin_img).save(out, format="PNG", optimize=True, dpi=(300, 300))
    return out.getvalue()
