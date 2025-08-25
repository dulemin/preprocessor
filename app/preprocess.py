# app/preprocess.py
import io
import math
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter

# Maximal erlaubter Deskew-Winkel (Grad)
MAX_SKEW_DEG = 18.0

def _pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """
    Konvertiert PIL -> NumPy.
    - L-Bilder (Graustufen) bleiben 2D (uint8)
    - RGB wird nach BGR konvertiert
    """
    arr = np.array(pil_img)
    if arr.ndim == 2:
        # Graustufe: already fine for OpenCV ops
        return arr
    # RGB -> BGR
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def _cv2_to_pil(img: np.ndarray) -> Image.Image:
    if img.ndim == 2:
        return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def _median_deg(values: list[float]) -> float:
    a = np.array(values, dtype=np.float32)
    return float(np.median(a)) if a.size else 0.0

def _clamp_angle(angle: float, max_abs: float = MAX_SKEW_DEG) -> float:
    """Begrenzt den Winkel auf [-max_abs, +max_abs]."""
    return max(min(angle, max_abs), -max_abs)

def _deskew_via_hough(img_gray: np.ndarray) -> float:
    """
    Schätzt den Skew-Winkel über Hough-Linien:
    - Nur (nahezu) horizontale Linien werden berücksichtigt.
    - Ergebnis ist der Medianwinkel in Grad, in [-MAX_SKEW_DEG, +MAX_SKEW_DEG] geklemmt.
    """
    # Sanft glätten, dann binarisieren (Otsu benötigt Single-Channel)
    blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Kanten + Hough
    edges = cv2.Canny(bw, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, 120)  # Schwelle je nach Bild anpassbar

    if lines is None:
        return 0.0

    angles = []
    for rho, theta in lines[:, 0]:
        # Winkel relativ zur Horizontalen
        deg = (theta * 180.0 / math.pi) - 90.0
        # normalisiere auf [-90, 90]
        if deg < -90.0:
            deg += 180.0
        elif deg > 90.0:
            deg -= 180.0
        # nur nahezu horizontale Linien sammeln
        if -MAX_SKEW_DEG <= deg <= MAX_SKEW_DEG:
            angles.append(deg)

    if not angles:
        return 0.0
    return _clamp_angle(_median_deg(angles))

def _deskew_fallback_minarearect(img_gray: np.ndarray) -> float:
    """
    Fallback: Winkel per RotatedRect (minAreaRect).
    Achtung: Dieser kann 90° liefern. Deshalb wird der Winkel
    auf sinnvolle Werte normalisiert und geklemmt.
    """
    # Wir möchten Textpixel: In der Regel sind die dunkel -> 0 nach THRESH_BINARY
    _, bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text_mask = (bw == 0).astype(np.uint8) * 255

    coords = np.column_stack(np.where(text_mask > 0))
    if coords.size == 0:
        return 0.0

    rect = cv2.minAreaRect(coords)
    ang = rect[-1]  # OpenCV gibt hier z. B. [-90, 0) zurück
    # normalize auf [-45, +45]
    if ang < -45.0:
        ang += 90.0
    # und sicherheitshalber klemmen
    return _clamp_angle(float(ang))

def _rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 0.1:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

def _deskew_cv(img_gray: np.ndarray) -> np.ndarray:
    """
    Zweistufig:
    1) Hough-Linien → Winkel in [-MAX_SKEW_DEG, +MAX_SKEW_DEG]
    2) Fallback minAreaRect (ebenfalls geklemmt)
    """
    angle = _deskew_via_hough(img_gray)
    if abs(angle) < 0.1:
        angle = _deskew_fallback_minarearect(img_gray)
    return _rotate(img_gray, angle)

def preprocess_image(file_bytes: bytes) -> bytes:
    # 1) Laden mit korrekter EXIF-Ausrichtung
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")

    # 2) Auflösung erhöhen (falls klein) – hilft OCR
    target_w = 1800
    if img.width < target_w:
        scale = target_w / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    # 3) In Graustufen + leichtes Entrauschen
    img_gray_pil = img.convert("L").filter(ImageFilter.MedianFilter(size=3))
    img_gray = _pil_to_cv2(img_gray_pil)  # 2D uint8

    # 4) Deskew (robust & geklemmt)
    img_deskew = _deskew_cv(img_gray)

    # 5) Adaptive Threshold + leichte Morphologie
    thr = cv2.adaptiveThreshold(
        img_deskew, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        35, 15
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 6) Mildes Schärfen
    pil_bin = Image.fromarray(thr).filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=3))

    # 7) Als PNG zurück
    out = io.BytesIO()
    pil_bin.save(out, format="PNG", optimize=True)
    return out.getvalue()
