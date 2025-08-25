# app/preprocess.py
import io
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter


def _pil_to_cv2_rgb(pil_img: Image.Image) -> np.ndarray:
    """PIL RGB -> OpenCV BGR (uint8)."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _compute_skew_angle(img_gray: np.ndarray) -> float:
    """
    Robust: erst Hough-Linien (horizontale Textzeilen), sonst Fallback minAreaRect.
    Rückgabe in Grad, positiv = gegen Uhrzeigersinn.
    """
    # 1) Hough-basierter Versuch
    # Kanten -> HoughLinesP -> Winkel der (nahezu) horizontalen Linien sammeln
    edges = cv2.Canny(img_gray, 60, 180)
    h, w = img_gray.shape[:2]
    min_len = int(0.25 * min(h, w))   # nur längere Linien
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=min_len, maxLineGap=20)

    angles = []
    if lines is not None:
        for l in lines[:2000]:
            x1, y1, x2, y2 = l[0]
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # nur (nahezu) horizontal berücksichtigen
            if -30 <= ang <= 30:
                angles.append(ang)

    if len(angles) >= 5:
        # Median ist robust gegen Ausreißer
        return float(np.median(angles))

    # 2) Fallback: minAreaRect über Textmaske (nach Otsu + Open)
    thr = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, cv2.getStructuringElement(
        cv2.MORPH_RECT, (3, 3)), iterations=1)
    coords = np.column_stack(np.where(thr > 0))
    if coords.size == 0:
        return 0.0

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]  # in [-90, 0)
    # auf [-45, 45) normalisieren
    if angle < -45:
        angle = 90 + angle
    # z. B. -7.3 ... +7.3
    return float(angle)


def preprocess_image(file_bytes: bytes) -> bytes:
    """
    - EXIF-Orientierung respektieren
    - ggf. hochskalieren (bessere OCR)
    - Deskew mit Hough-Fallback (clamped)
    - Adaptive Threshold + leichtes Schärfen
    - Ausgabe: PNG-Bytes (1-Kanal)
    """
    # 1) Laden (EXIF) und nach RGB
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")

    # 2) ggf. hochskalieren (Breite ~1800 px)
    target_w = 1800
    if img.width < target_w:
        scale = target_w / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    # 3) nach Graustufen (OpenCV) und leicht glätten
    img_bgr = _pil_to_cv2_rgb(img)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray, 3)

    # 4) Deskew (mit Begrenzung auf sinnvollen Bereich)
    angle = _compute_skew_angle(img_gray)
    ANGLE_LIMIT = 12.0  # harte Begrenzung gegen Fehlrotationen
    if angle > ANGLE_LIMIT:
        angle = ANGLE_LIMIT
    elif angle < -ANGLE_LIMIT:
        angle = -ANGLE_LIMIT

    if abs(angle) > 0.1:
        h, w = img_gray.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img_gray = cv2.warpAffine(
            img_gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

    # 5) Adaptive Threshold + kleine Morphologie
    thr = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, 1)), iterations=1)

    # 6) leichtes Nachschärfen im PIL
    pil_bin = Image.fromarray(thr).filter(
        ImageFilter.UnsharpMask(radius=1.0, percent=140, threshold=3)
    )

    # 7) PNG zurückgeben
    out = io.BytesIO()
    pil_bin.save(out, format="PNG", optimize=True)
    return out.getvalue()
