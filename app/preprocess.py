# app/preprocess.py
import io
import math
import cv2
import numpy as np
from PIL import Image, ImageOps

def _pil_to_cv2(img_pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def _cv2_to_pil(img_cv: np.ndarray) -> Image.Image:
    # img_cv is BGR or Gray; ensure RGB/ L for PIL
    if len(img_cv.shape) == 2:
        return Image.fromarray(img_cv)  # 'L'
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def _resize_to_width(img_pil: Image.Image, target_w: int = 1800) -> Image.Image:
    if img_pil.width >= target_w:
        return img_pil
    scale = target_w / img_pil.width
    new_size = (int(img_pil.width * scale), int(img_pil.height * scale))
    return img_pil.resize(new_size, Image.LANCZOS)

def _estimate_skew_deg(gray: np.ndarray) -> float:
    """
    Robust Skew-Schätzung über Hough-Linien.
    Wir suchen dominante quasi-horizontale Linien und mitteln den Winkel.
    Rückgabe in Grad. Positiv = im Uhrzeigersinn drehen nötig.
    """
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)
    if lines is None:
        return 0.0

    angles = []
    for rho, theta in lines[:, 0]:
        # theta nahe 0° oder 180° -> vertikale Linien; nahe 90° -> horizontale.
        # Wir wollen horizontale (Textzeilen), daher um 90° zentrieren:
        ang = (theta * 180.0 / np.pi) - 90.0
        # horizontnah (|ang| <= 20°), aber ignoriere diagonale Kästchen (~45°)
        if abs(ang) <= 20.0 and abs(abs(ang) - 45.0) > 5.0:
            angles.append(ang)

    if not angles:
        return 0.0
    # Median ist robuster als Mittelwert
    ang_med = float(np.median(angles))
    # Text “nach rechts fallend” (negativ) bedeutet: Gegenuhrzeigersinn drehen,
    # wir definieren positiv als "im Uhrzeigersinn drehen nötig":
    return -ang_med

def _safe_deskew(gray: np.ndarray) -> np.ndarray:
    """
    Drehe nur, wenn der geschätzte Skew klein ist (|angle| <= 5°).
    Vermeidet 45°-Fehldrehungen bei Formular-Kästchen.
    """
    angle = _estimate_skew_deg(gray)
    if abs(angle) < 0.4:  # < ~0.4°: vernachlässigbar
        return gray

    if abs(angle) > 5.0:
        # Zu unsicher/groß – Tesseract mit OSD erledigt das besser.
        return gray

    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def preprocess_image(file_bytes: bytes) -> bytes:
    # 1) Laden + EXIF-Orientierung
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")

    # 2) Auf sinnvolle Breite hochskalieren (OCR profitiert von ~300dpi)
    img = _resize_to_width(img, 1800)

    # 3) Nach OpenCV, Graustufen + leichter Rauschfilter
    img_cv = _pil_to_cv2(img)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    # 4) Lokalen Kontrast erhöhen (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 5) Nur vorsichtig deskewen
    gray = _safe_deskew(gray)

    # 6) Opportunistische Binarisierung – nur wenn der Otsu-Output "gesund" ist
    #    (zu viel Weiß oder Schwarz -> lieber Graustufe lassen, Tesseract binarisiert selbst)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = float(np.mean(otsu) / 255.0)
    use_binary = 0.30 < white_ratio < 0.88  # Daumenregel

    out = io.BytesIO()
    if use_binary:
        _cv2_to_pil(otsu).save(out, format="PNG", optimize=True)
    else:
        _cv2_to_pil(gray).save(out, format="PNG", optimize=True)
    return out.getvalue()
