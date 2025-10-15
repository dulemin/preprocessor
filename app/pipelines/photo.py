# app/pipelines/scan.py
import io
import cv2
import numpy as np
from PIL import Image, ImageOps


# --------- Utilities ---------

def _resize_to_width(img_pil: Image.Image, target_w: int = 1800) -> Image.Image:
    if img_pil.width >= target_w:
        return img_pil
    scale = target_w / img_pil.width
    new_size = (int(img_pil.width * scale), int(img_pil.height * scale))
    return img_pil.resize(new_size, Image.LANCZOS)


def _ensure_gray_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Stellt sicher, dass ein 1-Kanal uint8 Bild vorliegt.
    Akzeptiert RGB/BGR/BGRA/GRAY.
    """
    # KRITISCH: Zuerst prüfen ob es schon 2D ist
    if arr.ndim == 2:
        return arr.astype(np.uint8, copy=False)
    
    # Wenn 3D, müssen wir konvertieren
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
        elif arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        elif arr.shape[2] == 1:
            arr = arr[:, :, 0]
    
    return arr.astype(np.uint8, copy=False)


def _estimate_skew_deg(gray: np.ndarray) -> float:
    # Kanten + Hough-Linien
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)
    if lines is None:
        return 0.0

    angles = []
    for rho, theta in lines[:, 0]:
        ang = (theta * 180.0 / np.pi) - 90.0
        # Grobe Ausreißer vermeiden
        if abs(ang) <= 25.0 and abs(abs(ang) - 45.0) > 5.0:
            angles.append(ang)

    if not angles:
        return 0.0
    return -float(np.median(angles))


def _safe_deskew(gray: np.ndarray) -> np.ndarray:
    angle = _estimate_skew_deg(gray)
    # nur kleine Korrekturen, sonst lieber lassen
    if abs(angle) < 0.4 or abs(angle) > 5.0:
        return gray
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


# --------- Main pipeline ---------

def process_scan(file_bytes: bytes) -> bytes:
    # 1) Laden + Orientierung normalisieren
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img)
    
    # 2) Immer zu RGB konvertieren (vereinheitlicht die Basis)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # 3) Auf OCR-freundliche Breite
    img = _resize_to_width(img, 1800)

    # 4) Zu Graustufen konvertieren (via PIL für Konsistenz)
    if img.mode != "L":
        img = img.convert("L")
    
    # 5) Zu numpy und sicherstellen dass es 2D ist
    gray = np.array(img, dtype=np.uint8)
    gray = _ensure_gray_uint8(gray)
    
    # 6) Leicht entrauschen
    gray = cv2.medianBlur(gray, 3)

    # 7) Kontrast heben (CLAHE hilft bei dunklen Stories)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 8) Vorsichtig deskew
    gray = _safe_deskew(gray)

    # 9) Binarisierung: erst OTSU, bei grenzwertigen Bildern Fallback auf adaptiv
    try:
        _, otsu = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        white_ratio = float(np.mean(otsu) / 255.0)
        use_binary = 0.30 < white_ratio < 0.88
    except cv2.error as e:
        print("⚠️ OTSU failed, using adaptive threshold:", e)
        # Falls doch mal ein Fehler auftritt: direkt adaptiv
        otsu = None
        use_binary = False

    if not use_binary:
        # Adaptive Threshold (robust bei ungleichmäßiger Beleuchtung)
        bin_img = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35, 12
        )
        # leichte Morphologie gegen Löcher
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)
        out_img = bin_img
    else:
        out_img = otsu

    # 10) PNG zurückgeben
    out = io.BytesIO()
    Image.fromarray(out_img).save(out, format="PNG", optimize=True)
    return out.getvalue()
