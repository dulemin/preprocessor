# app/pipelines/scan.py
import io
import cv2
import numpy as np
from PIL import Image, ImageOps

def _resize_to_width(img_pil: Image.Image, target_w: int = 1800) -> Image.Image:
    if img_pil.width >= target_w:
        return img_pil
    scale = target_w / img_pil.width
    new_size = (int(img_pil.width * scale), int(img_pil.height * scale))
    return img_pil.resize(new_size, Image.LANCZOS)

def _estimate_skew_deg(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)
    if lines is None:
        return 0.0
    angles = []
    for rho, theta in lines[:, 0]:
        ang = (theta * 180.0 / np.pi) - 90.0
        if abs(ang) <= 20.0 and abs(abs(ang) - 45.0) > 5.0:
            angles.append(ang)
    if not angles:
        return 0.0
    return -float(np.median(angles))

def _safe_deskew(gray: np.ndarray) -> np.ndarray:
    angle = _estimate_skew_deg(gray)
    if abs(angle) < 0.4 or abs(angle) > 5.0:
        return gray
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def process_scan(file_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = _resize_to_width(img, 1800)

    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 3)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = _safe_deskew(gray)

    _, otsu = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    white_ratio = float(np.mean(otsu) / 255.0)
    use_binary = 0.30 < white_ratio < 0.88

    out = io.BytesIO()
    Image.fromarray(otsu if use_binary else gray).save(
        out, format="PNG", optimize=True
    )
    return out.getvalue()
