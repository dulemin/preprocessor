# app/pipelines/photo.py
import io
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter

def _to_gray_1ch(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("L").filter(ImageFilter.MedianFilter(size=3)), dtype=np.uint8)

def _deskew_minarearect(gray: np.ndarray) -> np.ndarray:
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if coords.size == 0:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def _resize_if_needed(img: Image.Image, target_w: int = 1800) -> Image.Image:
    if img.width >= target_w:
        return img
    scale = target_w / img.width
    return img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

def process_photo(file_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = _resize_if_needed(img, 1800)

    gray = _to_gray_1ch(img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = _deskew_minarearect(gray)

    block = max(31, (min(gray.shape) // 20) | 1)
    C = 8
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block, C
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    pil_bin = Image.fromarray(bw).filter(
        ImageFilter.UnsharpMask(radius=0.8, percent=120, threshold=3)
    )

    out = io.BytesIO()
    pil_bin.save(out, format="PNG", optimize=True)
    return out.getvalue()
