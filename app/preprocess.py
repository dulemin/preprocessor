import io
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter

def _pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def _cv2_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def _deskew_cv(img_gray: np.ndarray) -> np.ndarray:
    # Binarisieren für Winkelbestimmung
    thr = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if coords.size == 0:
        return img_gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img_gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def preprocess_image(file_bytes: bytes) -> bytes:
    # 1) Laden mit PIL (EXIF-Orientation wird berücksichtigt)
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")

    # 2) Größenskalierung: auf ~1800px Breite vergrößern (OCR hilft höhere Auflösung)
    target_w = 1800
    if img.width < target_w:
        scale = target_w / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    # 3) In Graustufen und leicht entrauschen
    img_gray_pil = img.convert("L").filter(ImageFilter.MedianFilter(size=3))
    img_gray = _pil_to_cv2(img_gray_pil)

    # 4) Deskew
    img_deskew = _deskew_cv(img_gray)

    # 5) Adaptive Threshold (binarisieren) + leichte Morphologie
    thr = cv2.adaptiveThreshold(img_deskew, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 35, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 6) Leichtes Schärfen
    pil_bin = Image.fromarray(thr).filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=3))

    # 7) Als PNG Bytes zurückgeben
    out = io.BytesIO()
    pil_bin.save(out, format="PNG", optimize=True)
    return out.getvalue()
