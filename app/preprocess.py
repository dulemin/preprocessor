import io
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter


def _pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """
    Konvertiert PIL -> OpenCV.
    - RGB/RGBA -> BGR (3-Kanal)
    - L (Graustufe) bleibt 1-Kanal (uint8)
    """
    arr = np.array(pil_img)
    # Graustufenbild (H, W)
    if arr.ndim == 2:
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr
    # RGBA -> RGB
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    # RGB -> BGR
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _cv2_to_pil(img: np.ndarray) -> Image.Image:
    """
    Konvertiert OpenCV -> PIL.
    - 1-Kanal bleibt 'L'
    - BGR -> RGB
    """
    if img.ndim == 2:
        return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def _deskew_cv(img_gray: np.ndarray) -> np.ndarray:
    """
    Schätzt Schräglage aus einem binarisierten Graubild und richtet aus.
    Erwartet 1-Kanal uint8.
    """
    if img_gray.ndim != 2:
        raise ValueError("Deskew erwartet ein 1-Kanal Graubild")

    # Binarisieren (invertiert), damit Text als Weiß auf Schwarz erscheint
    _, thr = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(thr > 0))
    if coords.size == 0:
        return img_gray

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    h, w = img_gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img_gray, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def preprocess_image(file_bytes: bytes) -> bytes:
    # 1) Laden mit PIL (EXIF-Orientation wird berücksichtigt)
    pil = Image.open(io.BytesIO(file_bytes))
    pil = ImageOps.exif_transpose(pil).convert("RGB")  # alpha weg, konsistent

    # 2) Auf sinnvolle Breite hochskalieren (OCR profitiert von Auflösung)
    target_w = 1800
    if pil.width < target_w:
        scale = target_w / pil.width
        pil = pil.resize((int(pil.width * scale), int(pil.height * scale)), Image.LANCZOS)

    # 3) Zu Graustufen konvertieren und leicht entrauschen
    pil_gray = pil.convert("L").filter(ImageFilter.MedianFilter(size=3))
    gray = _pil_to_cv2(pil_gray)  # -> uint8, 1-Kanal
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    # 4) Deskew auf Graubild
    gray = _deskew_cv(gray)

    # 5) Adaptive Threshold (nur 1-Kanal, uint8; blockSize=odd>=3)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,  # Blockgröße (ungerade)
        15   # C
    )

    # 6) Leichte Morphologie (optional; 1x1 ist idempotent, ggf. 2x2 testen)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 7) Leichtes Schärfen (PIL)
    pil_bin = Image.fromarray(thr).filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=3))

    # 8) Als PNG-Bytes zurückgeben
    out = io.BytesIO()
    pil_bin.save(out, format="PNG", optimize=True)
    out.seek(0)
    return out.getvalue()
