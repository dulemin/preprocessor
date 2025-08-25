import io
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter

def _pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """
    Robust: gibt bei Graubild ein 2D-Array (H x W, uint8) zurück,
    bei Farbbild ein BGR-Array (H x W x 3, uint8).
    """
    arr = np.array(pil_img, dtype=np.uint8)
    if arr.ndim == 2:  # already grayscale (L)
        return arr
    if arr.shape[2] == 4:  # RGBA -> RGB
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def _cv2_to_pil(img: np.ndarray) -> Image.Image:
    if img.ndim == 2:  # grayscale
        return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def _deskew_cv(img_gray: np.ndarray) -> np.ndarray:
    """
    Erwartet EIN-KANAL (uint8) Graubild.
    Bestimmt Schräglage über Otsu + minAreaRect und richtet aus.
    """
    # Safety: sicherstellen, dass es wirklich Graustufen sind
    if img_gray.ndim != 2:
        raise ValueError("deskew: erwartet ein Graubild (1 Kanal)")

    # Binarisieren für Winkelbestimmung (Otsu auf 1-Kanal OK)
    thr = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if coords.size == 0:
        return img_gray

    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect liefert [-90, 0); in "lesbaren" Winkel konvertieren
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = img_gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img_gray, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

def process_photo(file_bytes: bytes) -> bytes:
    # 1) Laden mit PIL (EXIF-Orientation beachten)
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")  # normalisieren

    # 2) Größe anheben (OCR dankt höhere Auflösung)
    target_w = 1800
    if img.width < target_w:
        scale = target_w / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    # 3) In Graustufen + sanft entrauschen
    img_gray_pil = img.convert("L").filter(ImageFilter.MedianFilter(size=3))
    # -> als echtes 1-Kanal-OpenCV-Bild holen
    img_gray = np.array(img_gray_pil, dtype=np.uint8)  # 2D (H x W)

    # 4) Deskew (arbeitet auf 1-Kanal)
    img_deskew = _deskew_cv(img_gray)

    # 5) Adaptive Threshold (nur 1-Kanal erlaubt) + Morphologie
    bw = cv2.adaptiveThreshold(
        img_deskew, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35, 15
    )
    # kleines Close gegen „Löcher“ in Zeichen
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 6) Leichtes Schärfen (im PIL-Raum)
    pil_bin = Image.fromarray(bw).filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=3))

    # 7) Als PNG-Bytes zurückgeben
    out = io.BytesIO()
    pil_bin.save(out, format="PNG", optimize=True)
    return out.getvalue()
