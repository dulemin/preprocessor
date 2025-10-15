import io
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter

def process_photo(file_bytes: bytes) -> bytes:
    # 1. Laden und normalisieren
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")

    # 2. Auflösung erhöhen
    target_w = 1800
    if img.width < target_w:
        scale = target_w / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    # 3. In Graustufen umwandeln
    gray = np.array(img.convert("L"), dtype=np.uint8)  # garantiert 1 Kanal

    # 4. Deskew (optional – hier einfache Version, um sicher Graubild zu behalten)
    try:
        thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        edges = cv2.Canny(thr, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        if lines is not None:
            angles = [np.degrees(theta) - 90 for _, theta in lines[:, 0]]
            angle = np.median(angles)
            (h, w) = gray.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
    except Exception as e:
        print(f"Deskew skipped: {e}")

    # 5. Adaptive Threshold – jetzt garantiert Graubild
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35, 15
    )

    # 6. Morphologische Reinigung
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 7. Leicht schärfen
    pil_bin = Image.fromarray(bw).filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=3))

    # 8. Zurückgeben als PNG
    out = io.BytesIO()
    pil_bin.save(out, format="PNG", optimize=True)
    return out.getvalue()
