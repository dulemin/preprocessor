import io
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter

def process_photo(file_bytes: bytes) -> bytes:
    # 1. Laden und EXIF orientieren
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img)

    # 2. Immer RGB, falls nicht schon
    if img.mode != "RGB":
        img = img.convert("RGB")

    # 3. Auflösung erhöhen
    target_w = 1800
    if img.width < target_w:
        scale = target_w / img.width
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.LANCZOS
        )

    # 4. Graustufen erzwingen
    img_gray_pil = img.convert("L")
    img_gray = np.array(img_gray_pil, dtype=np.uint8)

    # Sicherheit: prüfen, dass 2D
    if img_gray.ndim != 2:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    # 5. Leicht entrauschen
    img_gray = cv2.medianBlur(img_gray, 3)

    # 6. Adaptive Threshold mit Fallback
    try:
        bw = cv2.adaptiveThreshold(
            img_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35, 15
        )
    except cv2.error as e:
        print("⚠️ OpenCV adaptiveThreshold failed, fallback to Otsu:", e)
        _, bw = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # 7. Morphologische Glättung
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 8. Schärfen
    pil_bin = Image.fromarray(bw).filter(
        ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=3)
    )

    # 9. Ausgabe als PNG
    out = io.BytesIO()
    pil_bin.save(out, format="PNG", optimize=True)
    return out.getvalue()
