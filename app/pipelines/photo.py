import io
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter


# ---------- Hilfsfunktionen ----------

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """
    Robust: gibt bei Graubild ein 2D-Array (H x W, uint8) zurück,
    bei Farbbild ein BGR-Array (H x W x 3, uint8).
    """
    arr = np.array(pil_img, dtype=np.uint8)
    if arr.ndim == 2:  # bereits 1-Kanal (L)
        return arr
    if arr.shape[2] == 4:  # RGBA -> RGB
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    if img.ndim == 2:  # grayscale
        return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def ensure_gray(img: np.ndarray) -> np.ndarray:
    """
    Stellt sicher, dass das Bild 1-Kanal (uint8) ist.
    Akzeptiert BGR/BGRA/GRAY.
    """
    if img is None:
        return img
    if img.ndim == 2:
        # schon 1-Kanal, ggf. auf uint8 bringen
        if img.dtype != np.uint8:
            img = img.astype(np.uint8, copy=False)
        return img
    # 3/4-Kanal -> nach BGR2GRAY
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8, copy=False)
    return gray


def deskew_cv(img_gray: np.ndarray) -> np.ndarray:
    """
    Erwartet EIN-KANAL (uint8) Graubild.
    Bestimmt Schräglage über Hough-Lines und richtet aus.
    """
    # Safety: sicherstellen, dass es wirklich Graustufen sind
    img_gray = ensure_gray(img_gray)

    # Binarisieren für Winkelbestimmung (OTSU nur auf 1-Kanal!)
    thr = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Kantenerkennung
    edges = cv2.Canny(thr, 50, 150, apertureSize=3)

    # Hough-Linien für bessere Winkelbestimmung
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is not None and len(lines) > 0:
        # Sammle alle Winkel
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            # Normalisiere Winkel auf [-45, 45]
            if angle > 45:
                angle -= 90
            elif angle < -45:
                angle += 90
            angles.append(angle)

        # Verwende den Median der Winkel für Robustheit
        angle = float(np.median(angles))

    else:
        # Fallback: verwende minAreaRect
        coords = np.column_stack(np.where(thr > 0))
        if coords.size == 0:
            return img_gray

        rect = cv2.minAreaRect(coords)
        angle = rect[-1]

        # Winkelinterpretation verbessern
        (w_rect, h_rect) = rect[1]
        if w_rect < h_rect:
            angle = angle - 90

        # Begrenze auf [-45, 45]
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90

    # Kleine Winkel ignorieren
    if abs(angle) < 0.5:
        return img_gray

    # Rotation durchführen
    h, w = img_gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Canvas vergrößern, um Abschneiden zu vermeiden
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Matrix zentrieren
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        img_gray, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


# ---------- Haupt-Pipeline ----------

def process_photo(file_bytes: bytes) -> bytes:
    # 1) Laden mit PIL (EXIF-Orientation beachten)
    img = Image.open(io.BytesIO(file_bytes))
    # Falls das Bild Alphakanal hat (RGBA/LA), normalisieren wir auf RGB
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img = ImageOps.exif_transpose(img)

    # 2) Größe anheben (OCR profitiert von mehr Auflösung)
    target_w = 1800
    if img.width < target_w:
        scale = target_w / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    # 3) In Graustufen + sanft entrauschen (Median)
    img_gray_pil = img.convert("L").filter(ImageFilter.MedianFilter(size=3))
    img_gray = np.array(img_gray_pil, dtype=np.uint8)  # 2D (H x W)

    # 3b) Kontrast verbessern für dunkle Szenen (z. B. Gym): CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img_gray = clahe.apply(img_gray)

    # 4) Deskew (arbeitet auf 1-Kanal)
    img_deskew = deskew_cv(img_gray)

    # 5) Adaptive Threshold (nur 1-Kanal erlaubt) + Morphologie
    #    Leicht robustere Parameter: kleine Fenster vergrößern, C etwas reduzieren
    bw = cv2.adaptiveThreshold(
        img_deskew, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35, 12  # blockSize (ungerade), C
    )

    # 5b) kleines Close gegen „Löcher" in Zeichen
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 6) Leichtes Schärfen (im PIL-Raum)
    pil_bin = Image.fromarray(bw).filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=3))

    # 7) Als PNG-Bytes zurückgeben
    out = io.BytesIO()
    pil_bin.save(out, format="PNG", optimize=True)
    return out.getvalue()
