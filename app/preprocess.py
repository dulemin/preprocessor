import io
import math
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter

# ---------- PIL <-> OpenCV ----------
def _pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def _cv2_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# ---------- Deskew (Hough-basierend, sicher geklemmt) ----------
def _estimate_skew_deg(gray: np.ndarray) -> float:
    # Kanten & dünnes Threshold für Linienerkennung
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60,
                            minLineLength=max(50, gray.shape[1]//3),
                            maxLineGap=20)
    if lines is None:
        return 0.0
    angles = []
    for x1,y1,x2,y2 in lines[:,0,:]:
        ang = math.degrees(math.atan2((y2-y1), (x2-x1)))
        # nur (nahezu) horizontale Kandidaten
        if -30 <= ang <= 30:
            angles.append(ang)
    if not angles:
        return 0.0
    med = float(np.median(angles))
    # Sicherheit: harte Ausreißer vermeiden
    return float(np.clip(med, -7.0, 7.0))

def _rotate_gray(gray: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.1:
        return gray
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w/2.0, h/2.0), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# ---------- Beleuchtungskorrektur (für Fotos) ----------
def _illumination_correct(gray: np.ndarray) -> np.ndarray:
    # Hintergrund mit großem Opening schätzen
    k = max(31, (min(gray.shape[:2]) // 20) * 2 + 1)  # dynamische Kernelgröße, ungerade
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    # Division stabilisiert ungleichmäßige Ausleuchtung
    corrected = cv2.divide(gray, bg, scale=255)
    # Strecken/normalisieren
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    corrected = corrected.astype(np.uint8)
    # Lokaler Kontrast (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(corrected)

# ---------- Sauvola-Binarisierung (ohne skimage) ----------
def _sauvola(gray: np.ndarray, win: int = 25, k: float = 0.34, R: float = 128.0) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    # lokale Mittelwerte / Varianzen via Boxfilter
    mean = cv2.boxFilter(gray_f, ddepth=-1, ksize=(win, win), normalize=True)
    mean_sq = cv2.boxFilter(gray_f * gray_f, ddepth=-1, ksize=(win, win), normalize=True)
    var = np.clip(mean_sq - mean * mean, 0, None)
    std = np.sqrt(var)
    thresh = mean * (1 + k * ((std / R) - 1))
    bin_img = (gray_f > thresh).astype(np.uint8) * 255
    return bin_img

# ---------- Auto-Branching: Foto vs. Scan ----------
def _is_photo_like(gray: np.ndarray) -> bool:
    # Metrik für ungleichmäßige Beleuchtung
    # Mittelwert der Abweichung von stark geglätteter Version
    blur = cv2.GaussianBlur(gray, (0, 0), 21)
    uneven = cv2.mean(cv2.absdiff(gray, blur))[0]
    return uneven >= 4.0  # Schwelle empirisch, robust in der Praxis

def preprocess_image(file_bytes: bytes, force_mode: str | None = None,
                     return_binary: bool = False) -> bytes:
    # 1) Laden + EXIF-Orientierung
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")

    # 2) auf sinnvolle OCR-Breite hochskalieren (nur wenn nötig)
    target_w = 1800
    if img.width < target_w:
        scale = target_w / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    # 3) Graustufe & mildes Rauschen entfernen
    gray = _pil_to_cv2(img)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    # 4) Deskew (sicher)
    angle = _estimate_skew_deg(gray)
    gray = _rotate_gray(gray, angle)

    # 5) Modus bestimmen
    if force_mode is not None:
        mode = force_mode  # "photo" | "scan"
    else:
        mode = "photo" if _is_photo_like(gray) else "scan"

    if mode == "photo":
        norm = _illumination_correct(gray)
        if return_binary:
            out_gray = _sauvola(norm, win=25, k=0.34, R=128)
        else:
            out_gray = norm  # Graustufe an Tesseract geben
    else:
        # Scan: wenig anfassen
        sm = cv2.GaussianBlur(gray, (3, 3), 0)
        if return_binary:
            # Otsu ist hier meist stark genug
            _, out_gray = cv2.threshold(sm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            out_gray = sm

    # 6) leichtes Schärfen in Graustufen (nur sehr mild)
    out_pil = Image.fromarray(out_gray).filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))

    # 7) PNG ausgeben (Graustufe oder Binär, beide als 8-bit)
    buf = io.BytesIO()
    out_pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
