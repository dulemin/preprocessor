# app/preprocess.py
"""
Stabile Bildvorverarbeitung für OCR-Optimierung ohne problematische Dependencies.
Optimiert für Container-Umgebungen mit OpenCV + Pillow.

Hauptfeatures:
- Robustes Shadow/Background-Removal mit Clamping & weicher Mischung
- Mehrere Binarisierungs-Kandidaten (Adaptive + Otsu + Hybrid) und Auswahl nach Entropie
- Fail-safe: Fällt auf Graustufe+CLAHE zurück, wenn binär zu weiß/schwarz wäre
- Deskew (Hough + Contour) mit Konsens
- Sanfte Morphologie & Schärfung nur, wenn sinnvoll
"""

import io
import logging
from typing import Tuple, Optional, List
from enum import Enum
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance


# -----------------------------------------------------------------------------
# Konfiguration
# -----------------------------------------------------------------------------

class DocumentType(Enum):
    SCANNED = "scanned"
    PHOTOGRAPHED = "photographed"
    HANDWRITTEN = "handwritten"
    PRINTED = "printed"
    MIXED = "mixed"


@dataclass
class ProcessingConfig:
    # Größen/DPI
    target_width: int = 2400
    target_dpi: int = 300
    angle_limit: float = 10.0

    # Rauschunterdrückung
    noise_reduction_strength: float = 0.8
    bilateral_d: int = 5
    bilateral_sigma_color: float = 50
    bilateral_sigma_space: float = 50

    # Kontrast/Helligkeit
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (6, 6)
    gamma_correction: float = 1.1

    # Morphologie
    opening_kernel_size: int = 1
    closing_kernel_size: int = 2

    # Schärfung
    unsharp_radius: float = 0.8
    unsharp_percent: int = 140
    unsharp_threshold: int = 2

    # Features
    use_shadow_removal: bool = True
    use_deblurring: bool = False
    use_text_enhancement: bool = True
    min_text_height: int = 8


CONFIGS = {
    DocumentType.SCANNED: ProcessingConfig(
        noise_reduction_strength=0.5,
        gamma_correction=1.0,
        use_deblurring=False,
        use_shadow_removal=False,
        clahe_clip_limit=1.5,
        target_width=2200,
        use_text_enhancement=True,
    ),
    DocumentType.PHOTOGRAPHED: ProcessingConfig(
        noise_reduction_strength=1.0,
        gamma_correction=1.2,
        use_deblurring=False,
        use_shadow_removal=True,
        clahe_clip_limit=2.5,
        target_width=2600,
        use_text_enhancement=True,
    ),
    DocumentType.HANDWRITTEN: ProcessingConfig(
        target_width=2800,
        opening_kernel_size=1,
        closing_kernel_size=3,
        unsharp_percent=160,
        unsharp_threshold=1,
        clahe_clip_limit=1.8,
        use_shadow_removal=True,
        use_text_enhancement=True,
    ),
    DocumentType.PRINTED: ProcessingConfig(
        noise_reduction_strength=0.6,
        gamma_correction=1.05,
        unsharp_percent=130,
        use_shadow_removal=True,
        use_text_enhancement=True,
        target_width=2400,
    ),
    DocumentType.MIXED: ProcessingConfig(
        use_shadow_removal=True,
        use_text_enhancement=True,
        target_width=2400,
    ),
}

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Processor
# -----------------------------------------------------------------------------

class StableImageProcessor:
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()

    # ---------------------------- Erkennung/Skalierung ------------------------

    def detect_document_type(self, img_gray: np.ndarray) -> DocumentType:
        """Einfache Dokumenttyp-Erkennung nur mit OpenCV."""
        try:
            h, w = img_gray.shape

            edges = cv2.Canny(img_gray, 50, 150)
            edge_density = float(np.sum(edges > 0)) / edges.size

            laplacian_var = float(cv2.Laplacian(img_gray, cv2.CV_64F).var())
            mean_brightness = float(np.mean(img_gray))

            if edge_density < 0.02 and laplacian_var > 1000 and mean_brightness > 200:
                return DocumentType.SCANNED
            elif edge_density > 0.05 and laplacian_var < 500:
                return DocumentType.PHOTOGRAPHED
            elif mean_brightness < 150 and edge_density > 0.03:
                return DocumentType.HANDWRITTEN
            elif edge_density > 0.025 and laplacian_var > 800:
                return DocumentType.PRINTED
            else:
                return DocumentType.MIXED
        except Exception as e:
            logger.warning(f"Dokumenttyp-Erkennung fehlgeschlagen: {e}")
            return DocumentType.MIXED

    def intelligent_scaling_for_ocr(self, img: Image.Image) -> Image.Image:
        """
        Erhöht die Auflösung so, dass typische Zeilenhöhen ~20–25 px erreichen.
        """
        try:
            arr = np.array(img.convert("L"))
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(255 - arr, cv2.MORPH_OPEN, horizontal_kernel)
            contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                heights = []
                for c in contours:
                    _, _, _, h = cv2.boundingRect(c)
                    if h > 5:
                        heights.append(h)
                if heights:
                    avg_h = float(np.median(heights))
                    target = 22.0
                    if avg_h < target:
                        scale = min(target / avg_h, 3.0)
                        new_size = (int(img.width * scale), int(img.height * scale))
                        logger.info(f"OCR-Skalierung: {avg_h:.1f}px → {target}px (x{scale:.2f})")
                        return img.resize(new_size, Image.LANCZOS)

            if img.width < self.config.target_width:
                scale = self.config.target_width / img.width
                new_size = (int(img.width * scale), int(img.height * scale))
                return img.resize(new_size, Image.LANCZOS)

            return img
        except Exception as e:
            logger.warning(f"OCR-Skalierung fehlgeschlagen: {e}")
            return self._scale_image(img)

    # ----------------------------- Vorverarbeitung ----------------------------

    def stable_noise_reduction(self, img: np.ndarray) -> np.ndarray:
        """Strukturerhaltende Rauschunterdrückung."""
        try:
            denoised = cv2.bilateralFilter(
                img,
                self.config.bilateral_d,
                self.config.bilateral_sigma_color,
                self.config.bilateral_sigma_space,
            )
            if self.config.noise_reduction_strength > 1.0:
                k = int(self.config.noise_reduction_strength * 2) | 1
                denoised = cv2.GaussianBlur(denoised, (k, k), 0)
            return denoised
        except Exception as e:
            logger.warning(f"Rauschunterdrückung fehlgeschlagen: {e}")
            return img

    def remove_shadows_stable(self, img: np.ndarray) -> np.ndarray:
        """Robustes Shadow/Background-Removal (verhindert Weiß-Ausreißer)."""
        try:
            k = max(15, min(img.shape[:2]) // 30)
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            background = cv2.GaussianBlur(background, (0, 0), sigmaX=k / 6.0)

            safe_bg = np.clip(background.astype(np.float32), 10.0, 255.0)
            norm = (img.astype(np.float32) / safe_bg) * 128.0
            norm = np.clip(norm, 0, 255).astype(np.uint8)

            mixed = cv2.addWeighted(norm, 0.6, img, 0.4, 0)

            clahe = cv2.createCLAHE(
                clipLimit=max(1.2, self.config.clahe_clip_limit * 0.8),
                tileGridSize=self.config.clahe_tile_grid,
            )
            enhanced = clahe.apply(mixed)
            return enhanced
        except Exception as e:
            logger.warning(f"Beleuchtungskorrektur fehlgeschlagen: {e}")
            return img

    def simple_deblurring(self, img: np.ndarray) -> np.ndarray:
        """Einfache Entschärfung ohne zusätzliche Libs."""
        try:
            kernel = np.array(
                [[-1, -1, -1],
                 [-1,  9, -1],
                 [-1, -1, -1]]
            )
            sharp = cv2.filter2D(img, -1, kernel)
            return cv2.addWeighted(img, 0.4, sharp, 0.6, 0)
        except Exception as e:
            logger.warning(f"Entschärfung fehlgeschlagen: {e}")
            return img

    # ----------------------------- Deskew -------------------------------------

    def robust_skew_detection_stable(self, img: np.ndarray) -> float:
        angles: List[float] = []
        a1 = self._detect_skew_hough_stable(img)
        if a1 is not None:
            angles.append(a1)
        a2 = self._detect_skew_contours(img)
        if a2 is not None:
            angles.append(a2)
        if len(angles) >= 2:
            arr = np.array(angles, dtype=np.float32)
            med = np.median(arr)
            filt = arr[np.abs(arr - med) < 5]
            if len(filt) > 0:
                return float(np.mean(filt))
        return float(angles[0]) if angles else 0.0

    def _detect_skew_hough_stable(self, img: np.ndarray) -> Optional[float]:
        try:
            v = float(np.median(img))
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            edges = cv2.Canny(img, lower, upper)

            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=max(50, img.shape[1] // 15),
                minLineLength=img.shape[1] // 8,
                maxLineGap=img.shape[1] // 30,
            )
            if lines is None:
                return None

            angles: List[float] = []
            for line in lines[:100]:
                x1, y1, x2, y2 = line[0]
                ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if ang > 45:
                    ang -= 90
                elif ang < -45:
                    ang += 90
                if -15 <= ang <= 15:
                    angles.append(ang)
            return float(np.median(angles)) if len(angles) >= 3 else None
        except Exception as e:
            logger.debug(f"Hough-Skew fehlgeschlagen: {e}")
            return None

    def _detect_skew_contours(self, img: np.ndarray) -> Optional[float]:
        try:
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            angles: List[float] = []
            for c in contours:
                if cv2.contourArea(c) > 100:
                    rect = cv2.minAreaRect(c)
                    ang = float(rect[-1])
                    if ang < -45:
                        ang = 90 + ang
                    if -15 <= ang <= 15:
                        angles.append(ang)
            return float(np.median(angles)) if len(angles) >= 2 else None
        except Exception as e:
            logger.debug(f"Contour-Skew fehlgeschlagen: {e}")
            return None

    # ----------------------------- Binarisierung ------------------------------

    def _binary_entropy(self, b: np.ndarray) -> float:
        white = float(np.mean(b == 255))
        black = 1.0 - white
        eps = 1e-6
        return -(white * np.log2(white + eps) + black * np.log2(black + eps))

    def _pick_best_binary(self, candidates: List[np.ndarray]) -> np.ndarray:
        best = None
        best_score = -1e9
        for c in candidates:
            white_ratio = float(np.mean(c == 255))
            # gute Bandbreite bevorzugen
            if 0.2 <= white_ratio <= 0.9:
                score = self._binary_entropy(c)
            else:
                score = self._binary_entropy(c) - 1.0  # penalize
            if score > best_score:
                best_score = score
                best = c
        return best if best is not None else candidates[0]

    def multi_threshold_stable(self, img: np.ndarray) -> np.ndarray:
        """Erzeuge mehrere Kandidaten und wähle den besten."""
        try:
            h, w = img.shape[:2]
            blocks = []
            for bs in [15, 25, max(31, min(h, w) // 20), max(41, min(h, w) // 15)]:
                if bs % 2 == 0:
                    bs += 1
                blocks.append(bs)

            c_vals = [4, 8, 12]
            cands: List[np.ndarray] = []

            for bs in blocks:
                for C in c_vals:
                    cands.append(cv2.adaptiveThreshold(
                        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, bs, C
                    ))
                    cands.append(cv2.adaptiveThreshold(
                        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                        cv2.THRESH_BINARY, bs, C
                    ))

            _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cands.append(otsu)

            # Hybrid (Adaptive + Otsu)
            bs = max(31, min(h, w) // 18)
            if bs % 2 == 0:
                bs += 1
            adp = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, bs, 8)
            hybrid = cv2.addWeighted(adp, 0.6, otsu, 0.4, 0)
            cands.append((hybrid > 127).astype(np.uint8) * 255)

            return self._pick_best_binary(cands)
        except Exception as e:
            logger.error(f"Binarisierung fehlgeschlagen: {e}")
            _, result = cv2.threshold(img, np.mean(img), 255, cv2.THRESH_BINARY)
            return result

    # ----------------------------- Enhancement/Morph/Sharpen ------------------

    def enhance_text_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """Leichte Verstärkung von Buchstabenformen (nur auf binärem Bild)."""
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            dil = cv2.dilate(img, kernel, iterations=1)

            smooth = cv2.morphologyEx(
                dil, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            )

            vker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
            vert = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, vker)

            res = cv2.addWeighted(vert, 0.7, img, 0.3, 0)
            return res.astype(np.uint8)
        except Exception as e:
            logger.warning(f"Text-Enhancement fehlgeschlagen: {e}")
            return img

    def _morphology_cleanup(self, img: np.ndarray) -> np.ndarray:
        try:
            white_ratio = float(np.mean(img == 255))
            if white_ratio > 0.9:
                logger.warning(f"Bild bereits sehr weiß ({white_ratio:.2%}), Morphologie übersprungen.")
                return img

            if self.config.opening_kernel_size > 0:
                k = min(self.config.opening_kernel_size, 2)
                ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                img = cv2.morphologyEx(img, cv2.MORPH_OPEN, ker, iterations=1)

            white_ratio = float(np.mean(img == 255))
            if white_ratio > 0.9:
                return img

            if self.config.closing_kernel_size > 0:
                k = min(self.config.closing_kernel_size, 1)
                if k > 0:
                    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
                    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, ker, iterations=1)

            return img
        except Exception as e:
            logger.warning(f"Morphologie fehlgeschlagen: {e}")
            return img

    def _sharpen_pil(self, img_array: np.ndarray) -> Image.Image:
        try:
            pil_img = Image.fromarray(img_array)
            sharpened = pil_img.filter(
                ImageFilter.UnsharpMask(
                    radius=self.config.unsharp_radius,
                    percent=self.config.unsharp_percent,
                    threshold=self.config.unsharp_threshold,
                )
            )
            enhanced = sharpened.filter(ImageFilter.EDGE_ENHANCE_MORE)

            a = np.array(sharpened)
            b = np.array(enhanced)
            mixed = cv2.addWeighted(a, 0.8, b, 0.2, 0)
            mixed = np.clip(mixed, 0, 255).astype(np.uint8)

            result = Image.fromarray(mixed)
            enhancer = ImageEnhance.Contrast(result)
            return enhancer.enhance(1.05)
        except Exception as e:
            logger.warning(f"OCR-Schärfung fehlgeschlagen: {e}")
            return Image.fromarray(img_array)

    # ----------------------------- Hilfsfunktionen ----------------------------

    def _scale_image(self, img: Image.Image) -> Image.Image:
        if img.width < self.config.target_width:
            scale = self.config.target_width / img.width
            new_size = (int(img.width * scale), int(img.height * scale))
            return img.resize(new_size, Image.LANCZOS)
        return img

    def _apply_gamma(self, img: np.ndarray, gamma: float) -> np.ndarray:
        try:
            inv = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
            return cv2.LUT(img, table)
        except Exception:
            return img

    def _rotate_image_stable(self, img: np.ndarray, angle: float) -> np.ndarray:
        try:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            return cv2.warpAffine(
                img, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255
            )
        except Exception as e:
            logger.warning(f"Rotation fehlgeschlagen: {e}")
            return img

    # ----------------------------- Pipeline -----------------------------------

    def process_image(self, file_bytes: bytes) -> bytes:
        try:
            with Image.open(io.BytesIO(file_bytes)) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                logger.info(f"Originalgröße: {img.size}")

            # Skalierung
            img = self.intelligent_scaling_for_ocr(img)

            # nach OpenCV
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            # Doku-Typ
            doc_type = self.detect_document_type(gray)
            self.config = CONFIGS.get(doc_type, ProcessingConfig())
            logger.info(f"Dokumenttyp: {doc_type.value}")

            # Rauschen
            if self.config.noise_reduction_strength > 0:
                gray = self.stable_noise_reduction(gray)

            # Schatten
            if self.config.use_shadow_removal:
                gray = self.remove_shadows_stable(gray)

            # Entschärfung (optional)
            if self.config.use_deblurring:
                gray = self.simple_deblurring(gray)

            # Deskew
            angle = self.robust_skew_detection_stable(gray)
            angle = float(np.clip(angle, -self.config.angle_limit, self.config.angle_limit))
            if abs(angle) > 0.2:
                gray = self._rotate_image_stable(gray, angle)
                logger.info(f"Schräglage korrigiert: {angle:.2f}°")

            # Gamma
            if self.config.gamma_correction != 1.0:
                gray = self._apply_gamma(gray, self.config.gamma_correction)

            # Binarisierung (Kandidaten)
            binary = self.multi_threshold_stable(gray)
            white_ratio = float(np.mean(binary == 255))
            std_gray = float(np.std(gray))
            logger.info(f"Binarisierung: {white_ratio:.2%} weiß | std_gray={std_gray:.1f}")

            # Text-Enhancement nur bei sinnvollem Weißanteil
            if self.config.use_text_enhancement and (0.2 <= white_ratio <= 0.95):
                binary = self.enhance_text_for_ocr(binary)

            # Morphologie/Schärfung nur bei sinnvollem Weißanteil
            if 0.2 <= white_ratio <= 0.95:
                binary = self._morphology_cleanup(binary)
                final_img = self._sharpen_pil(binary)
            else:
                logger.warning("Binärbild außerhalb sinnvoller Grenzen – liefere Graustufe mit CLAHE.")
                clahe = cv2.createCLAHE(
                    clipLimit=max(1.2, self.config.clahe_clip_limit),
                    tileGridSize=self.config.clahe_tile_grid,
                )
                gray_clahe = clahe.apply(gray)
                final_img = Image.fromarray(gray_clahe)

            # PNG mit DPI
            buf = io.BytesIO()
            final_img.save(
                buf,
                format="PNG",
                optimize=True,
                compress_level=6,
                dpi=(self.config.target_dpi, self.config.target_dpi),
            )
            return buf.getvalue()

        except Exception as e:
            logger.error(f"Verarbeitungsfehler: {e}")
            # Fallback: Graustufe ohne viel Schnickschnack
            try:
                with Image.open(io.BytesIO(file_bytes)) as img:
                    img = ImageOps.exif_transpose(img).convert("L")
                    out = io.BytesIO()
                    img.save(out, format="PNG")
                    return out.getvalue()
            except Exception:
                raise e


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def preprocess_image_stable(file_bytes: bytes) -> bytes:
    """Hauptfunktion (stabil)."""
    proc = StableImageProcessor()
    return proc.process_image(file_bytes)


def preprocess_image(file_bytes: bytes) -> bytes:
    """Alias für Kompatibilität."""
    return preprocess_image_stable(file_bytes)
