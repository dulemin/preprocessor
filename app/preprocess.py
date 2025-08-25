# app/preprocess.py
"""
Stabile Bildvorverarbeitung für OCR-Optimierung ohne problematische Dependencies.
Optimiert für Container-Umgebungen mit nur Standard-Bibliotheken.
"""

import io
import logging
from typing import Tuple, Optional, List
from enum import Enum
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

class DocumentType(Enum):
    """Erkannte Dokumenttypen für spezialisierte Verarbeitung."""
    SCANNED = "scanned"
    PHOTOGRAPHED = "photographed" 
    HANDWRITTEN = "handwritten"
    PRINTED = "printed"
    MIXED = "mixed"

@dataclass
class ProcessingConfig:
    """Konfiguration für verschiedene Dokumenttypen."""
    # Grundeinstellungen
    target_width: int = 1800  # Reduziert von 2000
    angle_limit: float = 10.0  # Reduziert von 15.0
    
    # Rauschunterdrückung
    noise_reduction_strength: float = 0.8  # Reduziert von 1.2
    bilateral_d: int = 5  # Reduziert von 9
    bilateral_sigma_color: float = 50  # Reduziert von 75
    bilateral_sigma_space: float = 50  # Reduziert von 75
    
    # Kontrast & Helligkeit
    clahe_clip_limit: float = 2.0  # Reduziert von 3.0
    clahe_tile_grid: Tuple[int, int] = (6, 6)  # Reduziert von (8, 8)
    gamma_correction: float = 1.1  # Reduziert von 1.2
    
    # Morphologie (sehr sanft)
    opening_kernel_size: int = 1  # Reduziert von 2
    closing_kernel_size: int = 1  # Bleibt bei 1
    
    # Schärfung (sanfter)
    unsharp_radius: float = 1.0  # Reduziert von 1.5
    unsharp_percent: int = 120  # Reduziert von 150
    unsharp_threshold: int = 5  # Erhöht von 3
    
    # Features
    use_shadow_removal: bool = True  # Wieder aktiviert für Beleuchtungskorrektur
    use_deblurring: bool = False  # Bleibt aus

# Konfigurationen für verschiedene Dokumenttypen
CONFIGS = {
    DocumentType.SCANNED: ProcessingConfig(
        noise_reduction_strength=0.5,
        gamma_correction=1.0,
        use_deblurring=False,
        use_shadow_removal=False,  # Gescannte haben meist gleichmäßige Beleuchtung
        clahe_clip_limit=1.5
    ),
    DocumentType.PHOTOGRAPHED: ProcessingConfig(
        noise_reduction_strength=1.0,
        gamma_correction=1.2,
        use_deblurring=False,  # Bleibt aus
        use_shadow_removal=True,  # Wichtig für Fotos!
        clahe_clip_limit=2.5
    ),
    DocumentType.HANDWRITTEN: ProcessingConfig(
        target_width=2000,
        opening_kernel_size=1,
        closing_kernel_size=1,
        unsharp_percent=140,
        clahe_clip_limit=1.8,
        use_shadow_removal=True  # Oft ungleichmäßig beleuchtet
    ),
    DocumentType.PRINTED: ProcessingConfig(
        noise_reduction_strength=0.6,
        gamma_correction=1.05,
        unsharp_percent=110,
        use_shadow_removal=True  # Auch für gedruckte Dokumente
    ),
    DocumentType.MIXED: ProcessingConfig(
        use_shadow_removal=True  # Standard aktiviert
    )
}

logger = logging.getLogger(__name__)

class StableImageProcessor:
    """Stabile Bildverarbeitung nur mit OpenCV und PIL."""
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
    
    def detect_document_type(self, img_gray: np.ndarray) -> DocumentType:
        """
        Einfache Dokumenttyp-Erkennung nur mit OpenCV.
        """
        try:
            h, w = img_gray.shape
            
            # Kantendichte
            edges = cv2.Canny(img_gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Rauschmaß (Laplacian Variance)
            laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
            
            # Helligkeit
            mean_brightness = np.mean(img_gray)
            
            # Einfache Klassifikation
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
    
    def stable_noise_reduction(self, img: np.ndarray) -> np.ndarray:
        """Stabile Rauschunterdrückung nur mit OpenCV."""
        try:
            # Bilateral Filter
            denoised = cv2.bilateralFilter(
                img, 
                self.config.bilateral_d,
                self.config.bilateral_sigma_color,
                self.config.bilateral_sigma_space
            )
            
            # Zusätzliches Gaussian Blur bei starkem Rauschen
            if self.config.noise_reduction_strength > 1.0:
                kernel_size = int(self.config.noise_reduction_strength * 2) | 1  # ungerade Zahl
                denoised = cv2.GaussianBlur(denoised, (kernel_size, kernel_size), 0)
            
            return denoised
            
        except Exception as e:
            logger.warning(f"Rauschunterdrückung fehlgeschlagen: {e}")
            return img
    
    def remove_shadows_stable(self, img: np.ndarray) -> np.ndarray:
        """Sanfte Schatten-/Beleuchtungskorrektur für ungleichmäßige Beleuchtung."""
        try:
            # Kleinerer Kernel für sanftere Background-Schätzung
            kernel_size = min(img.shape[0], img.shape[1]) // 30  # Kleiner als vorher (war //20)
            kernel_size = max(kernel_size, 15)  # Mindestgröße
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            
            # Sanftere Division durch Gewichtung mit Original
            normalized = cv2.divide(img, background, scale=255)
            
            # Gewichtete Mischung: 70% korrigiert, 30% original
            alpha = 0.7
            mixed = cv2.addWeighted(normalized, alpha, img, 1-alpha, 0)
            
            # Sanftere CLAHE
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit * 0.8,  # 20% sanfter
                tileGridSize=self.config.clahe_tile_grid
            )
            enhanced = clahe.apply(mixed.astype(np.uint8))
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Beleuchtungskorrektur fehlgeschlagen: {e}")
            return img
    
    def simple_deblurring(self, img: np.ndarray) -> np.ndarray:
        """Einfache Entschärfung ohne scipy."""
        try:
            # Kernel für Sharpening
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            
            sharpened = cv2.filter2D(img, -1, kernel)
            
            # Gewichtete Mischung mit Original
            alpha = 0.6
            result = cv2.addWeighted(img, 1-alpha, sharpened, alpha, 0)
            
            return result
            
        except Exception as e:
            logger.warning(f"Entschärfung fehlgeschlagen: {e}")
            return img
    
    def robust_skew_detection_stable(self, img: np.ndarray) -> float:
        """Robuste Schräglage-Erkennung nur mit OpenCV."""
        angles = []
        
        # Methode 1: Hough Lines
        hough_angle = self._detect_skew_hough_stable(img)
        if hough_angle is not None:
            angles.append(hough_angle)
        
        # Methode 2: Contour-basiert
        contour_angle = self._detect_skew_contours(img)
        if contour_angle is not None:
            angles.append(contour_angle)
        
        # Konsens
        if len(angles) >= 2:
            angles = np.array(angles)
            median_angle = np.median(angles)
            filtered = angles[np.abs(angles - median_angle) < 5]
            if len(filtered) > 0:
                return float(np.mean(filtered))
        
        return angles[0] if angles else 0.0
    
    def _detect_skew_hough_stable(self, img: np.ndarray) -> Optional[float]:
        """Stabile Hough-basierte Schräglage-Erkennung."""
        try:
            # Adaptive Canny
            v = np.median(img)
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            
            edges = cv2.Canny(img, lower, upper)
            
            # Hough Lines
            lines = cv2.HoughLinesP(
                edges, 
                rho=1, 
                theta=np.pi/180,
                threshold=max(50, img.shape[1] // 15),
                minLineLength=img.shape[1] // 8,
                maxLineGap=img.shape[1] // 30
            )
            
            if lines is None:
                return None
            
            angles = []
            for line in lines[:100]:  # Begrenzt für Performance
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                
                # Normalisierung
                if angle > 45:
                    angle -= 90
                elif angle < -45:
                    angle += 90
                
                if -15 <= angle <= 15:
                    angles.append(angle)
            
            return float(np.median(angles)) if len(angles) >= 3 else None
            
        except Exception as e:
            logger.debug(f"Hough-Skew fehlgeschlagen: {e}")
            return None
    
    def _detect_skew_contours(self, img: np.ndarray) -> Optional[float]:
        """Contour-basierte Schräglage-Erkennung."""
        try:
            # Binärisierung
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Contours finden
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Größte Contours
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            
            angles = []
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    rect = cv2.minAreaRect(contour)
                    angle = rect[-1]
                    
                    # Normalisierung
                    if angle < -45:
                        angle = 90 + angle
                    
                    if -15 <= angle <= 15:
                        angles.append(angle)
            
            return float(np.median(angles)) if len(angles) >= 2 else None
            
        except Exception as e:
            logger.debug(f"Contour-Skew fehlgeschlagen: {e}")
            return None
    
    def multi_threshold_stable(self, img: np.ndarray) -> np.ndarray:
        """Verbesserte Binarisierung für ungleichmäßig beleuchtete Bilder."""
        try:
            # Für ungleichmäßige Beleuchtung: Größeres Fenster und sanftere Parameter
            block_size = max(31, min(img.shape) // 15)  # Adaptieve Fenstergröße
            if block_size % 2 == 0:  # Muss ungerade sein
                block_size += 1
            
            # Sanfteres C (weniger aggressiv)
            C = 8  # Sanfter als 10
            
            result = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, block_size, C
            )
            
            # Prüfe Ergebnis
            white_ratio = np.sum(result == 255) / result.size
            black_ratio = np.sum(result == 0) / result.size
            
            logger.info(f"Binarisierung: {white_ratio:.2%} weiß, {black_ratio:.2%} schwarz")
            
            # Wenn immer noch zu extrem, verwende gemischten Ansatz
            if white_ratio > 0.95 or black_ratio > 0.95:
                logger.warning("Adaptive Threshold zu extrem, verwende gemischten Ansatz")
                
                # Kombiniere Otsu mit Adaptive
                _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # 60% Adaptive, 40% Otsu
                alpha = 0.6
                result = cv2.addWeighted(result, alpha, otsu, 1-alpha, 0)
                result = np.where(result > 127, 255, 0).astype(np.uint8)
            
            return result
            
        except Exception as e:
            logger.error(f"Binarisierung fehlgeschlagen: {e}")
            # Notfall-Fallback
            _, result = cv2.threshold(img, np.mean(img), 255, cv2.THRESH_BINARY)
            return result
    
    def process_image(self, file_bytes: bytes) -> bytes:
        """
        Stabile Haupt-Pipeline ohne problematische Dependencies.
        """
        try:
            # 1. Bild laden
            with Image.open(io.BytesIO(file_bytes)) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                logger.info(f"Originalgröße: {img.size}")
            
            # 2. Skalierung
            img = self._scale_image(img)
            
            # 3. OpenCV Konvertierung
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # 4. Dokumenttyp erkennen
            doc_type = self.detect_document_type(img_gray)
            self.config = CONFIGS.get(doc_type, ProcessingConfig())
            logger.info(f"Dokumenttyp: {doc_type.value}")
            
            # 5. Rauschunterdrückung
            if self.config.noise_reduction_strength > 0:
                img_gray = self.stable_noise_reduction(img_gray)
            
            # 6. Schatten/Beleuchtung
            if self.config.use_shadow_removal:
                img_gray = self.remove_shadows_stable(img_gray)
            
            # 7. Entschärfung
            if self.config.use_deblurring:
                img_gray = self.simple_deblurring(img_gray)
            
            # 8. Schräglage korrigieren
            angle = self.robust_skew_detection_stable(img_gray)
            angle = np.clip(angle, -self.config.angle_limit, self.config.angle_limit)
            
            if abs(angle) > 0.2:
                img_gray = self._rotate_image_stable(img_gray, angle)
                logger.info(f"Schräglage: {angle:.2f}°")
            
            # 9. Gamma-Korrektur
            if self.config.gamma_correction != 1.0:
                img_gray = self._apply_gamma(img_gray, self.config.gamma_correction)
            
            # 10. Sanfte Binarisierung
            binary_img = self.multi_threshold_stable(img_gray)
            
            # Prüfe Ergebnis vor weiterer Verarbeitung
            white_ratio = np.sum(binary_img == 255) / binary_img.size
            logger.info(f"Binarisierung: {white_ratio:.2%} weiße Pixel")
            
            # Wenn zu weiß, überspringe Morphologie und Schärfung
            if white_ratio < 0.95:
                # 11. Sehr sanfte Morphologie
                binary_img = self._morphology_cleanup(binary_img)
                
                # 12. Sanfte Schärfung
                final_img = self._sharpen_pil(binary_img)
            else:
                logger.warning("Bild zu weiß, überspringe Nachbearbeitung")
                # Versuche weniger aggressive Binarisierung
                threshold_val = np.percentile(img_gray, 20)  # 20% Perzentil als Schwellwert
                _, binary_img = cv2.threshold(img_gray, threshold_val, 255, cv2.THRESH_BINARY)
                final_img = Image.fromarray(binary_img)
            
            # 13. Export
            output_buffer = io.BytesIO()
            final_img.save(output_buffer, format="PNG", optimize=True, compress_level=6)
            
            result_bytes = output_buffer.getvalue()
            logger.info(f"Verarbeitung OK. Größe: {len(result_bytes)} bytes")
            
            return result_bytes
            
        except Exception as e:
            logger.error(f"Verarbeitungsfehler: {e}")
            # Fallback: Originalbild zurückgeben
            try:
                with Image.open(io.BytesIO(file_bytes)) as img:
                    img = ImageOps.exif_transpose(img).convert("L")
                    output = io.BytesIO()
                    img.save(output, format="PNG")
                    return output.getvalue()
            except:
                raise e
    
    def _scale_image(self, img: Image.Image) -> Image.Image:
        """Einfache Skalierung."""
        if img.width < self.config.target_width:
            scale = self.config.target_width / img.width
            new_size = (int(img.width * scale), int(img.height * scale))
            return img.resize(new_size, Image.LANCZOS)
        return img
    
    def _apply_gamma(self, img: np.ndarray, gamma: float) -> np.ndarray:
        """Gamma-Korrektur."""
        try:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                             for i in range(256)]).astype(np.uint8)
            return cv2.LUT(img, table)
        except:
            return img
    
    def _rotate_image_stable(self, img: np.ndarray, angle: float) -> np.ndarray:
        """Stabile Bildrotation."""
        try:
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            return cv2.warpAffine(
                img, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255
            )
        except Exception as e:
            logger.warning(f"Rotation fehlgeschlagen: {e}")
            return img
    
    def _morphology_cleanup(self, img: np.ndarray) -> np.ndarray:
        """Sehr sanfte morphologische Bereinigung."""
        try:
            # Prüfe zuerst ob Bild nicht schon zu weiß ist
            white_ratio = np.sum(img == 255) / img.size
            if white_ratio > 0.9:
                logger.warning(f"Bild bereits zu weiß ({white_ratio:.2f}), überspringe Morphologie")
                return img
            
            # Nur sehr kleine Kerne verwenden
            if self.config.opening_kernel_size > 0:
                kernel_size = min(self.config.opening_kernel_size, 2)  # Maximal 2x2
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Prüfe wieder nach Opening
            white_ratio = np.sum(img == 255) / img.size
            if white_ratio > 0.9:
                logger.warning("Nach Opening zu weiß, überspringe Closing")
                return img
            
            # Sehr sanftes Closing
            if self.config.closing_kernel_size > 0:
                kernel_size = min(self.config.closing_kernel_size, 1)  # Maximal 1x1
                if kernel_size > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
                    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            return img
            
        except Exception as e:
            logger.warning(f"Morphologie fehlgeschlagen: {e}")
            return img
    
    def _sharpen_pil(self, img_array: np.ndarray) -> Image.Image:
        """PIL-basierte Schärfung."""
        try:
            pil_img = Image.fromarray(img_array)
            
            # Unsharp Mask
            sharpened = pil_img.filter(
                ImageFilter.UnsharpMask(
                    radius=self.config.unsharp_radius,
                    percent=self.config.unsharp_percent,
                    threshold=self.config.unsharp_threshold
                )
            )
            
            # Kontrast
            enhancer = ImageEnhance.Contrast(sharpened)
            enhanced = enhancer.enhance(1.05)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Schärfung fehlgeschlagen: {e}")
            return Image.fromarray(img_array)


def preprocess_image_stable(file_bytes: bytes) -> bytes:
    """
    Stabile Hauptfunktion ohne problematische Dependencies.
    """
    try:
        processor = StableImageProcessor()
        return processor.process_image(file_bytes)
    except Exception as e:
        logger.error(f"Kritischer Fehler: {e}")
        # Minimaler Fallback
        try:
            with Image.open(io.BytesIO(file_bytes)) as img:
                img = ImageOps.exif_transpose(img).convert("L")
                output = io.BytesIO()
                img.save(output, format="PNG")
                return output.getvalue()
        except:
            raise e


# Hauptfunktion für Rückwärtskompatibilität
def preprocess_image(file_bytes: bytes) -> bytes:
    """Hauptfunktion - Container-stabil."""
    return preprocess_image_stable(file_bytes)
