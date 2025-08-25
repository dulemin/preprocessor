# app/preprocess.py
"""
Erweiterte Bildvorverarbeitung für hochpräzise OCR-Optimierung.
Kombiniert klassische Computer Vision mit modernen Techniken für maximale Erkennungsgenauigkeit.
"""

import io
import logging
from typing import Tuple, Optional, Union, List
from enum import Enum
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from scipy import ndimage
from skimage import filters, morphology, measure, restoration
from skimage.feature import peak_local_maxima

class DocumentType(Enum):
    """Erkannte Dokumenttypen für spezialisierte Verarbeitung."""
    SCANNED = "scanned"
    PHOTOGRAPHED = "photographed" 
    HANDWRITTEN = "handwritten"
    PRINTED = "printed"
    MIXED = "mixed"

@dataclass
class ProcessingConfig:
    """Erweiterte Konfiguration für verschiedene Dokumenttypen."""
    # Grundeinstellungen
    target_width: int = 2000
    target_dpi: int = 300
    angle_limit: float = 15.0
    
    # Rauschunterdrückung
    noise_reduction_strength: float = 1.2
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75
    bilateral_sigma_space: float = 75
    
    # Kontrast & Helligkeit
    clahe_clip_limit: float = 3.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    gamma_correction: float = 1.2
    
    # Morphologie
    opening_kernel_size: int = 2
    closing_kernel_size: int = 1
    
    # Schärfung
    unsharp_radius: float = 1.5
    unsharp_percent: int = 150
    unsharp_threshold: int = 3
    
    # Dokumentspezifische Anpassungen
    use_document_detection: bool = True
    use_text_line_detection: bool = True
    use_deblurring: bool = True
    use_shadow_removal: bool = True

# Konfigurationen für verschiedene Dokumenttypen
CONFIGS = {
    DocumentType.SCANNED: ProcessingConfig(
        noise_reduction_strength=0.8,
        gamma_correction=1.0,
        use_deblurring=False,
        use_shadow_removal=False
    ),
    DocumentType.PHOTOGRAPHED: ProcessingConfig(
        noise_reduction_strength=1.5,
        gamma_correction=1.3,
        use_deblurring=True,
        use_shadow_removal=True,
        clahe_clip_limit=4.0
    ),
    DocumentType.HANDWRITTEN: ProcessingConfig(
        target_width=2400,
        opening_kernel_size=1,
        closing_kernel_size=2,
        unsharp_percent=180,
        clahe_clip_limit=2.5
    ),
    DocumentType.PRINTED: ProcessingConfig(
        noise_reduction_strength=1.0,
        gamma_correction=1.1,
        unsharp_percent=120
    ),
    DocumentType.MIXED: ProcessingConfig()  # Standard-Config
}

logger = logging.getLogger(__name__)

class AdvancedImageProcessor:
    """Erweiterte Bildverarbeitung mit adaptiven Algorithmen."""
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
    
    def detect_document_type(self, img_gray: np.ndarray) -> DocumentType:
        """
        Automatische Erkennung des Dokumenttyps basierend auf Bildcharakteristika.
        
        Args:
            img_gray: Graustufenbild
            
        Returns:
            DocumentType: Erkannter Dokumenttyp
        """
        try:
            # Kantendichte analysieren
            edges = cv2.Canny(img_gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Rauschen analysieren
            laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
            
            # Textur analysieren
            glcm_contrast = filters.rank.enhance_contrast(
                img_gray, morphology.disk(5)
            ).std()
            
            # Entscheidungslogik
            if edge_density < 0.02 and laplacian_var > 1000:
                return DocumentType.SCANNED
            elif edge_density > 0.05 and laplacian_var < 500:
                return DocumentType.PHOTOGRAPHED
            elif glcm_contrast > 80:
                return DocumentType.HANDWRITTEN
            elif edge_density > 0.03 and laplacian_var > 800:
                return DocumentType.PRINTED
            else:
                return DocumentType.MIXED
                
        except Exception as e:
            logger.warning(f"Dokumenttyp-Erkennung fehlgeschlagen: {e}")
            return DocumentType.MIXED
    
    def advanced_noise_reduction(self, img: np.ndarray) -> np.ndarray:
        """
        Mehrstufige Rauschunterdrückung mit adaptiven Filtern.
        """
        # 1. Bilateral Filter für Kantenerhaltung
        denoised = cv2.bilateralFilter(
            img, 
            self.config.bilateral_d,
            self.config.bilateral_sigma_color,
            self.config.bilateral_sigma_space
        )
        
        # 2. Non-local Means Denoising für feine Details
        denoised = cv2.fastNlMeansDenoising(
            denoised, 
            None, 
            h=self.config.noise_reduction_strength * 10,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # 3. Morphologische Glättung für kleine Artefakte
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return denoised
    
    def remove_shadows_and_lighting(self, img: np.ndarray) -> np.ndarray:
        """
        Entfernt Schatten und ungleichmäßige Beleuchtung.
        """
        try:
            # Background-Schätzung mit morphologischer Öffnung
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
            background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            
            # Normalisierung durch Division
            normalized = cv2.divide(img, background, scale=255)
            
            # CLAHE für lokale Kontrastanpassung
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_tile_grid
            )
            enhanced = clahe.apply(normalized)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Schatten-/Beleuchtungskorrektur fehlgeschlagen: {e}")
            return img
    
    def advanced_deblurring(self, img: np.ndarray) -> np.ndarray:
        """
        Erweiterte Unschärfe-Korrektur mit Wiener-Filter.
        """
        try:
            # Motion Blur Erkennung und Korrektur
            kernel = np.ones((5, 5), np.float32) / 25
            deblurred = cv2.filter2D(img, -1, kernel)
            
            # Wiener Filter Approximation
            noise_var = 0.01
            psf = np.ones((5, 5)) / 25
            deblurred = restoration.wiener(img, psf, noise_var)
            deblurred = (deblurred * 255).astype(np.uint8)
            
            return deblurred
            
        except Exception as e:
            logger.warning(f"Entschärfung fehlgeschlagen: {e}")
            return img
    
    def detect_text_regions(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Erkennt Textregionen für gezielte Verarbeitung.
        """
        try:
            # MSER für Textregion-Erkennung
            mser = cv2.MSER_create(
                delta=5,
                min_area=60,
                max_area=14400,
                max_variation=0.25
            )
            
            regions, _ = mser.detectRegions(img)
            
            # Bounding Boxes berechnen
            text_boxes = []
            for region in regions:
                x, y, w, h = cv2.boundingRect(region)
                # Filtere zu kleine oder zu große Regionen
                if 20 < w < img.shape[1] // 2 and 8 < h < img.shape[0] // 4:
                    text_boxes.append((x, y, w, h))
            
            return text_boxes
            
        except Exception as e:
            logger.warning(f"Textregion-Erkennung fehlgeschlagen: {e}")
            return []
    
    def robust_skew_detection(self, img: np.ndarray) -> float:
        """
        Robuste Schräglage-Erkennung mit mehreren Methoden.
        """
        angles = []
        
        # Methode 1: Hough-Transform
        hough_angle = self._detect_skew_hough(img)
        if hough_angle is not None:
            angles.append(hough_angle)
        
        # Methode 2: Projektionsprofile
        projection_angle = self._detect_skew_projection(img)
        if projection_angle is not None:
            angles.append(projection_angle)
        
        # Methode 3: MinAreaRect auf Textregionen
        text_angle = self._detect_skew_text_regions(img)
        if text_angle is not None:
            angles.append(text_angle)
        
        # Konsens-basierte Entscheidung
        if len(angles) >= 2:
            # Entferne Ausreißer
            angles = np.array(angles)
            median_angle = np.median(angles)
            filtered_angles = angles[np.abs(angles - median_angle) < 5]
            
            if len(filtered_angles) > 0:
                return float(np.mean(filtered_angles))
        
        return angles[0] if angles else 0.0
    
    def _detect_skew_hough(self, img: np.ndarray) -> Optional[float]:
        """Hough-basierte Schräglage-Erkennung (erweitert)."""
        try:
            # Adaptive Kanten-Erkennung
            sigma = 0.33
            median_val = np.median(img)
            lower = int(max(0, (1.0 - sigma) * median_val))
            upper = int(min(255, (1.0 + sigma) * median_val))
            
            edges = cv2.Canny(img, lower, upper)
            
            # Erweiterte Hough-Parameter
            lines = cv2.HoughLinesP(
                edges, 
                rho=1, 
                theta=np.pi/360,  # Höhere Winkelauflösung
                threshold=max(50, img.shape[1] // 20),
                minLineLength=img.shape[1] // 6,
                maxLineGap=img.shape[1] // 40
            )
            
            if lines is None:
                return None
            
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                
                # Normalisierung und Filterung
                if angle > 45:
                    angle -= 90
                elif angle < -45:
                    angle += 90
                
                if -20 <= angle <= 20:  # Erweiterte Toleranz
                    angles.append(angle)
            
            if len(angles) >= 3:
                return float(np.median(angles))
                
        except Exception as e:
            logger.debug(f"Hough-Skew-Detection fehlgeschlagen: {e}")
        
        return None
    
    def _detect_skew_projection(self, img: np.ndarray) -> Optional[float]:
        """Projektionsprofil-basierte Schräglage-Erkennung."""
        try:
            # Binärisierung für bessere Projektion
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            best_angle = 0.0
            max_variance = 0.0
            
            # Teste verschiedene Winkel
            for angle in np.arange(-10, 11, 0.5):
                # Rotiere Bild
                h, w = binary.shape
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(binary, M, (w, h))
                
                # Horizontale Projektion
                h_projection = np.sum(rotated, axis=1)
                variance = np.var(h_projection)
                
                if variance > max_variance:
                    max_variance = variance
                    best_angle = angle
            
            return best_angle if max_variance > 1000 else None
            
        except Exception as e:
            logger.debug(f"Projektions-Skew-Detection fehlgeschlagen: {e}")
            return None
    
    def _detect_skew_text_regions(self, img: np.ndarray) -> Optional[float]:
        """Textregionen-basierte Schräglage-Erkennung."""
        try:
            text_boxes = self.detect_text_regions(img)
            
            if len(text_boxes) < 3:
                return None
            
            # Berechne Winkel zwischen Textboxen
            angles = []
            for i in range(len(text_boxes) - 1):
                x1, y1, w1, h1 = text_boxes[i]
                x2, y2, w2, h2 = text_boxes[i + 1]
                
                # Zentren der Boxen
                center1 = (x1 + w1 // 2, y1 + h1 // 2)
                center2 = (x2 + w2 // 2, y2 + h2 // 2)
                
                angle = np.degrees(np.arctan2(
                    center2[1] - center1[1],
                    center2[0] - center1[0]
                ))
                
                if -20 <= angle <= 20:
                    angles.append(angle)
            
            return float(np.median(angles)) if angles else None
            
        except Exception as e:
            logger.debug(f"Textregionen-Skew-Detection fehlgeschlagen: {e}")
            return None
    
    def adaptive_threshold_multi_method(self, img: np.ndarray) -> np.ndarray:
        """
        Adaptive Schwellwertbildung mit mehreren Methoden.
        """
        methods = []
        
        # Methode 1: Otsu
        _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        methods.append(otsu)
        
        # Methode 2: Adaptive Gaussian
        adaptive_gauss = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 31, 15
        )
        methods.append(adaptive_gauss)
        
        # Methode 3: Adaptive Mean
        adaptive_mean = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 31, 15
        )
        methods.append(adaptive_mean)
        
        # Methode 4: Sauvola (approximiert)
        try:
            # Lokale Statistiken
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            local_mean = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            
            # Sauvola-ähnliche Schwellwertbildung
            k = 0.2
            R = 128
            threshold = local_mean * (1 + k * ((img / R) - 1))
            sauvola = np.where(img > threshold, 255, 0).astype(np.uint8)
            methods.append(sauvola)
        except:
            pass
        
        # Ensemble-Entscheidung (Mehrheitsentscheidung)
        if len(methods) >= 3:
            combined = np.stack(methods, axis=2)
            result = np.where(np.mean(combined, axis=2) > 127, 255, 0).astype(np.uint8)
            return result
        
        return methods[0]  # Fallback
    
    def process_image(self, file_bytes: bytes) -> bytes:
        """
        Haupt-Verarbeitungspipeline mit adaptiven Algorithmen.
        """
        try:
            # 1. Bild laden und EXIF korrigieren
            with Image.open(io.BytesIO(file_bytes)) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                logger.info(f"Originalgröße: {img.size}")
            
            # 2. Intelligente Skalierung
            img = self._intelligent_scaling(img)
            
            # 3. Nach OpenCV konvertieren
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # 4. Dokumenttyp erkennen und Config anpassen
            doc_type = self.detect_document_type(img_gray)
            self.config = CONFIGS.get(doc_type, ProcessingConfig())
            logger.info(f"Erkannter Dokumenttyp: {doc_type.value}")
            
            # 5. Erweiterte Rauschunterdrückung
            if self.config.noise_reduction_strength > 0:
                img_gray = self.advanced_noise_reduction(img_gray)
            
            # 6. Schatten- und Beleuchtungskorrektur
            if self.config.use_shadow_removal:
                img_gray = self.remove_shadows_and_lighting(img_gray)
            
            # 7. Entschärfung bei fotografierten Dokumenten
            if self.config.use_deblurring:
                img_gray = self.advanced_deblurring(img_gray)
            
            # 8. Robuste Schräglage-Korrektur
            angle = self.robust_skew_detection(img_gray)
            angle = np.clip(angle, -self.config.angle_limit, self.config.angle_limit)
            
            if abs(angle) > 0.2:
                img_gray = self._rotate_image(img_gray, angle)
                logger.info(f"Schräglage korrigiert: {angle:.2f}°")
            
            # 9. Gamma-Korrektur
            if self.config.gamma_correction != 1.0:
                img_gray = self._apply_gamma_correction(img_gray, self.config.gamma_correction)
            
            # 10. Multi-Method Binarisierung
            binary_img = self.adaptive_threshold_multi_method(img_gray)
            
            # 11. Morphologische Nachbearbeitung
            binary_img = self._morphological_cleanup(binary_img)
            
            # 12. Erweiterte Nachschärfung
            final_img = self._advanced_sharpening(binary_img)
            
            # 13. Qualitätskontrolle
            quality_score = self._assess_quality(final_img)
            logger.info(f"Qualitätsscore: {quality_score:.2f}")
            
            # 14. PNG-Export
            output_buffer = io.BytesIO()
            final_img.save(output_buffer, format="PNG", optimize=True, compress_level=6)
            
            result_bytes = output_buffer.getvalue()
            logger.info(f"Verarbeitung abgeschlossen. Ausgabe: {len(result_bytes)} bytes")
            
            return result_bytes
            
        except Exception as e:
            logger.error(f"Kritischer Fehler bei Bildverarbeitung: {e}")
            raise
    
    def _intelligent_scaling(self, img: Image.Image) -> Image.Image:
        """Intelligente Skalierung basierend auf Bildinhalt."""
        # DPI-basierte Skalierung wenn verfügbar
        if hasattr(img, 'info') and 'dpi' in img.info:
            current_dpi = img.info['dpi'][0]
            if current_dpi < self.config.target_dpi:
                scale_factor = self.config.target_dpi / current_dpi
                new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
                return img.resize(new_size, Image.LANCZOS)
        
        # Fallback: Breiten-basierte Skalierung
        if img.width < self.config.target_width:
            scale_factor = self.config.target_width / img.width
            new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
            return img.resize(new_size, Image.LANCZOS)
        
        return img
    
    def _apply_gamma_correction(self, img: np.ndarray, gamma: float) -> np.ndarray:
        """Gamma-Korrektur für Helligkeitsanpassung."""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype(np.uint8)
        return cv2.LUT(img, table)
    
    def _rotate_image(self, img: np.ndarray, angle: float) -> np.ndarray:
        """Verbesserte Bildrotation mit Randbehandlung."""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Berechne neue Bildgröße nach Rotation
        cos_val = abs(M[0, 0])
        sin_val = abs(M[0, 1])
        new_w = int((h * sin_val) + (w * cos_val))
        new_h = int((h * cos_val) + (w * sin_val))
        
        # Anpassung der Translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        return cv2.warpAffine(
            img, M, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255  # Weißer Hintergrund
        )
    
    def _morphological_cleanup(self, img: np.ndarray) -> np.ndarray:
        """Erweiterte morphologische Bereinigung."""
        # Opening für Rauschentfernung
        if self.config.opening_kernel_size > 0:
            kernel_open = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (self.config.opening_kernel_size, self.config.opening_kernel_size)
            )
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_open)
        
        # Closing für Lückenschließung
        if self.config.closing_kernel_size > 0:
            kernel_close = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (self.config.closing_kernel_size, self.config.closing_kernel_size)
            )
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_close)
        
        return img
    
    def _advanced_sharpening(self, img_array: np.ndarray) -> Image.Image:
        """Erweiterte Nachschärfung mit mehreren Techniken."""
        pil_img = Image.fromarray(img_array)
        
        # Unsharp Mask
        sharpened = pil_img.filter(
            ImageFilter.UnsharpMask(
                radius=self.config.unsharp_radius,
                percent=self.config.unsharp_percent,
                threshold=self.config.unsharp_threshold
            )
        )
        
        # Zusätzliche Kontrastverbesserung
        enhancer = ImageEnhance.Contrast(sharpened)
        enhanced = enhancer.enhance(1.1)
        
        return enhanced
    
    def _assess_quality(self, img: Image.Image) -> float:
        """Bewertet die Qualität des verarbeiteten Bildes."""
        try:
            img_array = np.array(img)
            
            # Kontrast-Messung
            contrast = img_array.std()
            
            # Schärfe-Messung (Laplacian Variance)
            gray = img_array if len(img_array.shape) == 2 else cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalisierte Qualitätsbewertung
            quality = min(100, (contrast / 50 + sharpness / 1000) * 20)
            
            return quality
            
        except Exception:
            return 50.0  # Fallback-Wert


def preprocess_image_advanced(file_bytes: bytes) -> bytes:
    """
    Erweiterte Hauptfunktion für OCR-optimierte Bildverarbeitung.
    
    Args:
        file_bytes: Eingabebild als Bytes
        
    Returns:
        bytes: Optimiertes PNG-Bild für OCR
    """
    processor = AdvancedImageProcessor()
    return processor.process_image(file_bytes)


# Alias für Rückwärtskompatibilität
preprocess_image = preprocess_image_advanced
