# app/preprocess.py
"""
Bildvorverarbeitung für OCR-Optimierung.
Enthält Funktionen für Orientierungskorrektur, Skalierung, Deskewing und Schwellwertbildung.
"""

import io
import logging
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter

# Konfiguration
CONFIG = {
    'target_width': 1800,
    'angle_limit': 12.0,
    'min_line_length_ratio': 0.25,
    'hough_threshold': 80,
    'hough_max_gap': 20,
    'max_lines': 2000,
    'min_angles_for_hough': 5,
    'horizontal_angle_range': 30,
    'median_blur_ksize': 3,
    'adaptive_thresh_block_size': 31,
    'adaptive_thresh_c': 15,
    'unsharp_radius': 1.0,
    'unsharp_percent': 140,
    'unsharp_threshold': 3
}

logger = logging.getLogger(__name__)


def pil_to_cv2_bgr(pil_img: Image.Image) -> np.ndarray:
    """
    Konvertiert PIL RGB-Bild zu OpenCV BGR-Format.
    
    Args:
        pil_img: PIL-Image im RGB-Format
        
    Returns:
        numpy.ndarray: Bild im BGR-Format (uint8)
    """
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def detect_skew_hough(img_gray: np.ndarray) -> Optional[float]:
    """
    Erkennt Schräglage mittels Hough-Transformation für Linien.
    
    Args:
        img_gray: Graustufenbild
        
    Returns:
        Optional[float]: Erkannter Winkel in Grad oder None
    """
    try:
        # Kantenerkennung
        edges = cv2.Canny(img_gray, 60, 180)
        h, w = img_gray.shape[:2]
        min_len = int(CONFIG['min_line_length_ratio'] * min(h, w))
        
        # Hough-Linien finden
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi / 180,
            threshold=CONFIG['hough_threshold'],
            minLineLength=min_len, 
            maxLineGap=CONFIG['hough_max_gap']
        )
        
        if lines is None:
            return None
            
        angles = []
        for line in lines[:CONFIG['max_lines']]:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # Nur nahezu horizontale Linien berücksichtigen
            if -CONFIG['horizontal_angle_range'] <= angle <= CONFIG['horizontal_angle_range']:
                angles.append(angle)
        
        if len(angles) >= CONFIG['min_angles_for_hough']:
            return float(np.median(angles))
            
    except Exception as e:
        logger.warning(f"Hough-basierte Schräglage-Erkennung fehlgeschlagen: {e}")
    
    return None


def detect_skew_minarea(img_gray: np.ndarray) -> float:
    """
    Fallback-Methode zur Schräglage-Erkennung mittels minAreaRect.
    
    Args:
        img_gray: Graustufenbild
        
    Returns:
        float: Erkannter Winkel in Grad
    """
    try:
        # Otsu-Schwellwert mit Morphologie
        _, binary = cv2.threshold(
            img_gray, 0, 255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Koordinaten der weißen Pixel finden
        coords = np.column_stack(np.where(binary > 0))
        
        if coords.size == 0:
            return 0.0
        
        # Minimales umschließendes Rechteck
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]  # Winkel in [-90, 0)
        
        # Normalisierung auf [-45, 45)
        if angle < -45:
            angle = 90 + angle
            
        return float(angle)
        
    except Exception as e:
        logger.warning(f"MinAreaRect-basierte Schräglage-Erkennung fehlgeschlagen: {e}")
        return 0.0


def compute_skew_angle(img_gray: np.ndarray) -> float:
    """
    Robuste Schräglage-Erkennung mit Hough-Transformation und Fallback.
    
    Args:
        img_gray: Graustufenbild
        
    Returns:
        float: Erkannter Winkel in Grad (positiv = gegen Uhrzeigersinn)
    """
    # Primärer Ansatz: Hough-Transformation
    angle = detect_skew_hough(img_gray)
    
    if angle is not None:
        logger.debug(f"Hough-basierte Schräglage erkannt: {angle:.2f}°")
        return angle
    
    # Fallback: minAreaRect
    angle = detect_skew_minarea(img_gray)
    logger.debug(f"MinAreaRect-basierte Schräglage erkannt: {angle:.2f}°")
    return angle


def scale_image_if_needed(img: Image.Image, target_width: int) -> Image.Image:
    """
    Skaliert Bild auf Zielbreite, falls es kleiner ist.
    
    Args:
        img: PIL-Image
        target_width: Gewünschte Mindestbreite
        
    Returns:
        Image.Image: Skaliertes Bild
    """
    if img.width >= target_width:
        return img
    
    scale_factor = target_width / img.width
    new_size = (
        int(img.width * scale_factor),
        int(img.height * scale_factor)
    )
    
    logger.debug(f"Skaliere Bild von {img.size} auf {new_size}")
    return img.resize(new_size, Image.LANCZOS)


def rotate_image(img_gray: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotiert Bild um gegebenen Winkel.
    
    Args:
        img_gray: Graustufenbild
        angle: Rotationswinkel in Grad
        
    Returns:
        np.ndarray: Rotiertes Bild
    """
    h, w = img_gray.shape[:2]
    center = (w / 2, h / 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    return cv2.warpAffine(
        img_gray, 
        rotation_matrix, 
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


def apply_adaptive_threshold(img_gray: np.ndarray) -> np.ndarray:
    """
    Wendet adaptive Schwellwertbildung mit Nachbearbeitung an.
    
    Args:
        img_gray: Graustufenbild
        
    Returns:
        np.ndarray: Binärbild
    """
    # Adaptive Schwellwertbildung
    binary = cv2.adaptiveThreshold(
        img_gray, 
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=CONFIG['adaptive_thresh_block_size'],
        C=CONFIG['adaptive_thresh_c']
    )
    
    # Kleine morphologische Operation zum Schließen von Lücken
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    binary = cv2.morphologyEx(
        binary, 
        cv2.MORPH_CLOSE, 
        kernel, 
        iterations=1
    )
    
    return binary


def apply_sharpening(img_array: np.ndarray) -> Image.Image:
    """
    Wendet Unsharp-Mask-Filter für Nachschärfung an.
    
    Args:
        img_array: Binärbild als numpy-Array
        
    Returns:
        Image.Image: Geschärftes PIL-Bild
    """
    pil_img = Image.fromarray(img_array)
    
    return pil_img.filter(
        ImageFilter.UnsharpMask(
            radius=CONFIG['unsharp_radius'],
            percent=CONFIG['unsharp_percent'],
            threshold=CONFIG['unsharp_threshold']
        )
    )


def preprocess_image(file_bytes: bytes) -> bytes:
    """
    Hauptfunktion für die Bildvorverarbeitung zur OCR-Optimierung.
    
    Verarbeitungsschritte:
    1. EXIF-Orientierung respektieren
    2. Hochskalierung bei Bedarf (bessere OCR-Qualität)
    3. Schräglage-Korrektur mit robuster Erkennung
    4. Adaptive Schwellwertbildung
    5. Nachschärfung
    
    Args:
        file_bytes: Bilddaten als Bytes
        
    Returns:
        bytes: Verarbeitetes PNG-Bild (Graustufenbild)
        
    Raises:
        Exception: Bei kritischen Verarbeitungsfehlern
    """
    try:
        # 1. Bild laden und EXIF-Orientierung korrigieren
        with Image.open(io.BytesIO(file_bytes)) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            logger.debug(f"Originalgröße: {img.size}")
        
        # 2. Bei Bedarf hochskalieren
        img = scale_image_if_needed(img, CONFIG['target_width'])
        
        # 3. Nach OpenCV-Format konvertieren und glätten
        img_bgr = pil_to_cv2_bgr(img)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.medianBlur(img_gray, CONFIG['median_blur_ksize'])
        
        # 4. Schräglage erkennen und korrigieren
        detected_angle = compute_skew_angle(img_gray)
        
        # Winkel begrenzen um Fehlrotationen zu vermeiden
        corrected_angle = np.clip(
            detected_angle, 
            -CONFIG['angle_limit'], 
            CONFIG['angle_limit']
        )
        
        if abs(corrected_angle) > 0.1:
            logger.debug(f"Rotiere um {corrected_angle:.2f}°")
            img_gray = rotate_image(img_gray, corrected_angle)
        
        # 5. Adaptive Schwellwertbildung
        binary_img = apply_adaptive_threshold(img_gray)
        
        # 6. Nachschärfung
        final_img = apply_sharpening(binary_img)
        
        # 7. Als optimiertes PNG zurückgeben
        output_buffer = io.BytesIO()
        final_img.save(output_buffer, format="PNG", optimize=True)
        
        result_bytes = output_buffer.getvalue()
        logger.info(f"Verarbeitung abgeschlossen. Ausgabegröße: {len(result_bytes)} bytes")
        
        return result_bytes
        
    except Exception as e:
        logger.error(f"Fehler bei Bildvorverarbeitung: {e}")
        raise
