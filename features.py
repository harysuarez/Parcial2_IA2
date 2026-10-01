# features.py
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

def preprocess_image(img_rgb):
    """
    Preprocesamiento invariante a resolución, escala y color:
    1. Redimensionar a 80x80.
    2. Convertir a Gris (elimina el sesgo de agua marrón/verde/azul).
    3. CLAHE para normalizar contrastes en barcos blancos, oscuros o con sombra.
    """
    if img_rgb.shape[0] != 80 or img_rgb.shape[1] != 80:
        img_rgb = cv2.resize(img_rgb, (80, 80), interpolation=cv2.INTER_AREA)
        
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Filtro mediano para limpiar ruido de compresión manteniendo bordes de contenedores
    denoised = cv2.medianBlur(gray, 3)
    
    # CLAHE de alto contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(denoised)
    
    return enhanced_gray

def extract_hog_features(img_gray):
    """
    HOG optimizado: 12 orientaciones (cubre diagonales de 15 en 15 grados)
    """
    hog_feats = hog(
        img_gray,
        orientations=12,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True
    )
    return hog_feats

def extract_lbp_features(img_gray):
    """
    LBP multirradio: analiza microtexturas de olas vs superficies metálicas.
    """
    # Radio 1 (detalle fino)
    lbp1 = local_binary_pattern(img_gray, P=8, R=1, method="uniform")
    h1, _ = np.histogram(lbp1.ravel(), bins=10, range=(0, 10), density=True)
    
    # Radio 2 (estructura media)
    lbp2 = local_binary_pattern(img_gray, P=8, R=2, method="uniform")
    h2, _ = np.histogram(lbp2.ravel(), bins=10, range=(0, 10), density=True)
    
    return np.hstack([h1, h2])

def extract_structural_energy(img_gray):
    """
    Mide la concentración de energía morfológica del barco
    (un barco genera fuertes gradientes perpendiculares que el agua no tiene).
    """
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    mean_mag = np.mean(magnitude)
    std_mag = np.std(magnitude)
    max_mag = np.max(magnitude)
    
    return np.array([mean_mag / 255.0, std_mag / 255.0, max_mag / 255.0])

def extract_all_descriptors(img_rgb):
    """
    Vector final de características geométricas y texturales (100% inmune a color del agua).
    """
    clean_gray = preprocess_image(img_rgb)
    
    f_hog = extract_hog_features(clean_gray)
    f_lbp = extract_lbp_features(clean_gray)
    f_energy = extract_structural_energy(clean_gray)
    
    return np.hstack([f_hog, f_lbp, f_energy])