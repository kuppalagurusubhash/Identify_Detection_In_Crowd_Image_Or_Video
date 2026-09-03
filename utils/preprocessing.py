import cv2
import numpy as np

def apply_clahe(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to equalize lighting across low-light or unevenly lit crowd images.
    """
    if image is None:
        return None
    
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced

def preprocess_face_crop(face_img, target_size=(112, 112)):
    """
    Resize and normalize face crop for deep neural network embedding extractor.
    """
    if face_img is None or face_img.size == 0:
        return None

    # Apply light enhancement if image is dark
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < 80:
        face_img = apply_clahe(face_img)

    # Resize to target size (112x112 standard ArcFace/InsightFace dimension)
    resized = cv2.resize(face_img, target_size, interpolation=cv2.INTER_AREA)
    return resized
