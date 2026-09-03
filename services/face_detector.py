import cv2
import numpy as np

class FaceDetector:
    def __init__(self, min_confidence=0.5):
        self.min_confidence = min_confidence
        
        # Load OpenCV default Haar cascade classifiers
        self.cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        self.cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

    def detect_faces(self, image):
        """
        Detects faces in an image.
        Returns:
            list of dicts: [
                {
                    'box': (x, y, w, h),
                    'confidence': 0.95,
                    'crop': np.ndarray (BGR face crop)
                }, ...
            ]
        """
        if image is None or image.size == 0:
            return []

        h_img, w_img = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for robust multi-illumination detection
        gray_eq = cv2.equalizeHist(gray)

        boxes = []

        # 1. Multi-scale frontal detection
        faces1 = self.cascade_frontal.detectMultiScale(
            gray_eq,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in faces1:
            boxes.append([x, y, w, h, 0.90])

        # 2. Multi-scale frontal detection (alt)
        faces2 = self.cascade_alt.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=2,
            minSize=(25, 25)
        )
        for (x, y, w, h) in faces2:
            boxes.append([x, y, w, h, 0.85])

        # 3. Profile face detection
        faces_prof = self.cascade_profile.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(25, 25)
        )
        for (x, y, w, h) in faces_prof:
            boxes.append([x, y, w, h, 0.80])

        # 4. Fallback Contour / Color-based Face Blob Detection if Cascades find no faces
        if not boxes:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Broad skin color range in HSV space
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([30, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_skin, upper_skin)

            # Morphological Operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area > 1000: # Filter small noise
                    x, y, w, h = cv2.boundingRect(c)
                    aspect_ratio = float(h) / w
                    if 0.7 <= aspect_ratio <= 1.8 and w < w_img * 0.9 and h < h_img * 0.9:
                        boxes.append([x, y, w, h, 0.75])

        if not boxes:
            return []

        # Apply Non-Maximum Suppression (NMS)
        boxes_nms = self._non_max_suppression(np.array(boxes), overlapThresh=0.35)

        results = []
        padding_ratio = 0.05

        for (x, y, w, h, conf) in boxes_nms:
            x, y, w, h = int(x), int(y), int(w), int(h)

            pad_w = int(w * padding_ratio)
            pad_h = int(h * padding_ratio)

            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w_img, x + w + pad_w)
            y2 = min(h_img, y + h + pad_h)

            face_crop = image[y1:y2, x1:x2]

            if face_crop.size > 0:
                results.append({
                    'box': (x, y, w, h),
                    'padded_box': (x1, y1, x2 - x1, y2 - y1),
                    'confidence': float(conf),
                    'crop': face_crop
                })

        return results

    def _non_max_suppression(self, boxes, overlapThresh=0.3):
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        scores = boxes[:, 4]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(scores)[::-1]

        pick = []
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[0]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[1:]]

            idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlapThresh)[0] + 1)))

        return boxes[pick]
