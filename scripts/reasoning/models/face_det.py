import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# 1. Initialize
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 2. Load your image
img = cv2.imread('test_image.jpg')

# 3. Perform analysis
faces = app.get(img)

# 4. Process each detected face
for i, face in enumerate(faces):
    # InsightFace returns bbox as floats: [x1, y1, x2, y2]
    # We convert them to integers for OpenCV cropping
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox

    # Ensure coordinates are within image boundaries (safety check)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

    # Extract/Crop the face from the original image
    face_crop = img[y1:y2, x1:x2]

    # Save or show the cropped face
    cv2.imwrite(f'face_crop_{i}.jpg', face_crop)

    print(f"Face {i} extracted: Box coords ({x1}, {y1}) to ({x2}, {y2})")

print(f"Successfully processed {len(faces)} faces.")