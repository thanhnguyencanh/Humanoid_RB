import cv2
import mediapipe as mp
import numpy as np
import time
from scripts.reasoning.utils.logger import save_to_path

class HeadPoseEstimator:
    def __init__(self, threshold_yaw=40):
        self.threshold_yaw = threshold_yaw
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1,
            refine_landmarks=True
        )
        self.model_points = np.array([
            (0.0, 0.0, 0.0),  # 1. Nose tip (Đầu mũi)
            (0.0, -330.0, -65.0),  # 2. Chin (Cằm)
            (-225.0, 170.0, -135.0),  # 3. Left eye left corner (Mắt trái)
            (225.0, 170.0, -135.0),  # 4. Right eye right corner (Mắt phải)
            (-150.0, -150.0, -125.0),  # 5. Left Mouth corner (Mépm trái)
            (150.0, -150.0, -125.0)  # 6. Right mouth corner (Mép phải)
        ])

    def process(self, image):
        """
        Input: Ảnh OpenCV (BGR)
        Output: (yaw_angle, is_valid_for_recognition, image_visualized)
        """
        h, w, c = image.shape
        drawed_image = image.copy()
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            print("No face detected in this frame.")
            return None, False, image

        face_landmarks = results.multi_face_landmarks[0]
        # MediaPipe Indices: Nose:1, Chin:152, EyeL:33, EyeR:263, MouthL:61, MouthR:291
        idx_list = [1, 152, 33, 263, 61, 291]
        image_points = []

        for idx in idx_list:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            image_points.append([x, y])

        image_points = np.array(image_points, dtype="double")

        # ---PnP---
        # focal_length_x = 4.4260431700991404e+02
        # focal_length_y = 4.6152026668010120e+02
        # center_x = 6.3584834258781643e+02
        # center_y = 3.8708963117313408e+02

        focal_length_x = w
        focal_length_y = w
        center_x = w / 2
        center_y = h / 2
        camera_matrix = np.array(
            [[focal_length_x, 0, center_x],
             [0, focal_length_y,center_y],
             [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4, 1))  # Giả sử không méo hình

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        rotation_mat, _ = cv2.Rodrigues(rotation_vector)
        pose_mat = cv2.hconcat((rotation_mat, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        pitch = euler_angles[0, 0]
        yaw = euler_angles[1, 0]
        roll = euler_angles[2, 0]
        is_valid = abs(yaw) <= self.threshold_yaw
        color = (0, 255, 0) if is_valid else (0, 0, 255)  # Green if valid, Red if invalid
        nose_end_point2D, _ = cv2.projectPoints(
            np.array([(0.0, 0.0, 500.0)]),
            rotation_vector,
            translation_vector,
            camera_matrix,
            dist_coeffs
        )

        p1 = (int(image_points[0][0]), int(image_points[0][1]))  # Nose Tip
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))  # Projected End

        cv2.line(drawed_image, p1, p2, color, 3)
        for p in image_points:
            cv2.circle(drawed_image, (int(p[0]), int(p[1])), 3, (255, 0, 0), -1)

        status_text = "PASS" if is_valid else "FAIL"
        cv2.putText(drawed_image, f"Yaw: {int(yaw)} deg | {status_text}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        timestamp = int(time.time() * 1000)
        file_path = f"results/head_pose/pose_{timestamp}.jpg"
        save_to_path(file_path, drawed_image)

        return yaw, is_valid, image