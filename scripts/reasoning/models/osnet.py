import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchreid
from PIL import Image
from torchvision import transforms
from scipy.spatial.distance import cosine
import pickle
import os
import argparse

class ReIDSystem:
    def __init__(self, yolo_checkpoint=None, osnet_weights=None, db_path="database/database.pkl"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.db_path = db_path
        print(f"Loading Detector: {yolo_checkpoint}")
        self.detector = YOLO(yolo_checkpoint)
        print(f"Loading Re-ID Backbone: {osnet_weights}")
        self.reid_model = torchreid.models.build_model(
            name="osnet_ain_x1_0",
            num_classes=1000,
            pretrained=False
        ).to(self.device)
        torchreid.utils.load_pretrained_weights(self.reid_model, osnet_weights)
        self.reid_model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.worker_db = self.load_database()

    def load_database(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                # Sửa pickle.read(f) thành pickle.load(f)
                return pickle.load(f)
        return {}

    def save_database(self):
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.worker_db, f)

    # def save_database(self):
    #     # Tính trung bình các vector cho mỗi Worker trước khi lưu
    #     processed_db = {}
    #     for name, feats in self.worker_db.items():
    #         # Gom tất cả vector lại và tính trung bình theo chiều ngang
    #         mean_feat = np.mean(np.array(feats), axis=0)
    #         processed_db[name] = [mean_feat] # Lưu lại dưới dạng list 1 phần tử để giữ cấu trúc code

    #     with open(self.db_path, 'wb') as f:
    #         pickle.dump(processed_db, f)
    #     print(f"Database optimized and saved to {self.db_path}")

    @torch.no_grad()
    def get_feature(self, cropped_img):
        img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.transform(img).unsqueeze(0).to(self.device)
        feature = self.reid_model(img)
        return feature.cpu().numpy().flatten()
