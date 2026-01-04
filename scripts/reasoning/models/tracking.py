import cv2
from ultralytics import YOLO
import torch


def process_video_with_tracking():
    # 1. Khởi tạo model và đưa lên GPU
    # ByteTrack hoạt động tốt nhất trên GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(checkpoint)
    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        device=device,
        verbose=False  # Tắt log để sạch màn hình terminal server
    )

    # Vẽ các hộp nhận diện và ID lên frame
    # results[0].plot() sẽ tự động vẽ cả Bounding Box và Tracking ID
    annotated_frame = results[0].plot()
