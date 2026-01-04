import cv2
import torch
import numpy as np
import random
from ultralytics import YOLO
import torchreid
from PIL import Image
from torchvision import transforms
from scipy.spatial.distance import cosine
import pickle
import os
import argparse
import glob
from insightface.app import FaceAnalysis

class ReIDSystem:
    def __init__(self, yolo_checkpoint=None, osnet_weights=None, db_path="database/database.pkl"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.db_path = db_path
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        print(f"--- Loading Models on {self.device} ---")
        self.detector = YOLO(yolo_checkpoint)
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

        # Initialize InsightFace
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

    def load_database(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    return pickle.load(f)
            except EOFError:
                return {}
        return {}

    # def save_database(self):
    #     print(f"Saving database with {len(self.worker_db)} identities to {self.db_path}...")
    #     with open(self.db_path, 'wb') as f:
    #         pickle.dump(self.worker_db, f)

    def save_database(self):
        processed_db = {}
        for name, feats in self.worker_db.items():
            mean_feat = np.mean(np.array(feats), axis=0)
            # norm = np.linalg.norm(mean_feat)
            # mean_feat = mean_feat / norm

            processed_db[name] = [mean_feat]

        with open(self.db_path, 'wb') as f:
            pickle.dump(processed_db, f)
        print(f"Database optimized and saved to {self.db_path}")

    @torch.no_grad()
    def get_feature(self, cropped_img):
        img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.transform(img).unsqueeze(0).to(self.device)
        feature = self.reid_model(img)
        return feature.cpu().numpy().flatten()

def extract_and_enroll(reid_sys, input_source, worker_name):
    """
    Reads Video OR Folder -> Detects Face -> Saves Crop to Folder -> Enrolls in DB.
    NO VIDEO OUTPUT.
    """
    random.seed(42)
    save_dir = os.path.join("results/enrolled_crops", worker_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Processing: {worker_name} ---")
    print(f"Source: {input_source}")
    print(f"Output: {save_dir}")
    if os.path.isdir(input_source):
        # Folder of images
        img_paths = glob.glob(os.path.join(input_source, "*.*"))
        # Generator that yields (frame, frame_id)
        frames_iter = [(cv2.imread(p), i) for i, p in enumerate(img_paths)]
    else:
        # Video file
        cap = cv2.VideoCapture(input_source)
        if not cap.isOpened():
            print(f"Error: Cannot open video {input_source}")
            return
        # Generator for video frames
        def video_gen():
            fid = 0
            while True:
                ret, f = cap.read()
                if not ret: break
                yield f, fid
                fid += 1

        frames_iter = video_gen()
    if worker_name not in reid_sys.worker_db:
        reid_sys.worker_db[worker_name] = []

    count_saved = 0
    max_samples = 50
    # frames = random.shuffle(frames_iter)

    for frame, frame_id in frames_iter:
        if frame is None: continue
        if count_saved >= max_samples: break

        # Skip frames for video to add variety (process every 5th frame)
        if not os.path.isdir(input_source) and frame_id % 6 != 0:
            continue

        results = reid_sys.detector(frame, classes=0, verbose=False)

        for r in results[0].boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            if (x2 - x1) < 30 or (y2 - y1) < 60: continue

            person_crop = frame[y1:y2, x1:x2]
            faces = reid_sys.face_app.get(person_crop)
            if not faces: continue

            # Get largest face
            face = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))[-1]
            fx1, fy1, fx2, fy2 = face.bbox.astype(int)
            fx1, fy1 = max(0, fx1), max(0, fy1)
            fx2, fy2 = min(person_crop.shape[1], fx2), min(person_crop.shape[0], fy2)

            if fx2 <= fx1 or fy2 <= fy1: continue

            face_crop = person_crop[fy1:fy2, fx1:fx2]

            filename = f"{worker_name}_{count_saved:04d}.jpg"
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, face_crop)

            feat = reid_sys.get_feature(face_crop)
            # feat = face.embedding
            reid_sys.worker_db[worker_name].append(feat)

            count_saved += 1
            print(f"Saved: {filename}", end='\r')
            break  # Only take one face per frame (the worker)

    print(f"\nDone. Saved {count_saved} images to {save_dir}")


def run_recognition_logging(reid_sys, video_path):
    """
    Runs recognition and saves crops of detected people to folders.
    NO VIDEO OUTPUT.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return

    print(f"--- Running Recognition on {video_path} ---")
    print("    Results will be saved to 'recognition_logs/'")

    frame_id = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        if frame_id % 5 != 0:  # Optimization
            frame_id += 1
            continue

        results = reid_sys.detector(frame, classes=0, verbose=False)
        for r in results[0].boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0: continue

            faces = reid_sys.face_app.get(person_crop)
            if faces:
                face = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))[-1]
                fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                fx1, fy1 = max(0, fx1), max(0, fy1)
                fx2, fy2 = min(person_crop.shape[1], fx2), min(person_crop.shape[0], fy2)

                face_crop = person_crop[fy1:fy2, fx1:fx2]
                if face_crop.size == 0: continue

                # Recognize
                feat = reid_sys.get_feature(face_crop)
                identity = "Unknown"
                best_sim = 0.0

                for name, saved_feats in reid_sys.worker_db.items():
                    for s_feat in saved_feats:
                        sim = 1 - cosine(feat, s_feat)
                        if sim > best_sim:
                            best_sim = sim
                            if sim > 0.60: identity = name

                # Save Log
                log_dir = os.path.join("recognition_logs", identity)
                os.makedirs(log_dir, exist_ok=True)
                cv2.imwrite(f"{log_dir}/frame_{frame_id}_{best_sim:.2f}.jpg", face_crop)

        frame_id += 1
        if frame_id % 30 == 0: print(f"Processed {frame_id} frames...", end='\r')

    cap.release()
    print("\nRecognition Completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=1, type=int, help="1: Enroll, 2: Recognize", required=True)
    parser.add_argument("--checkpoint_osnet",
                        default="osnet_checkpoints/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth",
                        type=str)
    parser.add_argument("--checkpoint_yolo", default="yolo_checkpoints/yolo11m.pt", type=str)
    parser.add_argument("--save_path", default="database/database.pkl", type=str)
    parser.add_argument("--test_video", default="docs/example_video/test_osnet.mp4", type=str)

    args = parser.parse_args()

    reid_sys = ReIDSystem(args.checkpoint_yolo, args.checkpoint_osnet, args.save_path)

    if args.mode == 1:
        # ==========================================
        # DEFINE INPUTS HERE (Video Paths OR Folders)
        # ==========================================
        targets = [
            ("human_data/worker_1/Capture_frames", "Worker_A"),
            ("human_data/worker_2/Capture_frames", "Worker_B"),
            ("human_data/worker_3/Capture_frames", "Worker_C"),
            # You can also use folders:
            # ("human_data/worker_C/raw_images", "Worker_C"),
        ]

        for src, name in targets:
            extract_and_enroll(reid_sys, src, name)

        reid_sys.save_database()

    else:
        run_recognition_logging(reid_sys, args.test_video)