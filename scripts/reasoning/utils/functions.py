import os
os.environ["OMP_PROC_BIND"] = "FALSE"
from scripts.reasoning.models.osnet import ReIDSystem
from ultralytics import YOLO
import torch
from scripts.reasoning.utils.logger import Log, save_to_path
from scipy.spatial.distance import cosine
import time
import cv2
from insightface.app import FaceAnalysis
import soundfile as sf
from kokoro import KPipeline
from IPython.display import display, Audio
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from scripts.reasoning.utils.config import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.simplefilter(action='ignore', category=FutureWarning)

class SCHEMA:
    def __init__(self, shared_data, args):
        self.state = shared_data
        self.args = args
        self.pipeline = KPipeline(lang_code='a')
        self.model = YOLO(self.args.yolo_checkpoint)
        self.model_ppe = YOLO(self.args.ppe_yolo_checkpoint)
        self.reid_sys = ReIDSystem(self.args.yolo_checkpoint, self.args.osnet_checkpoint, self.args.database)
        self.worker_db = self.reid_sys.load_database()
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
    def perception_detect(self, mode="standard"):
        """
        Called by LLM to re-scan the specific frame in memory.
        """
        Log.info(f"[System] Running YOLO Detection (Mode: {mode})...")
        if not hasattr(self.state, 'current_frame') or self.state.current_frame is None:
            return {"error": "No frame in memory"}
        result = self.model.predict(self.state.current_frame, device=device, verbose=False)
        detected = []
        for r in result[0].boxes:
            bbox = list(map(int, r.xyxy[0]))
            detected.append(bbox)

        self.state.metadata["object_bbox"] = detected
        return {"objects_count": len(detected), "bbox": detected}

    def perception_tracking(self, mode="standard"):

        """
        Tracks humans using ByteTrack.
        """
        Log.info(f"[System] Running Human Tracking (Mode: {mode})...")
        if not hasattr(self.state, 'current_frame') or self.state.current_frame is None:
            return {"error": "No frame in memory"}
        timestamp = int(time.time() * 1000)
        results = self.model.track(
            source=self.state.current_frame,
            persist=True,
            tracker="bytetrack.yaml",
            device=device,
            verbose=False,
            classes=[0]  # Important: Only track people
        )
        Log.warning(f"[System] Tracking Done")
        tracked_objects = []
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.int().cpu().tolist()
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            h, w, _ = self.state.current_frame.shape
            for bbox, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                human_crop = self.state.current_frame[y1:y2, x1:x2]
                tracked_objects.append({"id": track_id, "bbox": bbox, "human_crop": human_crop})
        # Update State
        self.state.current_human_crop = tracked_objects
        # self.state.metadata["tracking"] = tracked_objects
        return {"tracked_count": len(tracked_objects)}

    def perception_osnet(self, mode="identification"):
        """
        Called by LLM. Reads state.current_face_crop -> Writes state.current_fingerprint
        """
        Log.info(f"[System] Running OSNet (Mode: {mode})...")
        if not hasattr(self.state, 'current_face_crop') or self.state.current_face_crop is None:
            return {"error": "No face crop found."}
        feats = []
        feats_llm = []
        for item in self.state.current_face_crop:
            face_img = item["face_crop"]  # Đây là ảnh numpy array
            track_id = item["id"]
            if face_img.size == 0: continue
            # results = self.reid_sys.detector(self.state.current_face_crop, classes=0, verbose=False)
            # for r in results[0].boxes:
            #     x1, y1, x2, y2 = map(int, r.xyxy[0])
            #     crop = self.state.current_face_crop[y1:y2, x1:x2]
            #     if crop.size == 0: continue
            feat = self.reid_sys.get_feature(face_img)
            feats.append({"id": track_id, "feat": feat})
            feats_llm.append({"id": track_id, "feat": feat.tolist()[:5]})
        self.state.current_fingerprint = feats
        Log.warning(f"[System] OSNet Feature Extraction Done. Extracted {len(feats)} fingerprints.")
        return {"visual_fingerprint": feats_llm}

    def perception_insightface(self, mode="recognition"):
        """
        Called by LLM. Reads state.current_frame -> Writes state.current_face_crop
        """
        Log.info(f"[System] Running InsightFace (Mode: {mode})...")
        if not hasattr(self.state, 'current_human_crop') or self.state.current_human_crop is None:
            return {"error": "No humans tracked in memory"}
        bboxes = []
        cropped_images = []
        for item in self.state.current_human_crop:
            human_img = item["human_crop"]
            track_id = item["id"]
            human_bbox = item["bbox"]
            if human_img.size == 0: continue

            faces = self.app.get(human_img)
            if len(faces) > 0:
                face = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))[-1]
                bbox = face.bbox.astype(int)
                bboxes.append(bbox)
                x1, y1, x2, y2 = bbox
                # x1, y1 = max(0, x1), max(0, y1)
                # x2, y2 = min(self.state.current_frame.shape[1], x2), min(self.state.current_frame.shape[0], y2)
                # face_crop = self.state.current_frame[y1:y2, x1:x2]
                # cropped_images.append(face_crop)
                h, w, _ = human_img.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                face_crop = human_img[y1:y2, x1:x2]
                if face_crop.size > 0:
                    cropped_images.append({"id": track_id, "face_crop": face_crop})
        self.state.current_face_crop = cropped_images
        self.state.metadata["face_bbox"] = bboxes
        Log.warning(f"[System] InsightFace Detection Done. Detected {len(bboxes)} faces.")
        return {"faces_found": len(bboxes)}

    def perception_voice(self, voice="standard"):
        text = '''
        You do not have permission to this area.
        '''
        generator = self.pipeline(text, voice='af_heart', speed=1)
        return

    def perception_appearance(self, mode="standard"):
        """
        Called by LLM to re-scan the specific frame in memory.
        Calculates Threat Score based on Identity + PPE Violations.
        """
        Log.info(f"[System] Running YOLO PPE - Detection (Mode: {mode})...")
        # CLASS_SCORE_MAP = {}
        # for score, ids in RAW_THREAT_CONFIG.items():
        #     for cls_id in ids:
        #         CLASS_SCORE_MAP[cls_id] = score
        def get_threat_level(total_score):
            if total_score <= THREAT_RANGE["low"][1]: return "low"
            elif total_score <= THREAT_RANGE["medium"][1]: return "medium"
            else: return "high"
        if not hasattr(self.state, 'current_human_crop') or self.state.current_human_crop is None:
            return {"error": "No frame in memory"}

        detected_lists = []
        max_frame_score = 0
        max_frame_level = "low"
        all_detected_classes = []

        for i, item in enumerate(self.state.current_human_crop):
            human_img = item["human_crop"]
            track_id = item["id"]
            if human_img.size == 0: continue

            results = self.model_ppe.predict(human_img, device=device, verbose=False)
            class_names_map = self.model_ppe.names
            detected_objects = []
            found_classes_ids = set()
            # person_ppe_score = 0.0
            drawed_image = human_img.copy()

            for r in results[0].boxes:
                bbox = list(map(int, r.xyxy[0]))
                cls_id = int(r.cls[0])
                cls_name = class_names_map[cls_id]
                conf = float(r.conf[0])
                found_classes_ids.add(cls_id)
                all_detected_classes.append(cls_name)
                # score_increment = CLASS_SCORE_MAP.get(cls_id, 0.0)
                # person_ppe_score += score_increment
                x1, y1, x2, y2 = bbox
                # Color based on score (Red if high penalty, Green if safe/0)
                color = (0, 255, 0)

                cv2.rectangle(drawed_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(drawed_image, f"{cls_name} {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                detected_objects.append({
                    "bbox": bbox,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 2),
                })
                all_detected_classes.append(cls_name)
            missing_items = REQUIRED_PPE_IDS - found_classes_ids
            missing_count = len(missing_items)
            person_ppe_score = missing_count * 0.1
            missing_names = [PPE_CLASS.get(mid, "Unknown") for mid in missing_items]

            identity_info = self.results[i]["identity"] if i < len(self.results) else "Unknown"
            is_intruder = (identity_info == "Unknown")

            identity_penalty = 1.0 if is_intruder else 0.0
            total_person_score = person_ppe_score + identity_penalty
            person_level = get_threat_level(total_person_score)
            if total_person_score > max_frame_score:
                max_frame_score = total_person_score
                max_frame_level = person_level

            detected_lists.append({
                "id": track_id,
                "identity": identity_info,
                "ppe_score": person_ppe_score,
                "missing_items": missing_names,
                "id_score": identity_penalty,
                "total_score": total_person_score,
                "level": person_level,
                "detected": detected_objects
            })

        return {
            "detected_classes": list(set(all_detected_classes)),
            "threat_score": max_frame_score,
            "threat_level": max_frame_level,
            "details": detected_lists
        }

    def knowledge_vector_db(self, collection="worker"):
        """
        Called by LLM. Reads state.current_fingerprint -> Returns Match
        """
        Log.info(f"[System] Querying DB {collection}")

        if not hasattr(self.state, 'current_fingerprint') or self.state.current_fingerprint is None:
            return {"error": "No fingerprint found."}
        self.results = []
        # Iterate over all fingerprints found
        for item in self.state.current_fingerprint:
            track_id = item["id"]
            feat = item["feat"]
            best_sim = -1
            identity = "Unknown"

            for name, saved_feats in self.reid_sys.worker_db.items():
                for s_feat in saved_feats:
                    sim = 1 - cosine(feat, s_feat)
                    if sim > best_sim:
                        best_sim = sim
                        # Add threshold logic here if needed
            if best_sim > self.args.threshold:
                identity = "Worker"
                Log.warning("YOU ARE A WORKER!!!\n"
                      "THANK YOU SO MUCH. PLEASE CONTINUE YOUR WORK.")
            else:
                identity = "Intruder"
                Log.error("YOU ARE NOT A WORKER!!!\n"
                      "WE NEED TO GO THROUGH SOME PROCEDURES TO VERIFY YOUR IDENTITY. PLEASE COOPERATE.")
            self.results.append({"id": track_id, "identity": identity, "best_similarity": round(float(best_sim), 2)})
        return {"matches": self.results}

    def get_tools(self):
        """
        Returns the Function Map and the Corrected JSON Schemas
        """
        functions_map = {
            # "perception_detect": self.perception_detect,
            "perception_track": self.perception_tracking,
            "perception_insightface": self.perception_insightface,
            "perception_osnet": self.perception_osnet,
            "perception_appearance": self.perception_appearance,
            "knowledge_vector_db": self.knowledge_vector_db
        }

        # --- Define schema (JSON) ---
        perception_detection_schema = {
            "name": "perception_detect",
            "description": "Run YOLO object detection to count and locate objects/humans in the current frame.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["standard"]}
                }
            }
        }

        perception_tracking_schema = {
            "name": "perception_track",
            "description": "Run YOLO and Bytetrack to detect objects/humans and assign unique IDs to humans in the current frame. (Aways-on)",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["standard"]}
                }
            }
        }

        perception_insightface_schema = {
            "name": "perception_insightface",
            "description": "Detect and crop faces from the human cropped set.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["recognition"]}
                }
            }
        }

        perception_osnet_schema = {
            "name": "perception_osnet",
            "description": "Extract visual fingerprints from the cropped faces.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["identification"]}
                }
            }
        }

        perception_appearance_schema = {
            "name": "perception_appearance",
            "description": "Scan for Personal Protective Equipment violations and calculate the security threat score/level based on similarity score and appearance. Run after tracking"
                           "to capture full body",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["standard"]}
                }
            }
        }

        knowledge_vector_db_schema = {
            "name": "knowledge_vector_db",
            "description": "Query the vector database to identify the person via computing and comparing similarity score",
            "parameters": {
                "type": "object",
                "properties": {
                    "collection": {"type": "string", "enum": ["workers", "intruders"]}
                },
                "required": ["collection"]
            }
        }
        schema_list = [
            # perception_detection_schema,
            perception_tracking_schema,
            perception_insightface_schema,
            perception_osnet_schema,
            perception_appearance_schema,
            knowledge_vector_db_schema
        ]
        return functions_map, schema_list
