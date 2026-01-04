from google import genai
from google.genai import types
import os
import torch
from scripts.reasoning.utils.logger import Log
import cv2
import json
from ultralytics import YOLO
from scripts.reasoning.utils.prompt import prompt_data
from scripts.reasoning.utils.functions import SCHEMA
# from models.head_pose import HeadPoseEstimator
import argparse
from scripts.reasoning.utils.config import *
import matplotlib.pyplot as plt
import numpy as np
import time

def get_args():
    parser = argparse.ArgumentParser(description="scenario_1")
    parser.add_argument("--yolo_checkpoint", "-yolo", type=str, default="/home/robot/thang_project/unitree_isaac/scripts/reasoning/yolo_checkpoints/yolo11m.pt")
    parser.add_argument("--ppe_yolo_checkpoint", "-ppe", type=str, default="/home/robot/thang_project/unitree_isaac/scripts/reasoning/yolo_checkpoints/best.pt")
    parser.add_argument("--osnet_checkpoint", "-osnet", type=str, default="/home/robot/thang_project/unitree_isaac/scripts/reasoning/osnet_checkpoints/osnet.pth")
    parser.add_argument("--video_path", "-video", type=str, default="/home/robot/thang_project/unitree_isaac/scripts/reasoning/examples/videos/intruder_1.mp4")
    parser.add_argument("--database", "-c", type=str, default="/home/robot/thang_project/unitree_isaac/scripts/reasoning/database/database.pkl")
    parser.add_argument("--threshold", "-th", type=float, default=0.5)
    args = parser.parse_args()
    return args

with open("/home/robot/thang_project/unitree_isaac/.env", "r") as file:
    api_key = file.readlines()
api_key = api_key[0].strip()
client = genai.Client(api_key=api_key)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RobotState:
    def __init__(self):
        self.current_frame = None
        self.current_human_crop = None
        self.current_face_crop = None
        self.current_fingerprint = None
        self.metadata = {}

plt.ion()
figure, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 4))
# Initialize with empty data
img_display_human = axis1.imshow(np.zeros((640, 640, 3), dtype=np.uint8))
img_display_face = axis2.imshow(np.zeros((200, 200, 3), dtype=np.uint8)) # Face crops are usually smaller
axis1.set_title("Human Detection (RGB)")
axis2.set_title("Face Detection (RGB)")
plt.show(block=False)

global current_frame, current_human_crop

class REASONING:
    def __init__(self, args):
        self.args = args
        self.state = RobotState()
        self.schema = SCHEMA(self.state, args)
        self.model = YOLO(args.yolo_checkpoint)
        self.model_name = "models/gemini-3-flash-preview"
        self.functions_map, self.schema_list = self.schema.get_tools()
        self.tools = types.Tool(function_declarations=[*self.schema_list])
        self.config = types.GenerateContentConfig(tools=[self.tools], temperature=0)
        self.is_reasoning = False

    def reasoning(self, frame_bgr, depth):
        brg = frame_bgr.copy()
        self.state.current_frame = brg
        results = self.model.predict(brg, device=device, verbose=False, classes=[0])  # Class 0 = Person
        human_detected = False
        for r in results[0].boxes:
            human_detected = True
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            # current_human_crop = frame_bgr[y1:y2, x1:x2]
            cv2.rectangle(brg, (x1, y1), (x2, y2), (0, 255, 0), 6)
            break  # Process first human only for this demo

        # frame_rgb = cv2.cvtColor(brg, cv2.COLOR_BGR2RGB)
        img_display_human.set_data(brg)

        # We need to pause briefly to let the plot update
        figure.canvas.draw()
        figure.canvas.flush_events()

        if human_detected and depth < 2 and not self.is_reasoning:
            Log.warning("ZONE B IS RESTRICTED AFTER 18:00 SO PLEASE LET ME CHECK YOUR PERMISSION")
            self.resoning_complete = False
            Log.info("\n>>> STARTING REASONING SESSION")
            messages = [
                types.Content(role="user", parts=[types.Part(text=json.dumps(prompt_data))])
            ]

            CURRENT_TURN = 1
            MAX_TURN = 3
            safety_counter = 0

            while CURRENT_TURN <= MAX_TURN and safety_counter < 10:
                safety_counter += 1
                Log.info(f"\n[TURN {CURRENT_TURN} - ATTEMPT {safety_counter}] Thinking...")

                try:
                    response = client.models.generate_content(
                        model=self.model_name,  # gemini-2.0-flash-exp
                        contents=messages,
                        config=self.config
                    )
                except Exception as e:
                    Log.error(f"API Error: {e}")
                    time.sleep(1)
                    break

                # 1. Validate Response
                if not response.candidates or not response.candidates[0].content.parts:
                    Log.warning("Empty response from AI.")
                    continue

                full_content = response.candidates[0].content

                # 2. Append AI Request to History (CRITICAL STEP)
                messages.append(full_content)

                function_responses = []
                turn_success = False  # Flag to check if we pass the current level

                # 3. Process Parts (Thought & Action)
                for part in full_content.parts:
                    if part.text:
                        Log.info(f"[AI THOUGHT]: {part.text}")
                        # If AI gives up or finishes early
                        if "alert" in part.text.lower() or "threat" in part.text.lower():
                            Log.info(">>> THREAT DETECTED. ENDING PROCESS.")
                            CURRENT_TURN = 99  # Force exit
                    if part.function_call:
                        fc = part.function_call
                        fn_name = fc.name
                        fn_args = fc.args

                        Log.info(f"[AI ACTION]: Calling {fn_name} with {fn_args}")
                        if fn_name in self.functions_map:
                            try:
                                tool_result = self.functions_map[fn_name](**fn_args)
                            except Exception as e:
                                tool_result = {"error": str(e)}
                        else:
                            tool_result = {"error": f"Function {fn_name} not found"}
                        if "perception_insightface" in fn_name and self.state.current_face_crop is not None:
                            try:
                                # face_rgb = cv2.cvtColor(self.state.current_face_crop[0]["face_crop"], cv2.COLOR_BGR2RGB)
                                # For dynamic crop sizes, set_data might not work if aspect ratio changes drastically
                                # Re-imshow is safer for varying sizes, or just set_data + autoscale
                                img_display_face.set_data(self.state.current_face_crop[0]["face_crop"])
                                axis2.relim()
                                axis2.autoscale_view()
                                figure.canvas.draw()
                                figure.canvas.flush_events()
                            except Exception as viz_e:
                                Log.warning(f"Viz Error: {viz_e}")

                        Log.info(f"[AI OBSERVATION]: {tool_result}")
                        if CURRENT_TURN == 1:
                            if "visual_fingerprint" in tool_result and tool_result["visual_fingerprint"]:
                                Log.info(">>> SUCCESS: Turn 1 passed (Fingerprint obtained).")
                                turn_success = True
                        elif CURRENT_TURN == 2:
                            if "matches" in tool_result:
                                Log.info(">>> SUCCESS: Turn 2 passed (Similarity score obtained).")
                                turn_success = True
                        elif CURRENT_TURN == 3:
                            if "threat_level" in tool_result:
                                Log.info(">>> SUCCESS: Turn 3 passed (Threat score obtained).")
                                turn_success = True

                        # Prepare response for Gemini
                        function_responses.append(
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=fn_name,
                                    response={"result": tool_result}
                                )
                            )
                        )
                if function_responses:
                    messages.append(types.Content(role="user", parts=function_responses))
                if turn_success:
                    CURRENT_TURN += 1  # Advance to next level
                    safety_counter = 0  # Reset safety counter for the new level
                else:
                    pass
                    # Log.warning(f">>> FAILED condition for Turn {CURRENT_TURN}. Retrying...")
                    # We do NOT increment CURRENT_TURN.
                    # The loop repeats, giving Gemini the error/empty result to try again.
            self.is_reasoning = True
            # self.resoning_complete = True
            Log.info("--- INCIDENT CLOSED ---\n")
            # Log.warning("YOU ARE A WORKER!!!\n"
            #           "THANK YOU SO MUCH. PLEASE CONTINUE YOUR WORK.")
            # Log.error("YOU ARE NOT AN EMPLOYEE!!! WE NEED TO GO THROUGH SOME PROCEDURES TO VERIFY YOUR IDENTITY.\n")
        
# if __name__ == "__main__":
# args = get_args()
# reasoning = REASONING(args)
# Log.info("Reasoning system initialized.")
# cap = cv2.VideoCapture(args.video_path)
# i = 10
# while cap.isOpened():
#     success, frame = cap.read()
#     print(success)
#     if not success: break
#     if not reasoning.is_reasoning:
#         reasoning.reasoning(frame, depth=i)  # Dummy depth for testing
#     else:
#         break
#     i -= 0.5