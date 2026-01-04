from google import genai
from google.genai import types
import os
import torch
from utils.logger import Log
import cv2
import json
from ultralytics import YOLO
from utils.prompt import prompt_data
from utils.functions import SCHEMA
#from models.head_pose import HeadPoseEstimator
import argparse
from utils.config import *

def get_args():
    parser = argparse.ArgumentParser(description="scenario_1")
    parser.add_argument("--yolo_checkpoint", "-yolo", type=str, default="yolo_checkpoints/yolo11m.pt")
    parser.add_argument("--ppe_yolo_checkpoint", "-ppe", type=str, default="yolo_checkpoints/best.pt")
    parser.add_argument("--osnet_checkpoint", "-osnet", type=str, default="osnet_checkpoints/osnet.pth")
    parser.add_argument("--video_path", "-video", type=str, default="examples/videos/intruder_1.mp4")
    parser.add_argument("--database", "-c", type=str, default="database/database.pkl")
    parser.add_argument("--threshold", "-th", type=float, default=0.85)
    args = parser.parse_args()
    return args

#with open(os.path.join(find_project_root("ignore"), "api.txt"), "r") as file:
with open(os.path.join("/home/robot/ignore", "api.txt"), "r") as file:
    api_key = file.readlines()
api_key = api_key[0]
client = genai.Client(api_key=api_key)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RobotState:
    def __init__(self):
        self.current_frame = None
        self.current_human_crop = None
        self.current_face_crop = None
        self.current_fingerprint = None
        self.metadata = {}

def reasoning(args, rgb, depth):
    global current_frame, current_human_crop
    state = RobotState()
    schema = SCHEMA(state, args)
    model = YOLO(args.yolo_checkpoint)
    #estimator = HeadPoseEstimator(threshold_yaw=60)
    functions_map, schema_list = schema.get_tools()
    # Configure GenAI
    tools = types.Tool(function_declarations=[*schema_list])
    config = types.GenerateContentConfig(tools=[tools], temperature=0)
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        Log.info("Error: Cannot open video source.")
        return

    # State flags
    is_reasoning = False

    while cap.isOpened():
        success, frame = cap.read()
        #print(success)
        if not success: break
        state.current_frame = frame

        results = model.predict(frame, device=device, verbose=False, classes=[0])  # Class 0 = Person
        #angle, valid_trigger, _ = estimator.process(frame)
        #print(angle)
        human_detected = False
        for r in results[0].boxes:
            human_detected = True
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            current_human_crop = frame[y1:y2, x1:x2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            break  # Process first human only for this demo
        #add distance condition
        if human_detected and not is_reasoning:
            is_reasoning = True
            Log.info("\n>>> STARTING REASONING SESSION")
            messages = [
                types.Content(role="user", parts=[types.Part(text=json.dumps(prompt_data))])
            ]

            CURRENT_TURN = 1
            MAX_TURN = 3
            safety_counter = 0

            while CURRENT_TURN <= MAX_TURN and safety_counter < 10:
                safety_counter += 1
                Log.info(f"\n[TURN {CURRENT_TURN} - ATTEMPT {safety_counter}] Requesting Gemini...")

                try:
                    response = client.models.generate_content(
                        model="models/gemini-3-flash-preview",  # gemini-2.0-flash-exp
                        contents=messages,
                        config=config
                    )
                except Exception as e:
                    Log.error(f"API Error: {e}")
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
                        if fn_name in functions_map:
                            try:
                                tool_result = functions_map[fn_name](**fn_args)
                            except Exception as e:
                                tool_result = {"error": str(e)}
                        else:
                            tool_result = {"error": f"Function {fn_name} not found"}

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
                    Log.warning(f">>> FAILED condition for Turn {CURRENT_TURN}. Retrying...")
                    # We do NOT increment CURRENT_TURN.
                    # The loop repeats, giving Gemini the error/empty result to try again.
            is_reasoning = False
            Log.info("--- INCIDENT CLOSED ---\n")
            #if CURRENT_TURN == 3:
            break
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    cap.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    args = get_args()
    main(args)
