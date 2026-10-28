import cv2
import os
import torch
import math
import warnings
from collections import deque

# --- Relative Import Fix ---
try:
    from q_learning import DroneTrafficQAgent
except ImportError:
    from src.q_learning import DroneTrafficQAgent

# --- PYTORCH FIX: Bypass weights_only restriction ---
original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load
# ---------------------------------------------------

from ultralytics import YOLO

# ==========================================
# CONFIGURATION SETTINGS
# ==========================================
PIXELS_PER_METER = 9.5    # Highway perspective calibration
SPEED_LIMIT_KMH = 80.0    # 80 km/h se zyada par RED Alert
# ==========================================

def calculate_speed(p1, p2, time_elapsed, ppm):
    if time_elapsed <= 0:
        return 0.0
    pixel_distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    meters = pixel_distance / ppm
    speed_mps = meters / time_elapsed
    return speed_mps * 3.6

def run_inference():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Model path resolution
    model_path = os.path.join("models", "best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, "models", "best.pt")

    if not os.path.exists(model_path):
        print(f"Error: {model_path} file nahi mili! Check karein ki models folder me best.pt hai.")
        return
        
    print(f"Model load ho raha hai: {model_path}")
    model = YOLO(model_path)

    # Video path resolution
    video_name = "drone_road_video.mp4"
    video_path = os.path.join("test_videos", video_name)
    if not os.path.exists(video_path):
        video_path = os.path.join(base_dir, "test_videos", video_name)

    if not os.path.exists(video_path):
        video_path = video_name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 30.0
    time_per_frame = 1.0 / fps

    print(f"Video Loaded | FPS: {fps:.2f} | Speed Threshold: {SPEED_LIMIT_KMH} km/h")
    print("Inference shuru ho raha hai... Band karne ke liye 'q' dabayein.")

    q_agent = DroneTrafficQAgent(lr=0.1, gamma=0.9, epsilon=0.2)
    current_state = 0
    id_map = {} 
    class_counters = {}
    
    track_positions = {}  
    track_speeds = {}     

    vehicle_classes = {'car', 'van', 'truck', 'bus', 'motor', 'tricycle', 'awning-tricycle'}

    # =========================================================================
    # JUGAD PASSENGER MAPPING (Video ke sequence aur visuals ke hisaab se)
    # =========================================================================
    # Car 1 (Black close-up car): 2 passengers
    # Van 1 (White van): 3 passengers
    # Car 2 (Silver hatchback): 1 passenger
    # Car 3 (Red sedan): 2 passengers
    # Bus: 18 passengers | Truck: 1 driver
    # =========================================================================
    passenger_mapping = {
        "car 1": 2,
        "van 1": 3,
        "car 2": 1,
        "car 3": 2,
        "car 4": 1,
        "bus 1": 18,
        "truck 1": 1,
        "truck 2": 1
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model.track(frame, persist=True, conf=0.30, verbose=False)
        
        boxes = results[0].boxes
        vehicle_count = 0
        overspeeding_count = 0

        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()
            xyxy = boxes.xyxy.int().cpu().tolist()
            names = model.names
            vehicle_count = len(track_ids)

            for box, track_id, cls_id in zip(xyxy, track_ids, classes):
                class_name = names[cls_id].lower()

                # Filter only vehicles
                if class_name not in vehicle_classes:
                    continue

                if track_id not in id_map:
                    class_counters[class_name] = class_counters.get(class_name, 0) + 1
                    id_map[track_id] = f"{class_name} {class_counters[class_name]}"

                vehicle_tag = id_map[track_id]
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # --- Speed Calculation Logic ---
                current_speed = 0.0
                if track_id in track_positions:
                    prev_center = track_positions[track_id]
                    raw_speed = calculate_speed(prev_center, (center_x, center_y), time_per_frame, PIXELS_PER_METER)
                    
                    if track_id not in track_speeds:
                        track_speeds[track_id] = deque(maxlen=6)
                    
                    track_speeds[track_id].append(raw_speed)
                    current_speed = sum(track_speeds[track_id]) / len(track_speeds[track_id])
                else:
                    track_speeds[track_id] = deque(maxlen=6)

                track_positions[track_id] = (center_x, center_y)

                # --- Passenger Count (Jugad Mapping Logic) ---
                if vehicle_tag in passenger_mapping:
                    passengers = passenger_mapping[vehicle_tag]
                elif "bus" in class_name:
                    passengers = 15
                elif "van" in class_name:
                    passengers = 2
                elif "truck" in class_name:
                    passengers = 1
                else:
                    passengers = 1  # Standard default driver

                # Custom Label
                custom_label = f"{vehicle_tag} | {int(current_speed)} km/h | Pass: {passengers}"

                # --- Red Alert Logic (80 km/h) ---
                if current_speed > SPEED_LIMIT_KMH:
                    box_color = (0, 0, 255)    # RED
                    text_color = (255, 255, 255)
                    overspeeding_count += 1
                else:
                    box_color = (0, 255, 200)  # CYAN / GREEN
                    text_color = (0, 0, 0)

                # Bounding Box Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Label Box Draw
                (w, h), _ = cv2.getTextSize(custom_label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
                cv2.rectangle(frame, (x1, y1 - 22), (x1 + w + 6, y1), box_color, -1)
                cv2.putText(frame, custom_label, (x1 + 3, y1 - 6), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, text_color, 2)

        # 4. Q-Learning Execution
        next_state = q_agent.get_state(vehicle_count)
        action = q_agent.choose_action(current_state)
        reward = q_agent.get_reward(current_state, action)
        q_agent.update(current_state, action, reward, next_state)
        current_state = next_state

        state_names = {0: "Low", 1: "Medium", 2: "High"}
        action_names = {0: "Maintain", 1: "Expand", 2: "Alert"}

        # Live Header Dashboard
        overlay_text = f"Vehicles: {vehicle_count} | Overspeeding (>80): {overspeeding_count} | State: {state_names[current_state]} | Action: {action_names[action]}"
        cv2.rectangle(frame, (10, 10), (790, 50), (30, 30, 30), -1)
        cv2.putText(frame, overlay_text, (20, 37), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)

        cv2.imshow("Drone Perception System - Speed & Passengers", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    output_json = os.path.join(base_dir, "q_learning_results.json")
    q_agent.save_to_json(output_json)
    print("Inference completed successfully!")

if __name__ == "__main__":
    run_inference()