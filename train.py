import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict

def detect_behavior(prev_pos, curr_pos, threshold=5):
    """Detect car behavior based on movement."""
    if prev_pos is None:
        return "Stopped"
    dist = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
    return "Moving" if dist > threshold else "Stopped"

def is_point_in_circle(point, center, radius):
    """Check if a point (x, y) is inside a circle with given center and radius."""
    distance = np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
    return distance <= radius

def main():
    # Load video
    video_path = "trafficobj.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Load YOLO model
    model = YOLO("yolov8n.pt")

    # Prepare output video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    out = cv2.VideoWriter("traffic_analyzer_output.mp4", 
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Define the circular area (polynomial tracker) - used only for counting
    circle_center = (384, 216)  # (x, y) = (width/2, height/2)
    circle_radius = 150  # Adjust based on your video's area size

    # Track objects
    prev_positions = {}  # Object ID -> previous position
    object_detections = []  # To store detection data for CSV
    object_id = 0
    counts = {"Car": 0, "Traffic Light": 0, "Human": 0}  # Count objects in circle

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame. End of video or video file issue.")
            break

        frame_count += 1
        print(f"Processing frame {frame_count}, shape: {frame.shape}")

        # Detect objects (class 0 = person, class 2 = car, class 9 = traffic light in COCO)
        results = model(frame, classes=[0, 2, 9])
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        # Reset counts for this frame
        counts = {"Car": 0, "Traffic Light": 0, "Human": 0}
        current_objects = {}

        # Process detections
        for box, conf, cls in zip(boxes, confs, classes):
            if conf < 0.5:
                continue
            x1, y1, x2, y2 = map(int, box)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            object_id += 1
            current_objects[object_id] = center

            # Determine object type
            if cls == 0:
                obj_type = "Human"
                box_color = (255, 0, 0)  # Blue for humans
            elif cls == 2:
                obj_type = "Car"
                box_color = (0, 255, 0)  # Green for cars
            elif cls == 9:
                obj_type = "Traffic Light"
                box_color = (0, 255, 0)  # Green for traffic lights
            else:
                continue  # Skip other classes

            # Check if the object is in the circle
            in_circle = is_point_in_circle(center, circle_center, circle_radius)
            if in_circle:
                counts[obj_type] += 1

            # Detect behavior (for cars only)
            behavior = detect_behavior(prev_positions.get(object_id), center) if obj_type == "Car" else "N/A"

            # Store detection data
            object_detections.append({
                "Frame": frame_count,
                "Object_ID": object_id,
                "Type": obj_type,
                "Behavior": behavior,
                "Confidence": conf,
                "Center_X": center[0],
                "Center_Y": center[1],
                "In_Circle": in_circle
            })

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            # Add label with object type, confidence, and behavior (for cars)
            label = f"{obj_type} {object_id} (Conf: {conf:.2f})"
            if obj_type == "Car":
                label += f", {behavior}"
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)  # Yellow text
            print(f"Detected {obj_type} {object_id} at ({center[0]}, {center[1]}) with confidence {conf:.2f}, In Circle: {in_circle}")

        # Update previous positions
        prev_positions = current_objects

        # Display the counts of objects in the circle on the right side
        y_offset = 50
        for obj_type, count in counts.items():
            count_text = f"{obj_type}s in Circle: {count}"
            cv2.putText(frame, count_text, (width - 200, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # Cyan text
            y_offset += 30

        # Write frame
        out.write(frame)
        cv2.imshow("Traffic Analyzer", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Save detection data to CSV
    if object_detections:
        df = pd.DataFrame(object_detections)
        df.to_csv("object_detections.csv", index=False)
        print("Object detections saved as object_detections.csv")

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Output saved as traffic_analyzer_output.mp4")

if __name__ == "__main__":
    main()