# Object Detection with Computer Vision 🚀

This project uses the **YOLOv8** model to detect and analyze objects (cars 🚗, humans 👤, and traffic lights 🚦) in a traffic video. It tracks object movement, determines car behavior (moving or stopped), and counts objects within a defined circular region in the video frames. The output includes an annotated video 🎥 and a CSV file 📊 with detection data.

## Files 📂
- `train.py`: The main Python script for object detection, behavior analysis, and video processing using YOLOv8, OpenCV, NumPy, and pandas 🐍.
- `trafficobj.mp4`: The input video file used for object detection and analysis 🎬.
- `.gitignore`: Ensures only `train.py` and `trafficobj.mp4` are tracked, ignoring other files in the project directory 🔒.

## Prerequisites ✅
- Python 3.11 or higher 🐍
- A virtual environment (recommended) 🛠️
- The input video `trafficobj.mp4` (included in the repository) 📹
- The YOLOv8 model weights file `yolov8n.pt` (downloaded automatically by the `ultralytics` library or available from the [Ultralytics YOLOv8 repository](https://github.com/ultralytics/ultralytics)) 🤖

## Setup 🛠️
1. **Clone the Repository** 📥:
   ```bash
   git clone https://github.com/KazamaMasamune/Objectdetectioncv.git
   cd Objectdetectioncv

   Create and Activate a Virtual Environment (optional but recommended) 🌐:
   python -m venv py311_venv
source py311_venv/bin/activate  # On macOS/Linux 🍎
# or
py311_venv\Scripts\activate  # On Windows 🖥️

Install Dependencies 📦:
pip install opencv-python numpy pandas ultralytics
Running the Project ▶️





Ensure train.py and trafficobj.mp4 are in the same directory 📍.



Run the script:

python train.py



Outputs 🎉:





traffic_analyzer_output.mp4: Processed video with bounding boxes, object labels, and counts of objects in the circular region 🎥.



object_detections.csv: CSV file with detection data (frame number, object ID, type, behavior, confidence, coordinates, and whether the object is in the circular region) 📊.



Press q while the video window is active to stop processing early 🛑.

How It Works 🔍





Object Detection 🕵️: Uses YOLOv8 (yolov8n.pt) to detect cars 🚗 (COCO class 2), humans 👤 (class 0), and traffic lights 🚦 (class 9) in each video frame.



Behavior Analysis 📈: For cars, calculates movement distance between frames to classify as "Moving" or "Stopped" (threshold: 5 pixels).



Region Counting 🗺️: Counts objects within a circular region (center: (384, 216), radius: 150 pixels) in the video.



Visualization 🖼️: Draws bounding boxes, labels (with confidence and behavior), and object counts on the video frames.

Notes 📝





The script assumes the input video is 768x432 pixels. Adjust circle_center and circle_radius in train.py if using a different resolution 📏.



If trafficobj.mp4 is replaced, ensure the new video is compatible with OpenCV 🎬.



The yolov8n.pt model is lightweight but may miss some objects. For better accuracy, consider yolov8m.pt or yolov8l.pt (update the model path in train.py) ⚙️.

Future Improvements 🌟





Add support for real-time video streams 📡.



Implement multi-object tracking for consistent IDs across frames 🔗.



Enhance behavior analysis with speed estimation or direction detection 🧭.

License 📜

This project is licensed under the MIT License. See the LICENSE file for details (add a LICENSE file if desired).
