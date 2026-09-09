# Road-issues-detection
Simple YOLO based road issues detection system

## Files
1. Road Yolo.pt - The main custom trained YOLOv8n (nano) model to detect potholes, accidents and violence
2. YOLOv11n-pose.pt - Direct inference model for improved and dedicated violece detection. Used more robust logic to detect violence
3. detect.py - simple python script to make use of the Custom trained Yolo model (Road yolo.pt) on real time web cam feed
4. violence.py - final script to use both the trained model and the pose model for integrated detection of all kinds of issues.
