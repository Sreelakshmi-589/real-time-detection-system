Real-Time Face and Object Detection System
This project implements a real-time face and object detection system using a Raspberry Pi 4. It leverages OpenCV, face_recognition, and TensorFlow Lite for detecting and recognizing faces and objects.

Features
1. Real-time face recognition with pre-trained encodings.
2. Object detection using a TensorFlow Lite model.
3. Supports varying lighting conditions for robust performance.
4. Outputs include annotated video and confidence plots.

Installation
Hardware Requirements:
1. Raspberry Pi 4 with Raspberry Pi OS
2. Raspberry Pi Camera
3. Minimum 16 GB microSD card

Software Prerequisites:
1. Install OpenCV
2. Install TensorFlow Lite dependencies
3. Install the face-recognition package

Setting Up Virtual Environment
Create and activate a virtual environment for clean dependency management:
python3 -m venv tensorflow
source tensorflow/bin/activate

Directory Structure
real-time-detection-system/
│
├── Dataset/                # Contains labeled folders of training images
├── models/                 # Pre-trained TensorFlow Lite models
├── scripts/
│   ├── headshots.py        # Script to capture images for training
│   ├── train_model.py      # Script to train the face recognition model
│   ├── final_project.py    # Main script for real-time detection
├── install-prerequisites.sh # Shell script to install dependencies
├── encodings.pickle        # Trained model for face recognition
├── detection_output.mp4    # Annotated output video 
├── detection_confidence_plot.png # Confidence plot
└── README.md               # Documentation


Usage
1. Capture Training Images: Run headshots.py to capture images of individuals for training.
2. Train the Model: Execute train_model.py to generate the face encodings.
3. Run Detection: Use final_project.py to perform real-time detection.

Outputs
1. Annotated Video: Shows bounding boxes and confidence scores for detections (detection_output.mp4).
2. Confidence Plot: Visualizes detection performance under varying lighting conditions (detection_confidence_plot.png).
