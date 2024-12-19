# Import necessary packages
import face_recognition
import imutils
import pickle
import cv2
import numpy as np
import time
import statistics
import tflite_runtime.interpreter as tflite
import matplotlib.pyplot as plt

# Initialize variables
currentname = "Unknown"
encodings_path = "encodings.pickle"

# Load face recognition encodings
print("[INFO] Loading face encodings and face detection model...")
data = pickle.loads(open(encodings_path, "rb").read())

# Load object detection model
print("[INFO] Initializing object detection model...")
model_path = 'models/model.tflite'
labels_path = 'models/labels.txt'
threshold = 0.5

# Load model and labels
interpreter = tflite.Interpreter(model_path=model_path)
with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
interpreter.allocate_tensors()

# Get model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height, width = input_details[0]['shape'][1], input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean, input_std = 127.5, 127.5

# Start video capture
print("[INFO] Starting live video capture...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Unable to access the camera.")
    exit()

# Video recording settings
frame_rate = 1
duration = 35
start_time = time.time()
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('detection_output.mp4', fourcc, frame_rate, (frame_width, frame_height))

# Data structures for storing detection information
time_stamps = []
person_detections = {}
object_detections = {}

while time.time() - start_time < duration:
    # Capture a frame from the video
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Frame capture failed. Stopping.")
        break

    current_time = time.time() - start_time
    time_stamps.append(current_time)
    frame = imutils.resize(frame, width=500)

    # Face Detection and Recognition
    face_boxes = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_boxes)
    detected_faces = []

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"
        if True in matches:
            matched_idxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matched_idxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # Calculate confidence
            face_distances = face_recognition.face_distance(data["encodings"], encoding)
            fd_mean = statistics.mean([fd if fd < 0.5 else 1 for fd in face_distances])
            confidence = (1 - fd_mean) * 100
            name = max(counts, key=counts.get)

            # Store detection data
            if name not in person_detections:
                person_detections[name] = {'timestamps': [], 'confidences': []}
            person_detections[name]['timestamps'].append(current_time)
            person_detections[name]['confidences'].append(confidence)

        detected_faces.append(name)

    # Object Detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(resized_frame, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    obj_boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    obj_classes = interpreter.get_tensor(output_details[1]['index'])[0]
    obj_scores = interpreter.get_tensor(output_details[2]['index'])[0]

    for i in range(len(obj_scores)):
        if obj_scores[i] > threshold:
            label = labels[int(obj_classes[i])]
            confidence = obj_scores[i] * 100

            # Scale bounding box
            ymin, xmin, ymax, xmax = (obj_boxes[i] * [frame_height, frame_width, frame_height, frame_width]).astype(int)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            text = f"{label}: {confidence:.2f}%"
            cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Store detection data
            if label not in object_detections:
                object_detections[label] = {'timestamps': [], 'confidences': []}
            object_detections[label]['timestamps'].append(current_time)
            object_detections[label]['confidences'].append(confidence)

    # Annotate faces on the frame
    for ((top, right, bottom, left), name) in zip(face_boxes, detected_faces):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, f"{name}: {confidence:.2f}%", (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Write frame to video and display
    out.write(cv2.resize(frame, (frame_width, frame_height)))
    cv2.imshow("Real-Time Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Generate Confidence Level Plots
plt.figure(figsize=(12, 6))

# Plot person detection data
for person, data in person_detections.items():
    plt.plot(data['timestamps'], data['confidences'], label=f"Person: {person}")

# Plot object detection data
for obj, data in object_detections.items():
    plt.plot(data['timestamps'], data['confidences'], label=f"Object: {obj}")

# Customize plot
plt.xlabel("Time (seconds)")
plt.ylabel("Confidence (%)")
plt.title("Detection Confidence over Time")
plt.legend(loc='best')
plt.savefig("detection_confidence_plot.png")
plt.show()