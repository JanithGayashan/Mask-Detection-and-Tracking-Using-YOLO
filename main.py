import cv2
import time
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance

# Load YOLO model
model = YOLO("best.pt")

# Video source
video_path = 'hospital.mp4'
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Object tracking storage
tracked_objects = {}
next_object_id = 0
IOU_THRESHOLD = 0.1 
ASSOCIATION_DISTANCE_THRESHOLD = 100  

class_labels = {0: "Mask", 1: "No Mask"}  

# IoU Calculation
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# Adjust noise dynamically
def adjust_noise_by_confidence(confidence):
    if confidence > 0.8:
        return 1e-3, 1e-2  
    elif confidence > 0.5:
        return 5e-3, 5e-2   
    else:
        return 1e-2, 1e-1    

# Create Kalman Filter
def create_kalman_filter():
    kalman = cv2.KalmanFilter(6, 2)  
    kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0, 0]], dtype=np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0, 0.5, 0],
                                        [0, 1, 0, 1, 0, 0.5],
                                        [0, 0, 1, 0, 1, 0],
                                        [0, 0, 0, 1, 0, 1],
                                        [0, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 1]], dtype=np.float32)
    kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-3
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
    kalman.errorCovPost = np.eye(6, dtype=np.float32)
    return kalman

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  

    results = model(frame)
    current_detections = []

    # Collect detections
    for result in results:
        detections = result.boxes.data.cpu().numpy()
        print(f"0: {frame.shape[0]}x{frame.shape[1]} {len(detections)} objects, {result.speed['inference']:.1f}ms")

        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            class_id = int(class_id)
            class_label = class_labels.get(class_id, "Unknown")

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]])
            current_detections.append((x1, y1, x2, y2, width, height, confidence, class_id, measurement))

    updated_tracked_objects = {}

    for det in current_detections:
        x1, y1, x2, y2, width, height, confidence, class_id, measurement = det
        best_match_id = None
        highest_iou = 0
        min_distance = float("inf")

        process_noise, measurement_noise = adjust_noise_by_confidence(confidence)

        predicted_x, predicted_y, predicted_width, predicted_height = center_x, center_y, width, height  

        for obj_id, obj_data in tracked_objects.items():
            kalman, prev_bbox, _ = obj_data
            prediction = kalman.predict()
            pred_x, pred_y = prediction[0][0], prediction[1][0]
            pred_w, pred_h = prev_bbox[2] - prev_bbox[0], prev_bbox[3] - prev_bbox[1]
            pred_x1, pred_y1, pred_x2, pred_y2 = pred_x - pred_w / 2, pred_y - pred_h / 2, pred_x + pred_w / 2, pred_y + pred_h / 2

            dist = distance.euclidean((pred_x, pred_y), (center_x, center_y))
            iou = calculate_iou(prev_bbox, (x1, y1, x2, y2))

            if iou > IOU_THRESHOLD or dist < ASSOCIATION_DISTANCE_THRESHOLD:
                if iou > highest_iou or dist < min_distance:
                    best_match_id = obj_id
                    highest_iou = iou
                    min_distance = dist
                    predicted_x, predicted_y = pred_x, pred_y  
                    predicted_width, predicted_height = pred_w, pred_h  

        if best_match_id is None:
            best_match_id = next_object_id
            next_object_id += 1
            kalman = create_kalman_filter()
        else:
            kalman = tracked_objects[best_match_id][0]

        kalman.processNoiseCov *= process_noise
        kalman.measurementNoiseCov *= measurement_noise
        kalman.correct(measurement)

        updated_tracked_objects[best_match_id] = (kalman, (x1, y1, x2, y2), time.time())

        velocity_x = kalman.statePost[2][0]
        velocity_y = kalman.statePost[3][0]
        velocity = np.sqrt(velocity_x**2 + velocity_y**2)

        print(f"Class: {class_label} | Predicted Position: ({predicted_x:.2f}, {predicted_y:.2f}) | Velocity: {velocity:.2f} px/s | Confidence: {confidence:.2f}")

        color = (0, 255, 0) if class_id == 0 else (0, 0, 255)  
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"ID: {best_match_id} {class_label} C:{confidence:.2f}", 
                    (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # **Optimized Kalman Filter Predicted Bounding Box (Light Yellow)**
        pred_x1, pred_y1 = int(predicted_x - predicted_width / 2), int(predicted_y - predicted_height / 2)
        pred_x2, pred_y2 = int(predicted_x + predicted_width / 2), int(predicted_y + predicted_height / 2)
        cv2.rectangle(frame, (pred_x1, pred_y1), (pred_x2, pred_y2), (0, 255, 255), 2)

    tracked_objects = updated_tracked_objects

    cv2.imshow("Mask Detection Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


