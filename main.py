# import cv2
# import time
# import numpy as np
# from ultralytics import YOLO
# from scipy.spatial import distance

# # Load YOLO model
# model = YOLO("best.pt")

# # Video source
# video_path = 'hospital.mp4'
# cap = cv2.VideoCapture(video_path)
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# fps = cap.get(cv2.CAP_PROP_FPS)

# # Object tracking storage
# tracked_objects = {}
# next_object_id = 0
# IOU_THRESHOLD = 0.1 
# ASSOCIATION_DISTANCE_THRESHOLD = 100  

# class_labels = {0: "Mask", 1: "No Mask"}  

# # IoU Calculation
# def calculate_iou(boxA, boxB):
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])
#     interArea = max(0, xB - xA) * max(0, yB - yA)
#     if interArea == 0:
#         return 0.0
#     boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
#     return interArea / float(boxAArea + boxBArea - interArea)

# # Adjust noise dynamically
# def adjust_noise_by_confidence(confidence):
#     if confidence > 0.8:
#         return 1e-3, 1e-2  
#     elif confidence > 0.5:
#         return 5e-3, 5e-2   
#     else:
#         return 1e-2, 1e-1    

# # Create Kalman Filter
# def create_kalman_filter():
#     kalman = cv2.KalmanFilter(6, 2)  
#     kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
#                                          [0, 1, 0, 0, 0, 0]], dtype=np.float32)
#     kalman.transitionMatrix = np.array([[1, 0, 1, 0, 0.5, 0],
#                                         [0, 1, 0, 1, 0, 0.5],
#                                         [0, 0, 1, 0, 1, 0],
#                                         [0, 0, 0, 1, 0, 1],
#                                         [0, 0, 0, 0, 1, 0],
#                                         [0, 0, 0, 0, 0, 1]], dtype=np.float32)
#     kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-3
#     kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
#     kalman.errorCovPost = np.eye(6, dtype=np.float32)
#     return kalman

# start_time = time.time()

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break  

#     results = model(frame)
#     current_detections = []

#     # Collect detections
#     for result in results:
#         detections = result.boxes.data.cpu().numpy()
#         print(f"0: {frame.shape[0]}x{frame.shape[1]} {len(detections)} objects, {result.speed['inference']:.1f}ms")

#         for detection in detections:
#             x1, y1, x2, y2, confidence, class_id = detection
#             class_id = int(class_id)
#             class_label = class_labels.get(class_id, "Unknown")

#             center_x = (x1 + x2) / 2
#             center_y = (y1 + y2) / 2
#             width = x2 - x1
#             height = y2 - y1
#             measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]])
#             current_detections.append((x1, y1, x2, y2, width, height, confidence, class_id, measurement))

#     updated_tracked_objects = {}

#     for det in current_detections:
#         x1, y1, x2, y2, width, height, confidence, class_id, measurement = det
#         best_match_id = None
#         highest_iou = 0
#         min_distance = float("inf")

#         process_noise, measurement_noise = adjust_noise_by_confidence(confidence)

#         predicted_x, predicted_y, predicted_width, predicted_height = center_x, center_y, width, height  

#         for obj_id, obj_data in tracked_objects.items():
#             kalman, prev_bbox, _ = obj_data
#             prediction = kalman.predict()
#             pred_x, pred_y = prediction[0][0], prediction[1][0]
#             pred_w, pred_h = prev_bbox[2] - prev_bbox[0], prev_bbox[3] - prev_bbox[1]
#             pred_x1, pred_y1, pred_x2, pred_y2 = pred_x - pred_w / 2, pred_y - pred_h / 2, pred_x + pred_w / 2, pred_y + pred_h / 2

#             dist = distance.euclidean((pred_x, pred_y), (center_x, center_y))
#             iou = calculate_iou(prev_bbox, (x1, y1, x2, y2))

#             if iou > IOU_THRESHOLD or dist < ASSOCIATION_DISTANCE_THRESHOLD:
#                 if iou > highest_iou or dist < min_distance:
#                     best_match_id = obj_id
#                     highest_iou = iou
#                     min_distance = dist
#                     predicted_x, predicted_y = pred_x, pred_y  
#                     predicted_width, predicted_height = pred_w, pred_h  

#         if best_match_id is None:
#             best_match_id = next_object_id
#             next_object_id += 1
#             kalman = create_kalman_filter()
#         else:
#             kalman = tracked_objects[best_match_id][0]

#         kalman.processNoiseCov *= process_noise
#         kalman.measurementNoiseCov *= measurement_noise
#         kalman.correct(measurement)

#         updated_tracked_objects[best_match_id] = (kalman, (x1, y1, x2, y2), time.time())

#         velocity_x = kalman.statePost[2][0]
#         velocity_y = kalman.statePost[3][0]
#         velocity = np.sqrt(velocity_x**2 + velocity_y**2)

#         print(f"Class: {class_label} | Predicted Position: ({predicted_x:.2f}, {predicted_y:.2f}) | Velocity: {velocity:.2f} px/s | Confidence: {confidence:.2f}")

#         color = (0, 255, 0) if class_id == 0 else (0, 0, 255)  
#         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
#         cv2.putText(frame, f"ID: {best_match_id} {class_label} C:{confidence:.2f}", 
#                     (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         # **Optimized Kalman Filter Predicted Bounding Box (Light Yellow)**
#         pred_x1, pred_y1 = int(predicted_x - predicted_width / 2), int(predicted_y - predicted_height / 2)
#         pred_x2, pred_y2 = int(predicted_x + predicted_width / 2), int(predicted_y + predicted_height / 2)
#         cv2.rectangle(frame, (pred_x1, pred_y1), (pred_x2, pred_y2), (0, 255, 255), 2)

#     tracked_objects = updated_tracked_objects

#     cv2.imshow("Mask Detection Tracking", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import time
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from ultralytics import YOLO
from scipy.spatial import distance

class FuzzyObjectTracker:
    def __init__(self, model_path, video_path):
        # Load YOLO model
        self.model = YOLO(model_path)
        
        # Video capture
        self.cap = cv2.VideoCapture(video_path)
        self.class_labels = {0: "Mask", 1: "No Mask"}
        
        # Object tracking storage
        self.tracked_objects = {}
        self.next_object_id = 0
        
        # Fuzzy Controller Setup
        self.fuzzy_controller = self.create_fuzzy_controller()

    def create_fuzzy_controller(self):
        # Input variables
        velocity = ctrl.Antecedent(np.arange(0, 10, 0.1), 'velocity')
        confidence = ctrl.Antecedent(np.arange(0, 1, 0.01), 'confidence')
        position_change = ctrl.Antecedent(np.arange(0, 100, 1), 'position_change')

        # Output variables
        iou_threshold = ctrl.Consequent(np.arange(0.1, 0.5, 0.01), 'iou_threshold')
        association_distance = ctrl.Consequent(np.arange(50, 200, 1), 'association_distance')

        # Membership functions
        velocity['low'] = fuzz.trimf(velocity.universe, [0, 0, 3])
        velocity['medium'] = fuzz.trimf(velocity.universe, [1, 5, 8])
        velocity['high'] = fuzz.trimf(velocity.universe, [6, 10, 10])

        confidence['low'] = fuzz.trimf(confidence.universe, [0, 0, 0.4])
        confidence['medium'] = fuzz.trimf(confidence.universe, [0.3, 0.6, 0.8])
        confidence['high'] = fuzz.trimf(confidence.universe, [0.7, 1, 1])

        position_change['small'] = fuzz.trimf(position_change.universe, [0, 0, 30])
        position_change['medium'] = fuzz.trimf(position_change.universe, [20, 50, 80])
        position_change['large'] = fuzz.trimf(position_change.universe, [60, 100, 100])

        iou_threshold['low'] = fuzz.trimf(iou_threshold.universe, [0.1, 0.1, 0.25])
        iou_threshold['medium'] = fuzz.trimf(iou_threshold.universe, [0.2, 0.35, 0.4])
        iou_threshold['high'] = fuzz.trimf(iou_threshold.universe, [0.35, 0.5, 0.5])

        association_distance['small'] = fuzz.trimf(association_distance.universe, [50, 50, 100])
        association_distance['medium'] = fuzz.trimf(association_distance.universe, [80, 125, 160])
        association_distance['large'] = fuzz.trimf(association_distance.universe, [140, 200, 200])

        # Fuzzy Rules
        rules = [
            ctrl.Rule(velocity['low'] & confidence['low'] & position_change['small'], iou_threshold['low']),
            ctrl.Rule(velocity['low'] & confidence['medium'] & position_change['small'], iou_threshold['medium']),
            ctrl.Rule(velocity['low'] & confidence['high'] & position_change['small'], iou_threshold['high']),
            ctrl.Rule(velocity['medium'] & confidence['low'] & position_change['small'], iou_threshold['low']),
            ctrl.Rule(velocity['medium'] & confidence['medium'] & position_change['small'], iou_threshold['medium']),
            ctrl.Rule(velocity['medium'] & confidence['high'] & position_change['small'], iou_threshold['high']),
            ctrl.Rule(velocity['high'] & confidence['low'] & position_change['small'], iou_threshold['low']),
            ctrl.Rule(velocity['high'] & confidence['medium'] & position_change['small'], iou_threshold['medium']),
            ctrl.Rule(velocity['high'] & confidence['high'] & position_change['small'], iou_threshold['high']),
            ctrl.Rule(velocity['low'] & confidence['low'] & position_change['medium'], iou_threshold['low']),
            ctrl.Rule(velocity['low'] & confidence['medium'] & position_change['medium'], iou_threshold['medium']),
            ctrl.Rule(velocity['low'] & confidence['high'] & position_change['medium'], iou_threshold['medium']),
            ctrl.Rule(velocity['medium'] & confidence['low'] & position_change['medium'], iou_threshold['low']),
            ctrl.Rule(velocity['medium'] & confidence['medium'] & position_change['medium'], iou_threshold['medium']),
            ctrl.Rule(velocity['medium'] & confidence['high'] & position_change['medium'], iou_threshold['medium']),
            ctrl.Rule(velocity['high'] & confidence['low'] & position_change['medium'], iou_threshold['low']),
            ctrl.Rule(velocity['high'] & confidence['medium'] & position_change['medium'], iou_threshold['low']),
            ctrl.Rule(velocity['high'] & confidence['high'] & position_change['medium'], iou_threshold['medium']),
            ctrl.Rule(velocity['low'] & confidence['low'] & position_change['large'], iou_threshold['low']),
            ctrl.Rule(velocity['low'] & confidence['medium'] & position_change['large'], iou_threshold['low']),
            ctrl.Rule(velocity['low'] & confidence['high'] & position_change['large'], iou_threshold['medium']),
            ctrl.Rule(velocity['medium'] & confidence['low'] & position_change['large'], iou_threshold['low']),
            ctrl.Rule(velocity['medium'] & confidence['medium'] & position_change['large'], iou_threshold['low']),
            ctrl.Rule(velocity['medium'] & confidence['high'] & position_change['large'], iou_threshold['medium']),
            ctrl.Rule(velocity['high'] & confidence['low'] & position_change['large'], iou_threshold['low']),
            ctrl.Rule(velocity['high'] & confidence['medium'] & position_change['large'], iou_threshold['low']),
            ctrl.Rule(velocity['high'] & confidence['high'] & position_change['large'], iou_threshold['low']),


            # Rules for Association Distance
            ctrl.Rule(velocity['low'] & confidence['low'] & position_change['small'], association_distance['small']),
            ctrl.Rule(velocity['low'] & confidence['medium'] & position_change['small'], association_distance['small']),
            ctrl.Rule(velocity['low'] & confidence['high'] & position_change['small'], association_distance['medium']),
            ctrl.Rule(velocity['medium'] & confidence['low'] & position_change['small'], association_distance['small']),
            ctrl.Rule(velocity['medium'] & confidence['medium'] & position_change['small'], association_distance['medium']),
            ctrl.Rule(velocity['medium'] & confidence['high'] & position_change['small'], association_distance['medium']),
            ctrl.Rule(velocity['high'] & confidence['low'] & position_change['small'], association_distance['medium']),
            ctrl.Rule(velocity['high'] & confidence['medium'] & position_change['small'], association_distance['medium']),
            ctrl.Rule(velocity['high'] & confidence['high'] & position_change['small'], association_distance['large']),
            ctrl.Rule(velocity['low'] & confidence['low'] & position_change['medium'], association_distance['small']),
            ctrl.Rule(velocity['low'] & confidence['medium'] & position_change['medium'], association_distance['small']),
            ctrl.Rule(velocity['low'] & confidence['high'] & position_change['medium'], association_distance['medium']),
            ctrl.Rule(velocity['medium'] & confidence['low'] & position_change['medium'], association_distance['small']),
            ctrl.Rule(velocity['medium'] & confidence['medium'] & position_change['medium'], association_distance['medium']),
            ctrl.Rule(velocity['medium'] & confidence['high'] & position_change['medium'], association_distance['large']),
            ctrl.Rule(velocity['high'] & confidence['low'] & position_change['medium'], association_distance['medium']),
            ctrl.Rule(velocity['high'] & confidence['medium'] & position_change['medium'], association_distance['large']),
            ctrl.Rule(velocity['high'] & confidence['high'] & position_change['medium'], association_distance['large']),
            ctrl.Rule(velocity['low'] & confidence['low'] & position_change['large'], association_distance['medium']),
            ctrl.Rule(velocity['low'] & confidence['medium'] & position_change['large'], association_distance['medium']),
            ctrl.Rule(velocity['low'] & confidence['high'] & position_change['large'], association_distance['large']),
            ctrl.Rule(velocity['medium'] & confidence['low'] & position_change['large'], association_distance['medium']),
            ctrl.Rule(velocity['medium'] & confidence['medium'] & position_change['large'], association_distance['large']),
            ctrl.Rule(velocity['medium'] & confidence['high'] & position_change['large'], association_distance['large']),
            ctrl.Rule(velocity['high'] & confidence['low'] & position_change['large'], association_distance['large']),
            ctrl.Rule(velocity['high'] & confidence['medium'] & position_change['large'], association_distance['large']),
            ctrl.Rule(velocity['high'] & confidence['high'] & position_change['large'], association_distance['large'])
        ]

        return ctrl.ControlSystem(rules)

    def calculate_iou(self, boxA, boxB):
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

    def create_kalman_filter(self):
        kalman = cv2.KalmanFilter(6, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=np.float32)
        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0, 0.5, 0],
            [0, 1, 0, 1, 0, 0.5],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-3
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
        kalman.errorCovPost = np.eye(6, dtype=np.float32)
        return kalman

    def calculate_velocity(self, kalman):
        velocity_x = kalman.statePost[2][0]
        velocity_y = kalman.statePost[3][0]
        return np.sqrt(velocity_x**2 + velocity_y**2)

    def calculate_position_change(self, prev_bbox, curr_bbox):
        prev_center_x = (prev_bbox[0] + prev_bbox[2]) / 2
        prev_center_y = (prev_bbox[1] + prev_bbox[3]) / 2
        curr_center_x = (curr_bbox[0] + curr_bbox[2]) / 2
        curr_center_y = (curr_bbox[1] + curr_bbox[3]) / 2
        
        return distance.euclidean(
            (prev_center_x, prev_center_y), 
            (curr_center_x, curr_center_y)
        )

    def dynamically_adjust_thresholds(self, velocity, confidence, position_change):
        simulation = ctrl.ControlSystemSimulation(self.fuzzy_controller)
        
        simulation.input['velocity'] = velocity
        simulation.input['confidence'] = confidence
        simulation.input['position_change'] = position_change
        
        simulation.compute()
        
        return (
            simulation.output['iou_threshold'], 
            simulation.output['association_distance']
        )

    def track(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model(frame)
            current_detections = []

            for result in results:
                detections = result.boxes.data.cpu().numpy()

                for detection in detections:
                    x1, y1, x2, y2, confidence, class_id = detection
                    class_id = int(class_id)
                    class_label = self.class_labels.get(class_id, "Unknown")

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

                # Dynamically compute thresholds
                velocity = 0  # Initial default
                position_change = 0  # Initial default
                
                if self.tracked_objects:
                    # Take the first tracked object as reference (you might want to improve this)
                    first_obj = list(self.tracked_objects.values())[0]
                    kalman = first_obj[0]
                    velocity = self.calculate_velocity(kalman)
                    prev_bbox = first_obj[1]
                    position_change = self.calculate_position_change(prev_bbox, (x1, y1, x2, y2))

                IOU_THRESHOLD, ASSOCIATION_DISTANCE_THRESHOLD = self.dynamically_adjust_thresholds(
                    velocity, confidence, position_change
                )

                for obj_id, obj_data in self.tracked_objects.items():
                    kalman, prev_bbox, _ = obj_data
                    prediction = kalman.predict()
                    
                    pred_x, pred_y = prediction[0][0], prediction[1][0]
                    pred_w, pred_h = prev_bbox[2] - prev_bbox[0], prev_bbox[3] - prev_bbox[1]
                    pred_x1, pred_y1 = pred_x - pred_w/2, pred_y - pred_h/2
                    pred_x2, pred_y2 = pred_x + pred_w/2, pred_y + pred_h/2

                    dist = distance.euclidean((pred_x, pred_y), (center_x, center_y))
                    iou = self.calculate_iou(prev_bbox, (x1, y1, x2, y2))

                    if iou > IOU_THRESHOLD or dist < ASSOCIATION_DISTANCE_THRESHOLD:
                        if iou > highest_iou or dist < min_distance:
                            best_match_id = obj_id
                            highest_iou = iou
                            min_distance = dist

                if best_match_id is None:
                    best_match_id = self.next_object_id
                    self.next_object_id += 1
                    kalman = self.create_kalman_filter()
                else:
                    kalman = self.tracked_objects[best_match_id][0]

                kalman.correct(measurement)

                updated_tracked_objects[best_match_id] = (kalman, (x1, y1, x2, y2), time.time())

                # Visualization and drawing remains the same as your original code
                color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"ID: {best_match_id} {class_label} C:{confidence:.2f}", 
                            (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            self.tracked_objects = updated_tracked_objects

            cv2.imshow("Fuzzy Logic Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# Usage
tracker = FuzzyObjectTracker("best.pt", "hospital.mp4")
tracker.track()