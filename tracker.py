import cv2
import os
from object_detector import ObjectDetector
from object_detector import get_center
from object_detector import euclidean_distance
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
from kalman_filter import KalmanFilter
from scipy.optimize import linear_sum_assignment
import numpy as np

video_capture = cv2.VideoCapture("traffic1.mp4")
dt = 0.1
u_x = 1
u_y = 1
std_acc = 1
x_std_meas = 0.1
y_std_meas = 0.1

if not video_capture.isOpened():
    print("Error: Could not open video.")
else:

    # Get video properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print(f"Video FPS: {fps}")
    print(f"Total Frames: {frame_count}")
    print(f"Duration (s): {duration}")

    detector = ObjectDetector()

    kf = {}
    track_id = 0
    max_frames_to_keep = 20  # Number of frames to keep a track without updates
    track_frames = {}
    max_distance_threshold = 50  # Max distance to consider a detection for reactivation

    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if not ret:
            break

        car_boxes = detector.predict(frame, ["car", "bus", "train", "truck"])
        if not car_boxes:
            pass
        else:
            # detect and show YOLO results
            annotator = Annotator(frame)
            for box in car_boxes:
                b = box[0]
                annotator.box_label(b, "", (0, 255, 0))
            frame = annotator.result()

            # get detected centers
            detected_centers = [get_center(box) for box in car_boxes]

            # Predict the positions of existing tracks
            predicted_centers = {}
            for tid in list(kf.keys()):
                predicted_centers[tid] = kf[tid].predict()

            # Create a cost matrix
            cost_matrix = np.zeros((len(detected_centers), len(predicted_centers)))
            for i, d_center in enumerate(detected_centers):
                for j, (tid, p_center) in enumerate(predicted_centers.items()):
                    cost_matrix[i, j] = euclidean_distance(d_center, p_center)

            # Solve the assignment problem using the Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Create assignments and unassigned detections
            assignments = []
            unassigned_detections = set(range(len(detected_centers)))
            assigned_tracks = set()

            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < max_distance_threshold:  # Dynamic distance threshold
                    assignments.append((list(predicted_centers.keys())[j], i))
                    unassigned_detections.discard(i)
                    assigned_tracks.add(list(predicted_centers.keys())[j])

            # Update existing tracks
            updated_tracks = set()
            for tid, d_id in assignments:
                kf[tid].update(detected_centers[d_id])
                updated_tracks.add(tid)
                track_frames[tid] = 0  # Reset the count of missed frames
                cv2.circle(frame, (int(detected_centers[d_id][0]), int(detected_centers[d_id][1])), 5, (0, 0, 255), -1)
                car_box = car_boxes[d_id]
                x_val = int(car_box[0, 0].item())
                y_val = int(car_box[0, 1].item())
                cv2.putText(frame, f"ID: {tid}", (x_val, y_val - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Create new tracks for unassigned detections
            for d_id in unassigned_detections:
                # Try to match with inactive tracks first
                matched = False
                for tid, frame_count in track_frames.items():
                    if frame_count > 0 and euclidean_distance(detected_centers[d_id], kf[tid].predict()) < max_distance_threshold:
                        kf[tid].update(detected_centers[d_id])
                        track_frames[tid] = 0
                        matched = True
                        break

                if not matched:
                    kf[track_id] = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
                    kf[track_id].update(detected_centers[d_id])
                    track_frames[track_id] = 0
                    track_id += 1

            # Handle missed tracks
            for tid in list(kf.keys()):
                if tid not in updated_tracks:
                    track_frames[tid] += 1
                    if track_frames[tid] > max_frames_to_keep:
                        del kf[tid]  # Remove the track if it hasn't been updated for a while
                        del track_frames[tid]
                    else:
                        if tid in predicted_centers:
                            pred_center = predicted_centers[tid]
                            x_val = int(pred_center[0][0])
                            y_val = int(pred_center[0][1])
                            cv2.circle(frame, (x_val, y_val), 5, (0, 0, 255), -1)
                            cv2.putText(frame, f"ID: {tid} (pred)", (x_val, y_val - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('YOLO V8 Detection', frame)
        key = cv2.waitKey(50)
        if key & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
