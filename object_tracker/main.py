import cv2
from tracker import Tracker
from object_detector import ObjectDetector
from object_detector import get_center
from object_detector import euclidean_distance
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated


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
    tracker = Tracker()

    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if not ret:
            break

        car_boxes = detector.predict(frame, "car")
        if car_boxes:
            tracked_objects = tracker.update(frame, car_boxes)

            for tid, kf in tracked_objects.items():
                pred_center = kf.predict()
                x_val = int(pred_center[0, 0].item())
                y_val = int(pred_center[0, 1].item())
                cv2.circle(frame, (x_val, y_val), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"ID: {tid}", (x_val, y_val - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                for box in car_boxes:
                    x1 = int(box[0, 0].item())
                    y1 = int(box[0, 1].item())
                    x2 = int(box[0, 2].item())
                    y2 = int(box[0, 3].item())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('SIRT Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
