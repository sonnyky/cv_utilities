import cv2
import torch
from ultralytics.utils import yaml_load, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from tracker import Tracker
from object_detector import ObjectDetector
from object_detector import get_center
from object_detector import euclidean_distance
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
from bot_sort import BOTSORT

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
    tracker_yaml = check_yaml("botsort.yaml")
    cfg = IterableSimpleNamespace(**yaml_load(tracker_yaml))
    detector = ObjectDetector()
    tracker = BOTSORT(args=cfg, frame_rate=30)

    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if not ret:
            break

        boxes, results = detector.predict(frame, "car")
        # tracks = tracker.update(det, im0s[i])
        # here det is ultralytics.engine.results.Boxes object with attributes:
        tracks = tracker.update(boxes.cpu().numpy(), [frame])
        if len(tracks) != 0:
            idx = tracks[:, -1].astype(int)
            results[0] = results[0][idx]
            update_args = {"boxes": torch.as_tensor(tracks[:, :-1])}
            results[0].update(**update_args)
        annotated_frame = results[0].plot()
        cv2.imshow('Car tracker', annotated_frame)
        key = cv2.waitKey(50)
        if key & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
