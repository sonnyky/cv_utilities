import numpy as np

from ultralytics import YOLO


def get_center(box):
    x1 = box[0, 0].item()
    y1 = box[0, 1].item()
    x2 = box[0, 2].item()
    y2 = box[0, 3].item()
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return np.array([cx, cy])


def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)


class ObjectDetector:
    def __init__(self):
        # Load a pretrained YOLOv8n model
        self.model = YOLO("yolov8n.pt")
        self.names = self.model.names

    def predict(self, image, object_name):
        # Run inference on the source
        results = self.model(image, iou=0.7, conf=0.3)  # only process one image at a time, returns list of 1
        target_object_boxes_list = []

        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs

            if boxes is not None:
                return boxes, results
        return target_object_boxes_list, results
