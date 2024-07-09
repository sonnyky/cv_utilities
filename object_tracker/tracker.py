import cv2
from scipy.optimize import linear_sum_assignment
import numpy as np
from kalman_filter import KalmanFilter

class Tracker:
    def __init__(self, max_age=10):
        self.kf = {}
        self.track_id = 0
        self.max_age = max_age
        self.track_frames = {}
        self.appearance_features = {}

    def get_appearance(self, frame, box):
        x1 = int(box[0, 0].item())
        y1 = int(box[0, 1].item())
        x2 = int(box[0, 2].item())
        y2 = int(box[0, 3].item())
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return np.zeros((3, 256)) # return dummy histogram if patch is not available
        hist = cv2.calcHist([patch], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def predict(self):
        predicted_centers = {}
        for tid in list(self.kf.keys()):
            predicted_centers[tid] = self.kf[tid].predict()
        return predicted_centers

    def update(self, frame, detections):
        detected_centers = [((box[0, 0].item() + box[0, 2].item()) / 2, (box[0, 1].item() + box[0, 3].item()) / 2) for box in detections]
        #detected_features = [self.get_appearance(frame, box) for box in detections]

        cost_matrix = np.zeros((len(detections), len(self.kf)))
        for i, (d_center) in enumerate(zip(detected_centers)):
            for j, (tid, kf) in enumerate(self.kf.items()):
                p_center = kf.predict()[:2]
                #p_feature = self.appearance_features[tid]
                spatial_cost = np.linalg.norm(d_center - p_center)
                #appearance_cost = np.linalg.norm(d_feature - p_feature)
                cost_matrix[i, j] = spatial_cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignments = []
        unassigned_detections = set(range(len(detections)))
        assigned_tracks = set()

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 50:  # Distance threshold
                assignments.append((list(self.kf.keys())[j], i))
                unassigned_detections.discard(i)
                assigned_tracks.add(list(self.kf.keys())[j])

        updated_tracks = set()
        for tid, d_id in assignments:
            self.kf[tid].update(np.array(detected_centers[d_id]))
            #self.appearance_features[tid] = detected_features[d_id]
            updated_tracks.add(tid)
            self.track_frames[tid] = 0  # Reset the count of missed frames

        for d_id in unassigned_detections:
            self.kf[self.track_id] = KalmanFilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_std_meas=0.1, y_std_meas=0.1)
            self.kf[self.track_id].update(np.array(detected_centers[d_id]))
            #self.appearance_features[self.track_id] = detected_features[d_id]
            self.track_frames[self.track_id] = 0
            self.track_id += 1

        for tid in list(self.kf.keys()):
            if tid not in updated_tracks:
                self.track_frames[tid] += 1
                if self.track_frames[tid] > self.max_age:
                    del self.kf[tid]
                    del self.track_frames[tid]
                    #del self.appearance_features[tid]

        return self.kf