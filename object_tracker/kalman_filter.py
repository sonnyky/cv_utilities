# Reference taken from https://machinelearningspace.com/2d-object-tracking-using-kalman-filter/

import numpy as np


# by default we assume an isotropic sensor noise with std deviations in the y direction set to None
class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas=None):
        # define sampling time
        self.dt = dt
        self.u = np.array([[u_x], [u_y]])
        # define the state transition matrix
        self.F = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # define the control input array
        self.B = np.array([[0.5 * self.dt ** 2, 0],
                           [0, 0.5 * self.dt ** 2],
                           [self.dt, 0],
                           [0, self.dt]])

        # define the measurement matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # initial state covariance matrix
        self.P = np.eye(self.F.shape[1])
        self.Q = (std_acc ** 2) * np.array([[0.25 * dt ** 4, 0, 0.5 * dt ** 3, 0],
                                            [0, 0.25 * dt ** 4, 0, 0.5 * dt ** 3],
                                            [0.5 * dt ** 3, 0, dt ** 2, 0],
                                            [0, 0.5 * dt ** 3, 0, dt ** 2]])

        # Measurement noise covariance matrix
        if y_std_meas is None:
            self.R = x_std_meas ** 2 * np.eye(self.H.shape[0])
        else:
            self.R = np.array([[x_std_meas ** 2, 0],
                               [0, y_std_meas ** 2]])
        self.x = np.zeros((4, 1))  # Initialize as a column vector

    def predict(self):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[:1]

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))
        return self.x[:2]
