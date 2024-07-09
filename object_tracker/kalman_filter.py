# Reference taken from https://machinelearningspace.com/2d-object-tracking-using-kalman-filter/
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/utils/kalman_filter.py#L151
import numpy as np
import scipy.linalg

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

class KalmanFilterXYAH:
    def __init__(self):
        """Initialize Kalman filter model matrices with motion and observation uncertainty weights."""
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current state estimate. These weights control
        # the amount of uncertainty in the model.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple:
        """
        Create track from unassociated measurement.

        Args:
            measurement (ndarray): Bounding box coordinates (x, y, a, h) with center position (x, y), aspect ratio a,
                and height h.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector (8 dimensional) and covariance matrix (8x8 dimensional)
                of the new track. Unobserved velocities are initialized to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """
        Run Kalman filter prediction step.

        Args:
            mean (ndarray): The 8 dimensional mean vector of the object state at the previous time step.
            covariance (ndarray): The 8x8 dimensional covariance matrix of the object state at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector and covariance matrix of the predicted state. Unobserved
                velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """
        Project state distribution to measurement space.

        Args:
            mean (ndarray): The state's mean vector (8 dimensional array).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            (tuple[ndarray, ndarray]): Returns the projected mean and covariance matrix of the given state estimate.
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """
        Run Kalman filter prediction step (Vectorized version).

        Args:
            mean (ndarray): The Nx8 dimensional mean matrix of the object states at the previous time step.
            covariance (ndarray): The Nx8x8 covariance matrix of the object states at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector and covariance matrix of the predicted state. Unobserved
                velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> tuple:
        """
        Run Kalman filter correction step.

        Args:
            mean (ndarray): The predicted state's mean vector (8 dimensional).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (ndarray): The 4 dimensional measurement vector (x, y, a, h), where (x, y) is the center
                position, a the aspect ratio, and h the height of the bounding box.

        Returns:
            (tuple[ndarray, ndarray]): Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(
            self,
            mean: np.ndarray,
            covariance: np.ndarray,
            measurements: np.ndarray,
            only_position: bool = False,
            metric: str = "maha",
    ) -> np.ndarray:
        """
        Compute gating distance between state distribution and measurements. A suitable distance threshold can be
        obtained from `chi2inv95`. If `only_position` is False, the chi-square distribution has 4 degrees of freedom,
        otherwise 2.

        Args:
            mean (ndarray): Mean vector over the state distribution (8 dimensional).
            covariance (ndarray): Covariance of the state distribution (8x8 dimensional).
            measurements (ndarray): An Nx4 matrix of N measurements, each in format (x, y, a, h) where (x, y)
                is the bounding box center position, a the aspect ratio, and h the height.
            only_position (bool, optional): If True, distance computation is done with respect to the bounding box
                center position only. Defaults to False.
            metric (str, optional): The metric to use for calculating the distance. Options are 'gaussian' for the
                squared Euclidean distance and 'maha' for the squared Mahalanobis distance. Defaults to 'maha'.

        Returns:
            (np.ndarray): Returns an array of length N, where the i-th element contains the squared distance between
                (mean, covariance) and `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            return np.sum(z * z, axis=0)  # square maha
        else:
            raise ValueError("Invalid distance metric")


class KalmanFilterXYWH(KalmanFilterXYAH):
    """
    For BoT-SORT. A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space (x, y, w, h, vx, vy, vw, vh) contains the bounding box center position (x, y), width
    w, height h, and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location (x, y, w, h) is taken as direct
    observation of the state space (linear observation model).
    """

    def initiate(self, measurement: np.ndarray) -> tuple:
        """
        Create track from unassociated measurement.

        Args:
            measurement (ndarray): Bounding box coordinates (x, y, w, h) with center position (x, y), width, and height.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector (8 dimensional) and covariance matrix (8x8 dimensional)
                of the new track. Unobserved velocities are initialized to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance) -> tuple:
        """
        Run Kalman filter prediction step.

        Args:
            mean (ndarray): The 8 dimensional mean vector of the object state at the previous time step.
            covariance (ndarray): The 8x8 dimensional covariance matrix of the object state at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector and covariance matrix of the predicted state. Unobserved
                velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance) -> tuple:
        """
        Project state distribution to measurement space.

        Args:
            mean (ndarray): The state's mean vector (8 dimensional array).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            (tuple[ndarray, ndarray]): Returns the projected mean and covariance matrix of the given state estimate.
        """
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance) -> tuple:
        """
        Run Kalman filter prediction step (Vectorized version).

        Args:
            mean (ndarray): The Nx8 dimensional mean matrix of the object states at the previous time step.
            covariance (ndarray): The Nx8x8 covariance matrix of the object states at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector and covariance matrix of the predicted state. Unobserved
                velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement) -> tuple:
        """
        Run Kalman filter correction step.

        Args:
            mean (ndarray): The predicted state's mean vector (8 dimensional).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (ndarray): The 4 dimensional measurement vector (x, y, w, h), where (x, y) is the center
                position, w the width, and h the height of the bounding box.

        Returns:
            (tuple[ndarray, ndarray]): Returns the measurement-corrected state distribution.
        """
        return super().update(mean, covariance, measurement)
