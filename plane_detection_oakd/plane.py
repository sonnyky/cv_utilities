import numpy as np
import open3d as o3d


def fit_plane(self, point_cloud: o3d.geometry.PointCloud):
    points = np.asarray(point_cloud.points)
    G = np.ones_like(points)
    G[:, 0] = points[:, 0]
    G[:, 1] = points[:, 1]
    Z = points[:, 2]
    (k_x, k_y, k_z), resid, rank, s = np.linalg.lstsq(G, Z, rcond=None)
    normal = np.array([k_x, k_y, -1])
    normal = normal / np.linalg.norm(normal)

    self.plane_normal = normal
    self.plane_distance = k_z
    self.plane_coeffs = (k_x, k_y, k_z)

    self.fitted = True

    return normal, k_z
