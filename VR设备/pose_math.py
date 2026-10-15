"""SE(3) helpers shared by the two Cartesian backends (metres, radians)."""
import numpy as np


def rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def rotation_distance(a, b):
    return float(np.arccos(np.clip((np.trace(rpy_matrix(a).T @ rpy_matrix(b))-1)/2, -1, 1)))


def valid_transform(value):
    t = np.asarray(value, dtype=float).reshape(4, 4)
    if (not np.isfinite(t).all() or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5)
            or abs(np.linalg.det(t[:3, :3])-1) > 1e-5):
        raise ValueError("invalid rigid transform")
    return t.copy()
