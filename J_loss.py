import numpy as np

DIMENSION = 4


def J_line_single(c, n, x):
    diff = x[0:DIMENSION] - c
    return np.abs(np.linalg.norm(diff-np.dot(diff, n)*n))


def J_line(c, n, X):
    n = n[0]
    _sum = 0
    for x in X:
        _sum += J_line_single(c, n, x)
    return _sum / len(X)


def J_plane(c, n, X):
    _sum = 0
    for x in X:
        _sum += np.abs((np.dot(x[0:DIMENSION] - c, n)))
    return _sum / len(X)


def J_plane_single(c, n, x):
    return np.abs((np.dot(x[0:DIMENSION] - c, n)))


def J_sphere(c, r, X):
    sum_ = 0
    for x in X:
        sum_ += np.abs((np.linalg.norm(x - c) - r))
    return sum_ / len(X)


def J_cylinder(c, w, r, X):
    sum_ = 0
    for x in X:
        sum_ += np.abs((np.sqrt(np.linalg.norm(x - c))
                        - np.abs(np.dot(x - c, w) ** 2) - r))
    return sum_ / len(X)


def J_cone(c, w, alpha, X):
    """
    here alpha = cos(the inner angle of the cone)
    """
    sum_ = 0
    for x in X:
        error_angle = np.arccos(
            np.dot(x - c, w) / np.linalg.norm(x - c)) - np.arccos(alpha)
        sum_ += np.abs((np.linalg.norm(x - c) * np.sin(error_angle)))
    return sum_ / len(X)
