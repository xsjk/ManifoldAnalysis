import numpy as np
from typing import Literal
from itertools import product
import scipy.spatial
import scipy.spatial.distance
from sampler import SineDisribution, PolyDistribution
from typing import Iterable, Callable
import sympy as sp
from sklearn.decomposition import PCA
from functools import cache
import math

class Geometry:
    pass


class Subspace(Geometry):

    def __init__(self, x0: np.ndarray, basis: np.ndarray):
        '''
        Parameters
        ----------
        x0 : np.ndarray
            origin of the subspace
        basis : np.ndarray
            basis of the subspace
        '''
        x0 = np.array(x0)
        basis = np.array(basis)
        assert x0.ndim == 1, "x0 must be a vector"
        assert x0.shape[0] == basis.shape[1], "x0 and basis must have the same dimension"
        assert basis.ndim == 2, "basis must be a matrix"
        self.x0 = x0
        self.basis = basis
        self.dim = basis.shape[1]
        # orthonormal basis for the subspace
        self.basis = np.linalg.qr(basis.T)[0].T
        self.projection = self.basis.T @ self.basis
        # TODO: get the orthonormal basis for the complement subspace

    def project(self, X: np.ndarray):
        X = np.array(X, dtype=float)
        return (X - self.x0) @ self.projection + self.x0


class HyperPlane(Geometry):

    def __init__(self, x0: np.ndarray, n: np.ndarray):
        '''
        Parameters
        ----------
        x0 : np.ndarray
            point on the hyperplane
        n : np.ndarray
            normal vector of the hyperplane
        '''
        x0 = np.array(x0)
        n = np.array(n)
        assert x0.ndim == 1, "x0 must be a vector"
        assert x0.shape == n.shape, "x0 and n must have the same shape"
        self.x0 = x0
        self.n = n / np.linalg.norm(n)

    def project(self, X: np.ndarray):
        '''
        Project target point onto the hyperplane

        Parameters
        ----------
        targets : np.ndarray
            target points

        Returns
        -------
        np.ndarray
            projected points
        '''
        return self.scale(X, 0)

    def scale(self, X: np.ndarray, scale: float):
        '''
        Scale along the normal vector of the hyperplane

        Parameters
        ----------
        X : np.ndarray
            target points
        scale : float
            scale factor

        Returns
        -------
        np.ndarray
            scaled points
        '''
        if X.ndim == 1:
            X = X.reshape(1, -1)
        assert X.ndim == 2, "target must be a vector or a matrix"
        assert X.shape[1] == self.x0.shape[0], "target must have the same dimension as the hyperplane"
        d = np.dot((X - self.x0[None, :]), self.n)
        X_ = X + self.n[None, :] * d[:, None] * (scale - 1)
        return X_

    def distance(self, X: np.ndarray):
        '''
        Calculate the distance between the points and the hyperplane

        Parameters
        ----------
        X : np.ndarray
            target points

        Returns
        -------
        np.ndarray
            distances
        '''
        return np.abs(np.dot((X - self.x0), self.n))

class Line(Geometry):

    def __init__(self, x0: np.ndarray, d: np.ndarray):
        '''
        Parameters
        ----------
        x0 : np.ndarray
            point on the line
        d : np.ndarray
            direction vector of the line
        '''
        x0 = np.array(x0)
        d = np.array(d)
        assert x0.ndim == 1, "x0 must be a vector"
        assert x0.shape == d.shape, "x0 and d must have the same shape"
        self.x0 = x0
        self.d = d / np.linalg.norm(d)

    def intersect(self, hp: HyperPlane):
        '''
        Calculate the intersection point between the line and the hyperplane

        Parameters
        ----------
        hp : HyperPlane
            hyperplane

        Returns
        -------
        float
            intersection parameter t such that x = x0 + t * d
        '''
        if not isinstance(hp, HyperPlane):
            raise TypeError("hp must be a HyperPlane")

        n = hp.n
        x1 = hp.x0
        d = self.d
        x0 = self.x0
        t = np.dot((x1 - x0), n) / np.dot(d, n)
        return t

    def at(self, t: float | Iterable[float]):
        '''
        Calculate the point on the line at parameter t

        Parameters
        ----------
        t : float
            parameter

        Returns
        -------
        np.ndarray
            point on the line
        '''
        if isinstance(t, Iterable):
            t = np.array(t)
            return self.x0 + self.d * t[:, None]
        else:
            return self.x0 + self.d * t

    def project(self, X: np.ndarray):
        '''
        Project target point onto the line

        Parameters
        ----------
        targets : np.ndarray
            target points

        Returns
        -------
        np.ndarray
            projected points
        '''
        return self.at(np.dot((X - self.x0), self.d))


class ConvexHull(Geometry, scipy.spatial.ConvexHull):
    '''
    A convex hull in the N-dimensional space, inherited from scipy.spatial.ConvexHull to add some functionalities like sampling
    '''

    def sample(self, size):
        '''
        Uniformly sample points inside the convex hull

        Parameters
        ----------
        size : int
            number of points to be sampled

        Returns
        -------
        np.ndarray
            sampled points
        '''
        samples = []
        while len(samples) < size:
            new_samples = np.random.uniform(
                self.min_bound, self.max_bound, (size, len(self.min_bound)))
            new_samples = new_samples[self.contains(new_samples)]
            samples.extend(new_samples.tolist())
        return np.array(samples[:size])

    def contains(self, points):
        '''
        Check if the points are inside the convex hull

        Parameters
        ----------
        points : np.ndarray
            points to be checked

        Returns
        -------
        np.ndarray
            boolean array indicating whether the points are inside the convex hull
        '''
        A, b = self.equations[:, :-1], self.equations[:, -1:]
        eps = np.finfo(np.float32).eps
        return np.all(points @ A.T + b.T < eps, axis=1)


class HyperPlaneConvexHull(HyperPlane):
    '''
    A convex hull in the subspace of a hyperplane
    '''

    def __init__(self, x0: np.ndarray, n: np.ndarray, reduction: PCA, hull: ConvexHull):
        '''
        Parameters
        ----------
        x0 : np.ndarray
            point on the hyperplane
            (d dimensional)
        n : np.ndarray
            normal vector of the hyperplane
            (d dimensional)
        reduction : PCA
            PCA for dimension reduction
            (d dimensional <-> d-1 dimensional)
        hull : ConvexHull
            convex hull in the subspace
            (d-1 dimensional)
        '''
        super().__init__(x0, n)
        self.reduction = reduction
        self.hull = hull

    def sample(self, size: int) -> np.ndarray:
        samples = self.hull.sample(size)
        samples = self.reduction.inverse_transform(samples)
        return samples

    @property
    def area(self) -> float:
        # self.hull is created in a reduced dimension
        assert self.hull.points.shape[1] == self.x0.shape[0] - 1, "the dimension of the hull must be reduced by 1"
        return self.hull.volume

    @property
    def curvature(self) -> float:
        return 0.0

    @property
    def total_curvature(self) -> float:
        return 0.0

    @property
    def cur_area(self) -> float:
        return self.area


    @staticmethod
    def from_points(points: np.ndarray):
        '''
        Create a HyperPlaneConvexHull from points

        Parameters
        ----------
        points : np.ndarray
            points to be fitted

        Returns
        -------
        HyperPlaneConvexHull
            fitted HyperPlaneConvexHull
        '''
        dim = points.shape[1]
        x0 = points.mean(0)
        n = PCA(dim).fit(points).components_[-1]
        points = HyperPlane(x0, n).project(points)
        reduction = PCA(dim - 1).fit(points)
        return HyperPlaneConvexHull(x0, n, reduction, ConvexHull(reduction.transform(points)))


class HyperSphere(Geometry):

    def __init__(self, center: np.ndarray, radius: float):
        '''
        Parameters
        ----------
        center : np.ndarray
            center of the HyperSphere
        radius : float
            radius of the HyperSphere
        '''
        self.center = np.array(center)
        self.radius = radius
        assert self.center.ndim == 1, "center must be a vector"
        assert self.dimension > 1, "the dimension of the HyperSphere must be greater than 1"
        self.r_distribution = PolyDistribution(self.dimension - 1)

    def project(self, X: np.ndarray) -> np.ndarray:
        '''
        Project target point onto the HyperSphere Surface

        Parameters
        ----------
        X : np.ndarray
            target points

        Returns
        -------
        np.ndarray
            projected points
        '''
        X = np.array(X)
        assert X.ndim == 2, "target must be a matrix"

        X -= self.center[None, :]
        X *= self.radius / np.linalg.norm(X, axis=1)[..., None]
        X += self.center[None, :]
        return X

    def sample(self, size: int) -> np.ndarray:
        '''
        Sample points in the HyperSphere

        Parameters
        ----------
        size : int
            number of points to be sampled

        Returns
        -------
        np.ndarray
            sampled points
        '''
        X = np.random.normal(size=(size, self.dimension))
        X /= np.linalg.norm(X, axis=1)[..., None]
        X *= self.radius * self.r_distribution.rvs(size)[..., None]
        X += self.center
        return X

    @property
    def volume(self) -> float:
        '''
        Calculate the volume of the HyperSphere

        Returns
        -------
        float
            volume of the HyperSphere
        '''
        return np.pi ** (self.dimension / 2) / math.gamma(self.dimension / 2 + 1) * self.radius ** self.dimension

    @property
    def area(self) -> float:
        '''
        Calculate the area of the HyperSphere Surface

        Returns
        -------
        float
            area of the HyperSphere Surface
        '''
        return np.pi ** (self.dimension / 2) / math.gamma(self.dimension / 2 + 1) * self.radius ** (self.dimension - 1) * self.dimension

    @property
    def dimension(self) -> int:
        '''
        Get the dimension of the HyperSphere

        Returns
        -------
        int
            dimension of the HyperSphere
        '''
        return self.center.shape[0]

    @property
    def curvature(self) -> float:
        return 1 / self.radius ** 3

    @property
    def total_curvature(self) -> float:
        return self.area * self.curvature

    @property
    def cur_area(self) -> float:
        return self.area * (self.curvature + 1)


def spherical_to_cartesian_uniform(*angles: float | np.ndarray):
    '''
    Convert spherical coordinates to cartesian coodinates on the uniform sphere

    e.g. for 4D sphere, the input angles are `psi`, `theta`, `phi`
    ```
    x = np.sin(phi) * np.sin(theta) * np.sin(psi)
    y = np.sin(phi) * np.sin(theta) * np.cos(psi)
    z = np.sin(phi) * np.cos(theta)
    w = np.cos(phi)
    ```

    Parameters
    ----------
    *angles : float | Iterable[float]
        the angles of the spherical coordinate from outer to inner

    Returns
    -------
    np.ndarray : (n, d)
        the coordinates in cartesian
    '''
    return np.hstack((np.sin(angles[-1])[:, None] * spherical_to_cartesian_uniform(*angles[:-1]), np.cos(angles[-1])[:, None])) if len(angles) else 1


def scale_matrix(direction: np.ndarray, scale: float) -> np.ndarray:
    '''
    Get the scale matrix along the given direction

    Parameters
    ----------
    direction : np.ndarray
        direction of the scale
    scale : float
        scale factor

    Returns
    -------
    np.ndarray
        scale transformation matrix
    '''
    direction = direction / np.linalg.norm(direction)
    return np.eye(direction.shape[0]) + (scale - 1) * direction[:, None] @ direction[None, :]


def rotation_matrix(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    '''
    Get the rotation matrix that rotate v1 to v2

    Parameters
    ----------
    v1 : np.ndarray
        original vector
    v2 : np.ndarray
        target vector

    Returns
    -------
    np.ndarray
        rotation matrix
    '''
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return scale_matrix(v1 + v2, -1) @ scale_matrix(v1, -1)


class HyperSphereShell(HyperSphere):

    def sample(self, size: int) -> np.ndarray:
        '''
        Sample points on the HyperSphere Shell

        Parameters
        ----------
        size : int
            number of points to be sampled

        Returns
        -------
        np.ndarray
            sampled points
        '''
        X = np.random.normal(size=(size, self.dimension))
        X /= np.linalg.norm(X, axis=1)[..., None]
        X *= self.radius
        X += self.center
        return X

    def at(self, *angles):
        '''
        Get the point coordinate on the HyperSphere Shell at given angles

        Parameters
        ----------
        *angles : np.ndarray
            angles in HyperSphere coordinates

        Returns
        -------
        np.ndarray
            points on the HyperSphere Shell
        '''
        return self.radius * spherical_to_cartesian_uniform(*angles)


class HyperSphereShellCap(HyperSphere):

    def __init__(self, center: np.ndarray, radius: float, orientation: np.ndarray, angle: float):
        '''
        Parameters
        ----------
        center : np.ndarray
            center of the HyperSphereCap
        radius : float
            radius of the HyperSphereCap
        orientation : np.ndarray
            orientation of the HyperSphereCap
        angle : float
            half angle of the HyperSphereCap, must be in range [0, pi]
        '''
        super().__init__(center, radius)
        self.orientation = orientation / np.linalg.norm(orientation)
        self.angle = angle

        self.distributions = [SineDisribution(
            i) for i in range(self.dimension - 1)]
        self.rotation = rotation_matrix(
            np.array([0] * (self.dimension - 1) + [1]), self.orientation)

    def sample(self, size: int) -> np.ndarray:
        '''
        Sample points on the HyperSphereCap

        Parameters
        ----------
        size : int
            number of points to be sampled

        Returns
        -------
        np.ndarray
            sampled points
        '''
        r = [d.rvs(size) for d in self.distributions[:-1]] + \
            [self.distributions[-1].rvs(size, range=(0, self.angle))]
        samples = spherical_to_cartesian_uniform(*r)
        assert isinstance(samples, np.ndarray)
        samples[:, 0] *= np.random.choice([-1, 1], size)
        return samples @ self.rotation.T * self.radius + self.center[None, :]

    @property
    def area(self) -> float:
        '''
        Calculate the area of the HyperSphereCap Surface

        Returns
        -------
        float
            area of the HyperSphereCap
        '''
        return self.area_coeff_formula(self.dimension)(self.radius, self.angle)

    @staticmethod
    @cache
    def area_coeff_formula(d: int) -> Callable:
        '''
        Get the formula for the coefficient of the area of the d-dimension HyperSphereCap
        Parameters
        ----------
        d : int
            dimension of the HyperSphereCap
        Returns
        -------
        sympy.Expr
            the formula for the area of the d-dimension HyperSphereCap
        '''
        from sympy.abc import r, theta
        expr = 2 * r ** (d - 1) * sp.prod(sp.integrate(sp.sin(theta)
                                                       ** i, (theta, 0, theta)) for i in range(d - 1))
        return sp.lambdify((r, theta), expr, 'numpy')

    @staticmethod
    def from_points(x0: np.ndarray, radius: float, points: np.ndarray, filter_rate: float = 0.9):
        x0 = np.array(x0)

        assert points.ndim == 2, "`points` must be a matrix, each row is a point"
        assert points.shape[1] == x0.shape[0], f"`points` must have the same dimension as `x0`, got {points.shape[1]} and {x0.shape[0]}. You may need to manually set `usedim` when loading the component"

        r = radius
        points_whitened = (points - x0)
        points_whitened = points_whitened / \
            np.linalg.norm(points_whitened, axis=1)[:, np.newaxis]
        points_mean = np.mean(points_whitened, axis=0)
        points_mean /= np.linalg.norm(points_mean)

        sphere_distances = np.arccos(np.dot(points_whitened, points_mean))
        sphere_distances_argsort = np.argsort(sphere_distances)

        # 90% point indices
        sphere_distances_argsort_filtered = sphere_distances_argsort[:int(
            len(sphere_distances_argsort) * filter_rate)]
        points[sphere_distances_argsort_filtered]
        # 90% point edge radius

        edge_point_index = sphere_distances_argsort[int(
            len(sphere_distances_argsort) * filter_rate)]
        sphere_distances_max = sphere_distances[edge_point_index]

        theta = sphere_distances_max

        return HyperSphereShellCap(x0, r, points_mean, theta)


class HyperCone(Geometry):

    def __init__(self, vertex, direction, angle):
        '''
        Parameters
        ----------
        vertex : np.ndarray
            vertex of the HyperCone
        direction : np.ndarray
            direction of the HyperCone
        angle : float
            half angle of the HyperCone, must be in range [0, pi]
        '''
        self.vertex = np.array(vertex, dtype=float)
        self.direction = np.array(direction, dtype=float)
        self.direction /= np.linalg.norm(self.direction)
        self.angle = angle

    def project(self, X: np.ndarray) -> np.ndarray:
        '''
        Project target point onto the HyperCone Surface

        Parameters
        ----------
        X : np.ndarray
            target points

        Returns
        -------
        np.ndarray
            projected points
        '''
        X = np.array(X)
        assert X.ndim == 2, "target must be a matrix"
        X -= self.vertex
        t = np.dot(X, self.direction)
        X_ = self.direction * t[:, None]
        d = X - X_
        d_length = np.linalg.norm(d, axis=1)
        d /= d_length[:, None]
        l = np.sin(self.angle) * d + np.cos(self.angle) * self.direction
        l_length = d_length * np.sin(self.angle) + t * np.cos(self.angle)
        l *= l_length[:, None]
        return l + self.vertex


class LineSegment(Line):

    def __init__(self, x0: np.ndarray, x1: np.ndarray):
        '''
        Parameters
        ----------
        x0 : np.ndarray
            start point of the line segment
        x1 : np.ndarray
            end point of the line segment
        '''
        self.x0 = np.array(x0)
        self.x1 = np.array(x1)
        assert self.x0.shape == self.x1.shape, "x0 and x1 must have the same shape"
        assert self.x0.ndim == 1, "x0 must be a vector"
        d = self.x1 - self.x0
        self.length = np.linalg.norm(d)
        super().__init__(self.x0, d)

    def intersect(self, hp: HyperPlane):
        t = super().intersect(hp)
        if t < 0 or t > self.length:
            return None
        return t


class Box:

    def __init__(self, mins: np.ndarray, maxs: np.ndarray):
        '''
        Parameters
        ----------
        mins : np.ndarray
            minimum coordinates of the box
        maxs : np.ndarray
            maximum coordinates of the box
        '''

        mins = np.array(mins)
        maxs = np.array(maxs)

        assert mins.shape == maxs.shape, "mins and maxs must have the same shape"
        assert mins.ndim == 1, "mins must be a vector"

        dim = mins.shape[0]

        self.vertices = np.array(list(product(*zip(mins, maxs))))
        self.edges = []
        for i in range(dim):
            indices = list(range(1 << dim))
            while indices:
                src = indices.pop(0)
                dst = src ^ 1 << i
                indices.remove(dst)
                self.edges.append((src, dst))
        self.edges = np.array(self.edges)
        self.faces = self.edges.reshape(
            dim, -1, 2).transpose(0, 2, 1).reshape(dim << 1, -1)

    def intersect(self, hp: HyperPlane) -> np.ndarray:
        '''
        Calculate the intersection between the box and the hyperplane

        Parameters
        ----------
        hp : HyperPlane
            hyperplane

        Returns
        -------
        np.ndarray
            intersection points
        '''
        if not isinstance(hp, HyperPlane):
            raise TypeError("hp must be a HyperPlane")

        intersections = []
        for ij in self.edges:
            line = LineSegment(*self.vertices[ij])
            t = line.intersect(hp)
            if t:
                intersection = line.at(t)
                intersections.append(intersection)
        intersections = np.array(intersections)
        return intersections

def distance_to_spheres(
    data: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:

    assert data.ndim >= 3
    assert centers.ndim == 2
    assert radii.ndim == 1

    n_spheres, n_points_per_component, n_dim = data.shape[-3:]

    assert centers.shape == (n_spheres, n_dim)
    assert radii.shape == (n_spheres,)

    distances: np.ndarray = np.abs(np.linalg.norm(data - centers[:, None, :], axis=-1) - radii[:, None])
    assert distances.shape[-2:] == (n_spheres, n_points_per_component)

    return distances


def distance_to_sphere(
    data: np.ndarray,
    center: np.ndarray,
    radius: np.ndarray,
):
    assert data.ndim >= 1
    assert center.ndim == 1
    assert radius.ndim == 0

    distances: np.ndarray = np.abs(np.linalg.norm(data - center, axis=-1) - radius)
    return distances


def distance_to_planes(
    data: np.ndarray,
    x0s: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:

    assert data.ndim >= 3
    assert x0s.ndim == 2
    assert normals.ndim == 2

    n_planes, n_points_per_component, n_dim = data.shape[-3:]

    assert x0s.shape == (n_planes, n_dim)
    assert normals.shape == (n_planes, n_dim)


    distances = np.abs(np.einsum(
        '...ij,...j->...i',
        data - x0s[:, None, :],
        normals
    ))

    assert distances.shape[-2:] == (n_planes, n_points_per_component)
    return distances


def distance_to_plane(
    data: np.ndarray,
    x0: np.ndarray,
    normal: np.ndarray
) -> np.ndarray:

    assert data.ndim >= 1
    assert x0.ndim == 1
    assert normal.ndim == 1

    distance: np.ndarray = np.abs(np.dot(data - x0, normal))
    return distance

def symmetric_hausdorff_distance(X: np.ndarray, Y: np.ndarray) -> float:
    return max(scipy.spatial.distance.directed_hausdorff(X, Y)[0],
               scipy.spatial.distance.directed_hausdorff(Y, X)[0])

def distance_to_pointset(
    data: np.ndarray,
    points: np.ndarray,
    kernel: Literal["numpy", "scipy", "cupy", "torch"] = "scipy"
) -> np.ndarray:

    # use hausdorff distance
    assert data.ndim >= 2
    assert points.ndim == 2

    assert points.shape[-1] == data.shape[-1]

    n_points, n_dim = points.shape
    *n_data, n_points_per_data, n_dim = data.shape

    match kernel:
        case "numpy":
            data = data.reshape(-1, data.shape[-1])
            cdist = scipy.spatial.distance.cdist(data, points)
            cdist = cdist.reshape((*n_data, n_points_per_data, n_points))
            return np.maximum(cdist.min(-1).max(-1), cdist.min(-2).max(-1))

        case "scipy":
            result = np.empty(n_data)
            for i in np.ndindex(*result.shape):
                result[*i] = symmetric_hausdorff_distance(data[*i], points)
            return result

        case "cupy":
            import cupy as cp
            import cupyx.scipy.spatial.distance
            cdist = cupyx.scipy.spatial.distance.cdist(
                cp.array(data.reshape(-1, data.shape[-1])),
                cp.array(points)
            )
            cdist = cdist.reshape((*n_data, n_points_per_data, n_points))
            return cp.maximum(cdist.min(-1).max(-1), cdist.min(-2).max(-1)).get()

        case "torch":
            import torch
            cdist = torch.cdist(
                torch.from_numpy(data.reshape(-1, data.shape[-1])).cuda(),
                torch.from_numpy(points).cuda()
            )
            cdist = cdist.reshape((*n_data, n_points_per_data, n_points))
            return torch.maximum(
                cdist.min(-1).values.max(-1).values,
                cdist.min(-2).values.max(-1).values
            ).cpu().numpy()
        case _:
            raise ValueError("Invalid kernel")


def distance_to_pointsets(
    data: np.ndarray, # type: ignore
    points: np.ndarray, # type: ignore
    kernel: Literal["numpy", "scipy", "cupy", "torch"] = "numpy"
) -> np.ndarray:

    # use hausdorff distance
    assert data.ndim >= 3
    assert points.ndim == 3

    assert points.shape[-3] == data.shape[-3]
    assert points.shape[-1] == data.shape[-1]

    *n_data, n_components, n_points_per_data, n_dim = data.shape
    n_components, n_points, n_dim = points.shape

    match kernel:
        case "scipy":
            result = np.empty((*n_data, n_components))
            for i in np.ndindex(*n_data):
                for j in range(n_components):
                    result[*i, j] = symmetric_hausdorff_distance(data[*i, j], points[j])
            return result
        case "numpy":
            result = np.empty((*n_data, n_components))
            for j in range(n_components):
                cdist = scipy.spatial.distance.cdist(
                    data[..., j, :, :].reshape(-1, n_dim),
                    points[j, :, :]
                )
                cdist = cdist.reshape((*n_data, n_points_per_data, n_points))
                np.maximum(cdist.min(-1).max(-1), cdist.min(-2).max(-1), out=result[..., j])
            return result
        case "cupy":
            import cupy as cp
            import cupyx.scipy.spatial.distance
            result = cp.empty((*n_data, n_components))
            data = cp.array(data)
            points: cp.ndarray = cp.array(points) # type: ignore
            for j in range(n_components):
                cdist = cupyx.scipy.spatial.distance.cdist(
                    data[..., j, :, :].reshape(-1, n_dim),
                    points[j, :, :]
                )
                cdist = cdist.reshape((*n_data, n_points_per_data, n_points))
                cp.maximum(cdist.min(-1).max(-1), cdist.min(-2).max(-1), out=result[..., j])
            return result.get()
        case "torch":
            import torch
            # for torch there no nead to iterate n_components since torch.cdist support batch computation
            data: torch.Tensor = torch.from_numpy(data).cuda()
            data = torch.moveaxis(data, -3, 0)
            points: torch.Tensor = torch.from_numpy(points).cuda()
            cdist = torch.cdist(
                data.reshape(n_components, -1, n_dim),
                points
            )
            cdist = cdist.reshape((n_components, *n_data, n_points_per_data, n_points))
            result = torch.maximum(cdist.min(-1).values.max(-1).values, cdist.min(-2).values.max(-1).values)
            result = torch.moveaxis(result, 0, -1)
            return result.cpu().numpy()
        case _:
            result = np.empty((*n_data, n_components))
            for j in range(n_components):
                result[..., j] = distance_to_pointset(data[..., :, j, :], points[j, :, :], kernel=kernel)
            return result


if __name__ == '__main__':

    import plotly.express as px

    # HyperSphereShellCap test
    hc = HyperSphereShellCap(
        center=np.array([1, 1, 1]),
        radius=2,
        orientation=np.array([1, 1, 1]),
        angle=np.pi / 6
    )
    x, y, z = hc.sample(1000).T

    fig = px.scatter_3d(x=x, y=y, z=z)
    fig.update_layout(scene_aspectmode='data')
    fig.show()

    # HyperCone test
    hc = HyperCone(
        vertex=[-1, -1, -1],
        direction=[1, 1, 1],
        angle=np.pi / 6
    )
    X = hc.project(np.random.rand(1000, 3))
    fig = px.scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2])
    fig.update_layout(scene_aspectmode='data')
    fig.show()
