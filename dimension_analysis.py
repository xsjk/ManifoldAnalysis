from typing import Literal
import joblib
import numpy as np
import cupy as cp
import scipy.spatial.distance
import cuvs.distance

def estimate_dim(data: np.ndarray, t: float | None = None, neighbors: int | None = None, kernel: Literal["numpy", "cupy"] = "cupy") -> np.ndarray:
    """
    Compute the dimension of a point cloud

    Parameters
    ----------
    data : np.ndarray of shape (n_points, n_dimensions)
        The point cloud to estimate the dimension of
    t : float, optional
        The bandwidth of the kernel. If None, it is estimated as the inverse of the mean distance to the k-nearest neighbors
    neighbors : int, optional
        The number of neighbors to consider when estimating the bandwidth
    kernel : str, optional
        The library to use for the computation. Choose between 'numpy' and 'cupy'

    Returns
    -------
    np.ndarray of shape (n_points,)
        The estimated dimension of the point cloud
    """

    n_points, n_dimensions = data.shape

    match kernel:
        case "numpy":
            d = scipy.spatial.distance.cdist(data, data, metric='sqeuclidean')
            d.sort(axis=1)
            d = d[:, 1:][:, :neighbors]
            if t is None:
                t = 1 / d.mean()
            d -= d[:, [0]]
            w = np.exp(-d / (4 * t)) # type: ignore
            return (w * d).sum(axis=-1) / (2 * t * w.sum(axis=-1)) # type: ignore
        case "cupy":
            d = cuvs.distance.pairwise_distance(data, data, metric='sqeuclidean')
            d = cp.asarray(d)
            d.sort(axis=1)
            d = d[:, 1:][:, :neighbors]
            if t is None:
                t = 1 / d.mean()
            d -= d[:, [0]]
            w = cp.exp(-d / (4 * t)) # type: ignore
            return cp.asnumpy((w * d).sum(axis=-1) / (2 * t * w.sum(axis=-1))) # type: ignore
        case _:
            raise ValueError("Invalid kernel. Choose between 'numpy' and 'cupy'")


def estimate_dims(
    data: np.ndarray,
    t: float | None = None,
    neighbors: int | None = None,
    kernel: Literal["numpy", "cupy"] = "cupy",
    progress_bar: bool = False,
    parallel: bool = False,
    device: int | None = None,
) -> np.ndarray:
    """
    Compute the dimension of a point cloud

    Parameters
    ----------
    data : np.ndarray of shape (n_clouds, n_points, n_dimensions)
        The point clouds to estimate the dimension of
    t : float, optional
        The bandwidth of the kernel. If None, it is estimated as the inverse of the mean distance to the k-nearest neighbors
    neighbors : int, optional
        The number of neighbors to consider when estimating the bandwidth
    kernel : str, optional
        The library to use for the computation. Choose between 'numpy' and 'cupy'
    progress_bar : bool, optional
        Whether to show a progress bar
    parallel : bool, optional
        Whether to run the computation in parallel
    device : int, optional
        The GPU to use when running the computation in parallel

    Returns
    -------
    np.ndarray of shape (n_clouds, n_points)
        The estimated dimension of the point clouds
    """

    n_clouds, n_points, n_dimensions = data.shape

    match kernel:
        case "numpy":
            iteratable = tqdm(data) if progress_bar else data
            if parallel:
                return np.array(joblib.Parallel(n_jobs=-1)(
                    joblib.delayed(estimate_dim)(d, t, neighbors, kernel)
                    for d in iteratable
                )) # type: ignore
            else:
                return np.array([estimate_dim(d, t, neighbors, kernel) for d in iteratable])

        case "cupy":
            if parallel:
                n = cp.cuda.runtime.getDeviceCount()
                return np.concatenate(joblib.Parallel(n_jobs=n)(
                    joblib.delayed(estimate_dims)(data_chunk, t, neighbors, kernel, progress_bar, False, i)
                    for i, data_chunk in enumerate(np.array_split(data, n))
                )) # type: ignore
            else:
                with cp.cuda.Device(device):
                    iteratable = tqdm(cp.asarray(data), position=device) if progress_bar else cp.asarray(data)
                    return np.array([estimate_dim(d, t, neighbors, kernel) for d in iteratable])



if __name__ == '__main__':

    from tqdm import tqdm
    data = np.vstack((np.load('data/AD.npy'),
                      np.load('data/Normal.npy')))

    res = estimate_dims(data, kernel='cupy', progress_bar=True, parallel=True)
    np.save('data/dimensions.npy', res)

