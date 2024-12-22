import numpy as np
from scipy.linalg import null_space

def fit_normal(mat):
    """
    find the best null-vector of a Nx3 matrix
    """
    u, s, vh = np.linalg.svd(mat, full_matrices=False)
    return vh.T[:, -1]


def fit_basis(mat):
    """
    find the best null-vector of a Nx3 matrix
    """
    u, s, vh = np.linalg.svd(mat, full_matrices=False)
    # print(vh.shape)
    return vh


def optimize_cr_given_w(w, X):
    # given w, optimize c
    # select pairs of points
    P = []
    i = 0
    while i + 1 < len(X):
        P.append((i, i + 1))
        i += 2
    # fill A,b
    A = []
    b = []
    for p in P:
        i, j = p[0], p[1]
        A_p = X[j] - X[i]
        A.append(A_p)
        b_p = (np.linalg.norm(X[i]) ** 2 - np.linalg.norm(X[j])
               ** 2 - np.dot(w, X[i]) ** 2 + np.dot(w, X[j]) ** 2) / 2
        b.append(b_p)
    A = np.array(A)
    b = np.array(b)
    n_dims = A.shape[1]

    mat = np.zeros((n_dims + 1, n_dims + 1))
    mat[:n_dims, :n_dims] = np.dot(A.T, A)
    mat[:n_dims, n_dims] = w
    mat[n_dims, :n_dims] = w.T

    y = np.zeros(n_dims + 1)
    y[:n_dims] = -np.dot(A.T, b)
    coeff = np.dot(np.linalg.pinv(mat), y)
    c = coeff[:n_dims]
    sum_ = 0
    for x in X:
        temp = np.linalg.norm(x - c) ** 2 - np.dot(x - c, w) ** 2
        if temp > 0:
            sum_ += temp
    r = np.sqrt(sum_ / len(X))
    return c, r


def fit_line(X):
    c = np.mean(X, axis=0)
    n = fit_basis(X - c)
    return c, n


def fit_plane(X):
    c = np.mean(X, axis=0)
    n = fit_normal(X - c)
    return c, n


def fit_sphere(X):
    P = []
    i = 0
    while i + 1 < len(X):
        P.append((i, i + 1))
        i += 2
    # fill A,b
    A = []
    b = []
    for p in P:
        i, j = p[0], p[1]
        A_p = X[j] - X[i]
        A.append(A_p)
        b_p = (np.linalg.norm(X[i]) ** 2 - np.linalg.norm(X[j]) ** 2) / 2
        b.append(b_p)
    A = np.array(A)
    b = np.array(b)
    c = np.dot(np.linalg.pinv(np.dot(A.T, A)), -np.dot(A.T, b))
    sum_ = 0
    for x in X:
        sum_ += np.linalg.norm(x - c) ** 2
    r = np.sqrt(sum_ / len(X))
    return c, r

def fit_cylinder(X, k):
    dimension = 3
    # find the k-neighbors

    def kNN(p, k, X_prime):
        # find k nearest neighbors
        l = []
        for x in X_prime.copy():
            dist = np.linalg.norm(x - p).copy()
            l.append((x, dist))
        l.sort(key=lambda a: a[1])
        k_arrays = np.array([a[0] for a in l[:k]]).copy()
        return k_arrays

    def cal_space(X_, dimension, col):
        # find the tangent space basis
        U, Sigma, V_T = np.linalg.svd(X_, full_matrices=False)
        # complement of the tangent pace basis
        W = V_T.T[:, 0:dimension].copy()
        complement = null_space(W.T)[:, col]
        return complement

    # tanspose and sum
    B = np.zeros([X.shape[1], X.shape[1]])
    for l in range(dimension, X.shape[1]):
        A_l = []
        for x in X.copy():
            neighbors = kNN(x, k, X).copy()
            A_l.append(cal_space(neighbors, dimension,
                                 X.shape[1] - l - 1).tolist())  # !!! notice that here np..array are row vectors
        A_l_array = np.array(A_l)
        # print(A_l_array.shape)
        B_l = np.dot(A_l_array.copy().T, A_l_array.copy())
        B = B + B_l

    # get w
    eig_value, eig_vector = np.linalg.eig(B)
    # index of the minimum eigen value
    sorted_indices = np.argsort(eig_value)
    # the eigen vector , that is w
    w = eig_vector[sorted_indices[-1]]

    sum_x = 0
    for x in X.copy():
        sum_x += np.linalg.norm(x) ** 2

    sum_dot = 0
    for x in X.copy():
        sum_dot += np.dot(x, w) ** 2

    delt = []
    X_bar = np.mean(X.copy(), axis=0)
    for x in X.copy():
        del_k = x - X_bar - np.dot(x - X_bar, w) * w
        a_k = np.linalg.norm(
            x) ** 2 - sum_x / X.shape[0] - (np.dot(x, w) ** 2 - sum_dot / X.shape[0])
        delt.append(del_k)

    delt_array = np.array(delt)

    sum_a_del = 0
    for x in X.copy():
        a_k = np.linalg.norm(
            x) ** 2 - sum_x / X.shape[0] - (np.dot(x, w) ** 2 - sum_dot / X.shape[0])
        del_k = x - X_bar - np.dot(x - X_bar, w) * w
        sum_a_del += a_k * del_k

    c = np.dot(np.linalg.pinv(np.dot(delt_array.T, delt_array)), sum_a_del)

    sum_r_2 = 0
    for x in X.copy():
        sum_ = np.abs(np.linalg.norm(x - c) ** 2 - np.dot(x - c, w) ** 2)
        sum_r_2 += sum_
    r = np.sqrt(sum_r_2 / X.shape[0])
    return c, w, r

# TODO fix this
def fit_cone(X, k):
    dimension = 3
    # find the k-neighbors

    def kNN(p, k, X_prime):
        # find k nearest neighbors
        l = []
        for x in X_prime:
            dist = np.linalg.norm(x - p)
            l.append((x, dist))
        l.sort(key=lambda a: a[1])
        return np.array([a[0] for a in l[:k]])

    def cal_space(X, dimension, col):
        # find the tangent space basis
        U, Sigma, V_T = np.linalg.svd(X, full_matrices=False)
        # complement of the tangent pace basis
        W = V_T.T[:, 0:dimension].copy()
        complement = null_space(W.T)
        return complement[:, col]

    # tanspose and sum
    B = np.zeros([X.shape[1], X.shape[1]])
    sum_right = np.zeros(X.shape[1])
    for l in range(dimension, X.shape[1]):
        E_l = []
        y_l = []
        for x in X:
            neighbors = kNN(x, k, X)
            E_l.append(cal_space(neighbors, dimension,
                       X.shape[1]-l-1).tolist())
        E_l = np.array(E_l)
        for i in range(X.shape[0]):
            y_l.append(np.dot(E_l, X[i])[i])
        y_l = np.array(y_l)
        # print(y_l.shape)
        sum_right += np.dot(E_l.T, y_l)
        # print(E_l.shape)
        B_l = np.dot(E_l.T.copy(), E_l.copy())
        B = B + B_l

    # print(sum_right)
  #  sum_L = np.zeros(X.shape[1])

    c = np.dot(np.linalg.inv(B), sum_right*(-1))

    delt = []
    for x in X.copy():
        del_k = (x-c)/np.linalg.norm(x - c)
        delt.append(del_k)

    delt = np.array(delt)

    target_matrix = np.dot(delt.T, delt)

    # get w
    eig_value, eig_vector = np.linalg.eig(target_matrix)
    # index of the minimum eigen value
    sorted_indices = np.argsort(eig_value)
    # the eigen vector , that is w
    w = eig_vector[sorted_indices[-1]]

    cos_theta = np.dot(np.mean(delt, axis=0).copy(), w)

    return c, w, cos_theta


if __name__ == '__main__':
    from geometry import HyperSphereShell
    import J_loss

    print("Testing Cone Fitting")
    vertex = np.array([1,1,1,1])
    direction = np.array([0,0,0,1])
    theta = np.pi / 6
    reduction_mat = np.array([[1,0,0,0],
                              [0,1,0,0],
                              [0,0,1,0]])

    samples = HyperSphereShell(np.zeros(3), np.tan(theta)).sample(1000)
    samples = samples @ reduction_mat
    samples += direction
    samples *= np.random.rand(1000)[:, None]
    samples += vertex

    print('Theoretical:', J_loss.J_cone(vertex, direction, np.cos(theta), samples))
    cCone, wCone, alphaCone = fit_cone(samples, 5)
    JCone = J_loss.J_cone(cCone, wCone, alphaCone, samples)
    print('Fitted:', JCone)
    print()

