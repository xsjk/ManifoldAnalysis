import numpy as np
from typing import Iterable, Literal
from sklearn.mixture import GaussianMixture
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

class GMMClusterer:

    def __init__(self, n_components: int | Iterable[int] = range(3, 10), random_state: int = None):
        """
        Initializes the GMMClusterer with the given number of components.

        Parameters
        ----------
        n_components : int or Iterable[int], optional
            The number of components to fit the GMM with.
            If an iterable, the optimal number of components will be determined.
            Defaults to range(3, 10).
        random_state : int, optional
            The random state to use for the GMM.
        """
        self.__n_components = n_components
        self.random_state = random_state

    def fit(self, X: np.ndarray, verbose: bool = False) -> 'GMMClusterer':
        '''
        Fits a Gaussian Mixture Model to the given feature
        and returns the model parameters and the labels of the feature.

        Parameters
        ----------
        n_components : int, optional
            The number of components to fit the GMM with.
            If None, the optimal number of components will be determined.
        plot : bool, optional
            Whether to plot the feature distribution and the GMM components.
        '''
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        assert X.ndim == 2, 'X must be a 2D array'

        self.X = X.copy()

        def display_info(gmm: GaussianMixture) -> None:
            print(f'n = {gmm.n_components}', # type: ignore
                  f'Score = {gmm.score(X):.2f}',
                  f'AIC = {gmm.aic(X):.2f}',
                  f'BIC = {gmm.bic(X):.2f}', sep='\t')

        def fit(n: int) -> GaussianMixture:
            return GaussianMixture(n, covariance_type='spherical', random_state=self.random_state, init_params='k-means++').fit(X)

        if isinstance(self.__n_components, Iterable):
            n_components_iter = filter(lambda n: n <= X.shape[0], self.__n_components)
            if verbose:
                print('Optimizing number of components...')
                best_aic = np.inf
                for n in n_components_iter:
                    display_info(gmm := fit(n))
                    if (aic := gmm.aic(X)) < best_aic:
                        best_aic = aic
                        self.n_components = n
                print(f'Best number of components: {self.n_components}')
            else:
                self.n_components = min(n_components_iter, key=lambda n: fit(n).aic(X))
        elif isinstance(self.__n_components, int | np.integer):
            self.n_components = self.__n_components
        else:
            raise ValueError('n_components must be an integer or an iterable of integers')

        self.gmm = fit(self.n_components)
        if verbose:
            display_info(self.gmm)

        self.means_: np.ndarray = self.gmm.means_ # type: ignore
        self.covariances_: np.ndarray = self.gmm.covariances_ # type: ignore
        self.weights_: np.ndarray = self.gmm.weights_ # type: ignore

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Fits a Gaussian Mixture Model to the given feature
        and returns the labels of the feature.

        Parameters
        ----------
        X : np.ndarray
            The feature to fit the GMM with.
            Must be 1D.

        Returns
        -------
        np.ndarray
            The component index of each sample.
            Shape: (n_samples,)
            Data type: int
        '''
        return self.fit(X).predict(X)

    def plot(self,
             fill: Literal['dist', 'rect'] | None = 'dist',
             nbins: int = 50,
             legend: bool = True,
             ylim: tuple[float, float] = (0, 1),
             cmap=plt.cm.tab10 # type: ignore
             ) -> 'GMMClusterer':
        '''
        Plots the feature distribution and the GMM components.

        Parameters
        ----------
        fill : {'dist', 'rect'}, optional
            The type of fill to use for the component distributions.
            'dist' fills the area under the distribution curve.
            'rect' fills the area between the nodes.
            Defaults to 'dist'.
        nbins : int, optional
            The number of bins to use for the histogram.
            Defaults to 50.
        legend : bool, optional
            Whether to display the legend.
            Defaults to True.
        ylim : tuple[float, float], optional
            The y-axis limits.
        cmap : str or matplotlib.colors.Colormap, optional
            The colormap to use for the component colors.
            Defaults to plt.cm.tab10.
        '''

        if self.X.shape[1] != 1:
            raise ValueError(f'Only 1D features are supported, {self.X.shape[1]}D features are not supported for plotting yet')

        gmm_order = np.argsort(self.means_[0])

        means = self.means_[0, gmm_order]
        covariances = self.covariances_[0, gmm_order]
        weights = self.weights_[0, gmm_order]
        nodes = np.empty(self.n_components - 1)
        for i in range(self.n_components - 1):
            (m1, m2), (s1, s2), (w1, w2) = means[i: i+2], covariances[i: i+2], weights[i: i+2]
            if np.allclose(s1, s2):
                nodes[i] = np.log(w2 / w1) / (m1  - m2) * s1 + (m1 + m2) / 2
            else:
                nodes[i] = (np.sqrt((2 * np.log(w2 / w1 * np.sqrt(s1 / s2)) * (s1 - s2) + (m1 - m2) ** 2) * s1 * s2) + m1 * s2 - m2 * s1) / (s2 - s1)


        norm_f = Normalize(0, self.n_components)
        x_range = (self.X.min(), self.X.max())
        x = np.linspace(*x_range, 500)


        label = self.predict(self.X)
        plt.hist(self.X, bins=nbins, alpha=0.15, color='black', density=True)
        plt.scatter(self.X, np.zeros(self.X.shape), c=label,
                    cmap=cmap, s=50, zorder=10, norm=norm_f)

        cont_dist = self.predict_proba(x)
        nodes = [x_range[0]] + nodes.tolist() + [x_range[1]]
        for i in range(self.n_components):
            mean, cov, weight = self.means_[i], self.covariances_[i], self.weights_[i]
            plt.plot(x, cont_dist[:, i], label=f'{weight:.2f} N({mean:.2f}, {np.sqrt(cov):.2f})', color=cmap(norm_f(i)))
            if fill == 'rect':
                plt.fill_between(nodes[i: i+2], 0, 1, color=cmap(norm_f(i)), alpha=0.2)
            elif fill == 'dist':
                plt.fill_between(x, 0, cont_dist[:, i], color=cmap(norm_f(i)), alpha=0.2)
            plt.vlines(nodes, 0, 1, color='black', linestyles='dashed', alpha=0.1, linewidth=1)

        if legend:
            plt.legend()
        plt.xticks(nodes)
        plt.ylim(ylim)
        plt.grid()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Predicts the component index of each sample.

        Parameters
        ----------
        X : np.ndarray
            The samples to predict the components for.
            Must be 1D.

        Returns
        -------
        np.ndarray
            The component index of each sample.
            Shape: (n_samples,)
            Data type: int
        '''

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.gmm.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        '''
        Predicts the probability of each sample belonging to each component.

        Parameters
        ----------
        X : np.ndarray
            The samples to predict the probabilities for.
            Must be 1D.

        Returns
        -------
        np.ndarray
            The probabilities of each sample belonging to each component.
            Shape: (n_samples, n_components)
            Data type: float
        '''
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.gmm.predict_proba(X)


class MDSClusterer:

    def __init__(self, hidden_dim: int = None, k: int | Literal['silhouette', 'davies_bouldin'] = 'silhouette'):
        '''
        Initializes the MDSClusterer with the given number of hidden dimensions.

        Parameters
        ----------
        hidden_dim : int, optional
            The number of hidden dimensions to reduce the data to.
            If None, use the number of samples - 1.
            Defaults to None.
        k : int or {'silhouette', 'davies_bouldin'}, optional
            The number of clusters to fit the KMeans with.

        '''
        self.__hidden_dim = hidden_dim
        self.__k = k

        self.kmeans: KMeans
        self.k: int


    def fit(self, X: np.ndarray) -> 'MDSClusterer':
        '''
        Fits the MDSClusterer to the given dissimilarity matrix (distance matrix).

        Parameters
        ----------
        X : np.ndarray
            The dissimilarity matrix (distance matrix) to fit the MDSClusterer with.
            Must be a square matrix.
        '''
        assert X.ndim == 2, 'X must be a 2D array'
        assert X.shape[0] == X.shape[1], 'X must be a square matrix'

        self.hidden_dim = X.shape[0] - 1 if self.__hidden_dim is None else self.__hidden_dim
        self.embedding = MDS(n_components=self.hidden_dim, dissimilarity='precomputed').fit_transform(X)

        if self.__k in ['silhouette', 'davies_bouldin']:
            score, agg  = {'silhouette': (silhouette_score, max), 'davies_bouldin': (davies_bouldin_score, min)}[self.__k]
            _, self.k, self.kmeans = agg((score(X, (kmeans:=KMeans(n_clusters=k)).fit_predict(self.embedding)), k, kmeans) for k in range(2, 11))
        else:
            assert isinstance(self.__k, int), 'k must be an integer'
            self.k = self.__k
            self.kmeans = KMeans(self.k).fit(self.embedding)

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Fits the MDSClusterer to the given dissimilarity matrix (distance matrix)
        and returns the labels of the data.

        Parameters
        ----------
        X : np.ndarray
            The dissimilarity matrix (distance matrix) to fit the MDSClusterer with.
        '''

        return self.fit(X).kmeans.labels_

    def get_embedding(self, X: np.ndarray) -> np.ndarray:
        # TODO: get the embedding points for data while keep the same embedding points for the training data
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        embedding = self.get_embedding(X)
        return self.kmeans.predict(embedding)

    def plot(self, X: np.ndarray = None, cmap: str | Colormap = None, random_state: int = None) -> 'MDSClusterer':
        '''
        Plots the embedding of the data by scattering the TSNE reduced 2D points with the cluster colors.

        Parameters
        ----------
        X : np.ndarray, optional
            The data to plot.
            If None, plot the training data.
            Defaults to None.
        cmap : str or matplotlib.colors.Colormap, optional
            The colormap to use for the cluster colors.
            Defaults to 'coolwarm'.
        random_state : int, optional
            The random state to use for the TSNE.
        '''
        if X is None:
            X = self.embedding
            c = self.kmeans.labels_
        else:
            X = self.get_embedding(X)
            c = self.predict(X)

        plt.scatter(*TSNE(random_state=random_state).fit_transform(X).T, c=c, cmap=cmap)
        return self


if __name__ == '__main__':
    import warnings
    from core import Core

    # turn warnings to error
    warnings.filterwarnings('error')

    patients_core = Core(
        data=np.load("data/AD.npy"),
        dataType="AD",
    )

    patients_core.fit_manifolds()
    people_components = patients_core.component_groups

    k_sums = people_components.inverse_total_curvatures
    plt.figure(figsize=(15, 5))
    print('Optimal number of components')
    for n in range(1, 7):
        plt.subplot(2, 3, n)
        GMMClusterer(n).fit(k_sums, verbose=True).plot()
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.title('Curvature Distribution')
    plt.xlabel('Curvature')
    plt.ylabel('Density')
    GMMClusterer(5).fit(k_sums).plot(fill='rect')
    plt.show()
