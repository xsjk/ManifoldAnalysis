import os
import pandas as pd
import numpy as np
import xarray as xr
from sklearn.model_selection import StratifiedKFold
from component import ComponentGroups
from core import Core, use_dim_type
from typing import Any, Literal
from sklearn.base import BaseEstimator, ClassifierMixin


class ManifoldClassifier(BaseEstimator, ClassifierMixin):

    use_dim: use_dim_type
    manifolds_cache: ComponentGroups

    def __init__(
        self,
        use_dim: use_dim_type,
        cluster_configs: list[dict[str, Any]],
        analyze_config: dict[str, Any],
        classify_config: dict[str, Any],
        fit_manifold_config: dict[str, Any] = dict(),
    ):
        self.fit_manifold_config = fit_manifold_config
        self.cluster_configs = cluster_configs
        self.analyze_config = analyze_config
        self.classify_config = classify_config
        self.use_dim = use_dim



    def fit(self, X: xr.DataArray, y: xr.DataArray) -> "ManifoldClassifier":
        assert X.ndim == 3
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]
        assert X.shape[2] == 4  # currently only support 4D data
        assert np.issubdtype(X.dtype, np.floating)
        assert np.issubdtype(y.dtype, np.integer)

        assert isinstance(X, xr.DataArray)
        assert isinstance(y, xr.DataArray)

        # split X in to classes as y
        self.classes_ = np.unique(y)
        assert len(self.classes_) == len(self.cluster_configs), \
            "Number of classes must match the number of cluster configs"

        self.cores = [Core(data=X[y == i].values, dataType=f"{i}") for i in self.classes_]
        for i, (core, cluster_config) in enumerate(zip(self.cores, self.cluster_configs)):
            if hasattr(self, "manifolds_cache"):
                manifolds_cache = getattr(self, "manifolds_cache")
                assert isinstance(manifolds_cache, ComponentGroups)
                core.component_groups = manifolds_cache[X[X.dims[0]][y == i].values]
            else:
                core.fit_manifolds(**self.fit_manifold_config)

            core.cluster_by_gmm(**cluster_config)
            core.analyze_typical(use_dim=self.use_dim, analyze_dist=self.classify_config.get("dist_type", "default"))

        self.mixed_core = sum(self.cores[1:], start=self.cores[0])
        self.mixed_nodes = np.cumsum([c.clusterer.n_components for c in self.cores])[:-1]

        return self

    def predict_proba(self, X: xr.DataArray | np.ndarray) -> np.ndarray:
        if isinstance(X, xr.DataArray):
            X = X.values
        pred_scores = self.mixed_core.classify_with_typical(
            data=X, use_dim=self.use_dim, **self.classify_config
        )
        pred_scores_split = np.split(pred_scores, self.mixed_nodes, axis=-1)
        pred_scores = np.vstack([s.max(axis=-1) for s in pred_scores_split])
        pred_scores /= pred_scores.sum(axis=0, keepdims=True)
        pred_scores = pred_scores.T
        return pred_scores

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=-1)

    def score(self, X, y):
        return (self.predict(X) == y).mean()



if __name__ == "__main__":

    from tqdm import tqdm
    import concurrent.futures
    import xarray
    from sklearn.model_selection import cross_validate
    from core import use_dim_type
    import argparse
    import warnings
    import rich
    from sklearn.exceptions import UndefinedMetricWarning

    warnings.filterwarnings(
        "ignore",
        category=UndefinedMetricWarning,
        message="Precision is ill-defined.*"
    )

    data_AD = np.load("data/AD.npy")
    data_NL = np.load("data/Normal.npy")
    protein_coding_indices = pd.read_csv(
        'Desktop/generated_data/protein_coding_ID.csv', index_col=0).index.to_numpy()

    X = np.concatenate([data_AD, data_NL], axis=0)
    y = np.concatenate([np.ones(len(data_AD), dtype=np.int8),
                        np.zeros(len(data_NL), dtype=np.int8)], axis=0)
    X = xr.DataArray(X, dims=['person', 'gene', 'section'], coords={'person': np.arange(X.shape[0])})
    y = xr.DataArray(y, dims=['person'], coords={'person': np.arange(y.shape[0])})
    # print(X.shape, y.shape)

    parser = argparse.ArgumentParser(
        description="Manifold Classifier Cross-Validation")
    parser.add_argument("--max_n1", type=int, default=10,
                        help="Maximum value for n1 group range")
    parser.add_argument("--max_n2", type=int, default=10,
                        help="Maximum value for n2 group range")
    parser.add_argument("--max_k", type=int, default=30,
                        help="Maximum value for k top range")
    parser.add_argument("--score_agg_method", type=str, default="mean",
                        help="Method to aggregate scores")
    parser.add_argument("--dist_type", type=str, choices=["default", "sphere", "plane", "hausdorff"], default="default",
                        help="Distance type for classification")
    parser.add_argument("--n_jobs", type=int, default=128,
                        help="Number of parallel")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Number of splits for cross-validation")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random state for cross-validation")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--verbose", action="store_true", default=False)

    args = parser.parse_args()


    max_n1_group_range: int = args.max_n1
    max_n2_group_range: int = args.max_n2
    max_k_top_range: int = args.max_k
    n_splits: int = args.n_splits
    random_state: int = args.random_state
    dist_type: Literal["default", "sphere", "plane", "hausdorff"] = args.dist_type
    score_agg_method: Literal["min", "mean", "max"] = args.score_agg_method
    n_jobs: int = args.n_jobs
    verbose: bool = args.verbose

    if verbose:
        rich.print(args.__dict__)

    os.makedirs(args.output_dir, exist_ok=True)

    output_file = os.path.join(
        args.output_dir,
        f"cv_scores_{n_splits}fold_{random_state}.nc"
    )

    if verbose:
        rich.print(f"Output file: {output_file}")

    n1_group_range = range(1, max_n1_group_range + 1)
    n2_group_range = range(1, max_n2_group_range + 1)
    k_top_range = range(1, max_k_top_range + 1)
    use_dim_range: list[use_dim_type] = ["4D4D", "1D1D", "1D4D"]
    scoring = [
        "accuracy",
        "average_precision",
        "balanced_accuracy",
        "f1",
        "f1_macro",
        "f1_micro",
        "f1_weighted",
        "jaccard",
        "jaccard_macro",
        "jaccard_micro",
        "jaccard_weighted",
        "matthews_corrcoef",
        "precision",
        "precision_macro",
        "precision_micro",
        "precision_weighted",
        "recall",
        "recall_macro",
        "recall_micro",
        "recall_weighted",
        "roc_auc",
    ]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=random_state)


    old_params = {'use_dim': [], 'n1': [], 'n2': [], 'k': []}

    if os.path.exists(output_file):
        if verbose:
            print("Loading existing results")
        cv_scores = xarray.open_dataarray(output_file, engine="netcdf4")
        os.remove(output_file)

        # test save
        cv_scores.to_netcdf(output_file)

        old_params = cv_scores.coords
        if verbose:
            print(old_params)

        cv_scores = cv_scores.reindex(
            use_dim=sorted(set(use_dim_range) | set(old_params['use_dim'].values)),
            n1=sorted(set(n1_group_range) | set(old_params['n1'].values)),
            n2=sorted(set(n2_group_range) | set(old_params['n2'].values)),
            k=sorted(set(k_top_range) | set(old_params['k'].values)),
        )

    else:

        cv_scores = xarray.DataArray(
            np.zeros((len(use_dim_range),
                      len(n1_group_range),
                      len(n2_group_range),
                      len(k_top_range),
                      len(scoring) + 2, n_splits)),
            dims=['use_dim', 'n1', 'n2', 'k', 'metric', 'fold'],
            coords={
                'use_dim': use_dim_range,
                'n1': n1_group_range,
                'n2': n2_group_range,
                'k': k_top_range,
                'metric': ['fit_time', 'score_time'] + [f'test_{metric}' for metric in scoring],
                'fold': range(n_splits)
            }
        )

    new_params = cv_scores.coords

    def feature_getter(cgs: ComponentGroups) -> np.ndarray:
        return np.hstack([cgs.total_curvatures[:, None], cgs.areas[:, None]])

    def typical_analyzer(cgs: ComponentGroups, people: np.ndarray, attr_mean: np.ndarray):
        total_curvature, mean_area = attr_mean
        typical_person_index = np.argmin(
            (cgs.total_curvatures[people] - total_curvature) ** 2
            + (cgs.areas[people] - mean_area) ** 2
        )
        return people[typical_person_index]

    ManifoldClassifier.manifolds_cache = Core(data=X.values, dataType="all").fit_manifolds()

    def evaluate_model(use_dim: use_dim_type, n: tuple[int, int], k: int) -> tuple[use_dim_type, tuple[int, int], int, dict[str, np.ndarray]]:

        scores = cross_validate(
            ManifoldClassifier(
                use_dim=use_dim,
                fit_manifold_config=dict(),
                cluster_configs=[dict(
                    n_group=i,
                    feature_getter=feature_getter
                ) for i in n],
                analyze_config=dict(
                    top_k=k,
                    selectable_indices=protein_coding_indices,
                    typical_analyzer=typical_analyzer
                ),
                classify_config=dict(
                    score_agg_method=score_agg_method,
                    dist_type=dist_type
                ),
            ), X, y, cv=cv, # type: ignore
            scoring=scoring
        )

        return use_dim, n, k, scores

    try:

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_params = [
                executor.submit(evaluate_model, use_dim, (n1, n2), k)
                for use_dim in new_params['use_dim'].values
                for n1 in new_params['n1'].values
                for n2 in new_params['n2'].values
                for k in new_params['k'].values
                if not (
                    use_dim in old_params['use_dim'] and
                    n1 in old_params['n1'] and
                    n2 in old_params['n2'] and
                    k in old_params['k']
                ) or cv_scores.loc[use_dim, n1, n2, k, 'fit_time'].isnull().any()
            ]
            progress_bar = tqdm(total=len(future_to_params), desc="Evaluating models" if verbose else f"Evaluating models to {output_file}")
            for future in concurrent.futures.as_completed(future_to_params):
                use_dim, n, k, scores = future.result()
                for metric, score in scores.items():
                    cv_scores.loc[use_dim, n[0], n[1], k, metric] = score

                progress_bar.update(1)

            # print(f"{n = :d}, {k = :d}, scores = {scores.mean():.3f} ± {scores.std():.3f}")
    except KeyboardInterrupt:
        pass

    finally:

        try:
            cv_scores.to_netcdf(output_file)
            if verbose:
                print(f"Results saved to {output_file}")
        except PermissionError:
            print("Permission denied to write file")
            pass
