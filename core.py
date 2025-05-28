import pickle
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Literal

import numpy as np
from joblib import Parallel, delayed
from tqdm import trange

import geometry
import utils
from clusterer import GMMClusterer
from component import (
    ComponentGroup,
    ComponentGroups,
    Shape,
)
from grouper import GroupResult

eps = np.finfo(np.float32).eps


@dataclass
class AnalysisData:
    people: np.ndarray
    distance_means: list[np.ndarray]
    distance_stds: list[np.ndarray]
    indices: list[np.ndarray]
    num_people: int

    @property
    def num_groups(self) -> int:
        return len(self.people)

    def __add__(self, other: "AnalysisData") -> "AnalysisData":
        return AnalysisData(
            people=np.hstack([self.people, other.people + self.num_people]),
            distance_means=self.distance_means + other.distance_means,
            distance_stds=self.distance_stds + other.distance_stds,
            indices=self.indices + other.indices,
            num_people=self.num_people + other.num_people,
        )


@dataclass
class ClusterResult:
    group_result: GroupResult
    cluster_means: np.ndarray

    def __add__(self, other: "ClusterResult") -> "ClusterResult":
        return ClusterResult(
            group_result=self.group_result + other.group_result,
            cluster_means=np.append(self.cluster_means, other.cluster_means),
        )


use_dim_type = Literal["1D1D", "1D4D", "4D4D"]
analyze_dist_type = Literal["default", "sphere", "plane", "hausdorff"]
typical_analyzer_type = Callable[
    [
        ComponentGroups,
        np.ndarray[tuple[int], np.dtype[np.integer]],
        np.ndarray[tuple[int], np.dtype[np.floating]],
    ],
    np.integer | int,
]

feature_getter_type = Callable[
    [ComponentGroups],
    np.ndarray[tuple[int, int], np.dtype[np.floating]],
]


score_agg_type = Literal["max", "mean", "median", "min"]


class Core:
    data: np.ndarray  # shape: (n_people, n_samples, 4)
    dataType: np.ndarray  # data type for each person

    component_groups: ComponentGroups
    resampled_data: np.ndarray
    dist_sheet: np.ndarray

    clusterer: GMMClusterer
    cluster_result: ClusterResult

    feature_getter: feature_getter_type
    typical_analyzer: typical_analyzer_type

    sphere_typical_data: dict[use_dim_type, AnalysisData]
    plane_typical_data: dict[use_dim_type, AnalysisData]
    hausdorff_typical_data: dict[use_dim_type, AnalysisData]

    pred_scores: np.ndarray
    pred_label: np.ndarray
    true_label: np.ndarray

    def __init__(
        self,
        data: np.ndarray,
        dataType: str | np.ndarray,
        component_groups: ComponentGroups = None,
    ):
        self.data = data
        if isinstance(dataType, str):
            self.dataType = np.array([dataType] * len(data), dtype=f"U{len(dataType)}")
        elif isinstance(dataType, np.ndarray):
            self.dataType = dataType
        else:
            raise ValueError(f"Invalid type for dataType: {type(dataType)}")
        if component_groups is not None:
            self.component_groups = component_groups

    def fit_manifolds(
        self,
        split_config: dict = dict(constraint=200),
        analyze_config: dict = dict(),
        n_jobs: int = 16,
        progress_bar: bool = False,
    ) -> ComponentGroups:
        self.component_groups = ComponentGroups(
            Parallel(n_jobs=n_jobs)(
                delayed(ComponentGroup.fit)(
                    data=self.data[i],
                    split_config=split_config,
                    analyze_config=analyze_config,
                    component_config=dict(
                        personID=i,
                        dataType=self.dataType[i],
                        data=self.data[i],
                    ),
                )
                for i in (trange if progress_bar else range)(len(self.data))
            )
        )

        return self.component_groups

    def load_manifolds(self, dir_path: str):
        self.component_groups = pickle.load(open(dir_path, "rb"))

    def save_manifolds(self, file_path: str):
        pickle.dump(self.component_groups, open(file_path, "wb"))

    def cluster_by_gmm(
        self,
        n_group: int,
        feature_getter: feature_getter_type,
        random_state: int = 42,
    ):
        """
        Cluster the components by GMM

        Parameters
        ----------
        n_group : int
            the number of groups
        feature_getter : Callable[[ComponentGroups], np.ndarray]
            the function to get the features for clustering
            it should take the component_groups as input and return the features as a 2D array
        random_state : int, optional
            the random state, by default 42
        """
        self.clusterer = GMMClusterer(n_components=n_group, random_state=random_state)

        self.feature_getter = feature_getter

        features = self.feature_getter(self.component_groups)

        self.clusterer.fit(features)
        self.cluster_result = ClusterResult(
            group_result=GroupResult(self.clusterer.predict(features)),
            cluster_means=self.clusterer.means_,
        )

    @property
    def n_group(self) -> int:
        return self.clusterer.n_components

    @property
    def group_result(self) -> GroupResult:
        return self.cluster_result.group_result

    dist_f_map: dict[str, Callable] = {
        "sphere": geometry.distance_to_spheres,
        "plane": geometry.distance_to_planes,
        "hausdorff": geometry.distance_to_pointsets,
    }

    def _analyze_impl(
        self,
        use_dim: use_dim_type,
        analyze_dist: analyze_dist_type,
        people_indices: np.ndarray,
        selectable_indices: np.ndarray = None,
        top_k: int = 20,
    ) -> AnalysisData:
        distance_means = []
        distance_stds = []
        indices = []

        dist_f = self.use_dim_decorator(use_dim)(self.dist_f_map[analyze_dist])

        if not hasattr(self, "cluster_result"):
            raise ValueError("The cluster_result has not been assigned yet")

        for people, mean, typical_person in zip(
            self.group_result.group2people,
            self.cluster_result.cluster_means,
            people_indices,
        ):
            assert mean.ndim == 1

            typical_components: ComponentGroup = self.component_groups[typical_person]

            match analyze_dist:
                case "sphere":
                    typical_components = typical_components.filter(lambda c: c.shape == Shape.SPHERE)
                case "plane":
                    typical_components = typical_components.filter(lambda c: c.shape == Shape.PLANE)

            n_components = len(typical_components)

            if n_components == 0:
                distance_means.append(np.empty((0, top_k)))
                distance_stds.append(np.empty((0, top_k)))
                indices.append(np.empty((0, top_k), dtype=int))
                continue

            selected_indices = typical_components.get_typical_indices(top_k, selectable_indices)
            if selectable_indices is not None:
                assert np.all(np.isin(selected_indices, selectable_indices))
            assert selected_indices.shape == (n_components, top_k)

            n_people = len(people)

            selected_data = self.data[people][:, selected_indices, :]
            assert selected_data.shape == (n_people, n_components, top_k, 4)

            match analyze_dist:
                case "sphere":
                    centers = typical_components.param1
                    radii = typical_components.param2
                    assert centers.shape == (n_components, 4)
                    assert radii.shape == (n_components,)
                    args = [centers, radii]
                case "plane":
                    x0s = typical_components.param1
                    normals = typical_components.param2
                    assert x0s.shape == (n_components, 4)
                    assert normals.shape == (n_components, 4)
                    args = [x0s, normals]
                case "hausdorff":
                    typical_person_data = self.data[typical_person, selected_indices, :]
                    assert typical_person_data.shape == (n_components, top_k, 4)
                    args = [typical_person_data]
                case _:
                    raise ValueError(f"Invalid value for analyze_dist: {analyze_dist}")

            distance = dist_f(selected_data, *args)
            if analyze_dist == "hausdorff":
                assert distance.shape == (n_people, n_components)
            else:  # sphere or plane
                assert distance.shape == (n_people, n_components, top_k)

            means: np.ndarray = distance.mean(axis=0)
            stds: np.ndarray = distance.std(axis=0)
            if analyze_dist == "hausdorff":
                assert means.shape == (n_components,)
                assert stds.shape == (n_components,)
            else:  # sphere or plane
                assert means.shape == (n_components, top_k)
                assert stds.shape == (n_components, top_k)

            distance_means.append(means)
            distance_stds.append(stds)
            indices.append(selected_indices)

        return AnalysisData(
            people=people_indices,
            distance_means=distance_means,
            distance_stds=distance_stds,
            indices=indices,
            num_people=self.group_result.n_people,
        )

    def analyze_typical_person(self, typical_analyzer: typical_analyzer_type = None) -> np.ndarray[tuple[int], np.dtype[np.integer]]:
        """
        Analyze the typical person for each group

        Parameters
        ----------
        typical_analyzer : Callable[[ComponentGroups, np.ndarray, np.ndarray], np.integer | int], optional
            the function should take the component_groups, people indices, and the attr of cluster mean as input and return the typical person index, by default None
            If None, the function to get the typical person, by default it gets the person with the attribute ( use self.feature_getter ) closest to the cluster mean

        Returns
        -------
        np.ndarray
            the typical person for each group
        """

        typical = []

        if not hasattr(self, "cluster_result"):
            raise ValueError("The cluster_result has not been assigned yet")

        for people, mean in zip(
            self.group_result.group2people,
            self.cluster_result.cluster_means,
        ):
            assert mean.ndim == 1

            people = np.array(people)

            typical_person: np.integer | int
            if typical_analyzer is None:
                assert hasattr(self, "feature_getter"), "The feature_getter has not been assigned yet, make sure you have run the cluster_by_gmm method"
                feature = self.feature_getter(self.component_groups)[people]
                assert feature.ndim == 2
                assert feature.shape[1] == mean.shape[0]
                distance = np.linalg.norm(feature - mean, axis=-1)
                typical_person = people[distance.argmin()]  # type: ignore
            else:
                typical_person = typical_analyzer(self.component_groups, people, mean)

            typical.append(typical_person)

        return np.array(typical)

    def analyze_typical(
        self,
        use_dim: use_dim_type,
        analyze_dist: analyze_dist_type,
        typical_analyzer: typical_analyzer_type = None,
        **kwargs,
    ):
        """
        Analyze the typical components for each group

        Parameters
        ----------
        use_dim : use_dim_type
            the dimension to use, by default '4D4D'
            `1D1D` means 1D data and 1D center
            `1D4D` means 1D data and 4D center
            `4D4D` means 4D data and 4D center
        analyze_dist : analyze_dist
            the distance type to use, by default "default"
            `default` means to analyze both sphere and plane
            `sphere` means to analyze the typical sphere
            `plane` means to analyze the typical plane
            `hausdorff` means to analyze the typical hausdorff
        typical_analyzer : Callable[[ComponentGroups, np.ndarray, np.ndarray], np.integer | int], optional
            the function should take the component_groups, people indices, and the attr of cluster mean as input and return the typical person index, by default None
            If None, the function to get the typical person, by default it gets the person with the attribute ( use self.feature_getter ) closest to the cluster mean
        """

        if analyze_dist == "default":
            self.analyze_typical(use_dim, "sphere", typical_analyzer, **kwargs)
            self.analyze_typical(use_dim, "plane", typical_analyzer, **kwargs)
            return

        typical_people = self.analyze_typical_person(typical_analyzer)

        attr_name = f"{analyze_dist}_typical_data"
        if not hasattr(self, attr_name):
            setattr(self, attr_name, {})
        getattr(self, attr_name)[use_dim] = self._analyze_impl(use_dim, analyze_dist, typical_people, **kwargs)

    def _classify_impl(
        self,
        use_dim: use_dim_type,
        analyze_dist: analyze_dist_type,
        analysis_data: AnalysisData,
        data: np.ndarray = None,
        score_agg_method: Literal["mean", "max", "min"] = "mean",
        verbose: bool = False,
        normalize_score: bool = True,
    ) -> np.ndarray:
        """
        Classify the data with typical data

        Parameters
        ----------
        use_dim: use_dim_type
            The dimension of the data to be classified
        data: np.ndarray
            The data to be classified, if None, self.data will be used
        score_agg_method: Literal["mean", "max", "min"]
            The method to aggregate the scores of different components
        verbose: bool
            whether to print the information, by default False

        Returns
        -------
        pred_scores: np.ndarray
        """

        if data is None:
            data = self.data

        # evaluate the classification score for every person in all groups
        n_people, n_proteins, n_dim = data.shape
        n_group = self.group_result.n_group

        pred_scores = np.empty((n_people, n_group))
        assert pred_scores.shape == (n_people, n_group)

        assert len(analysis_data.people) == n_group
        assert len(analysis_data.distance_means) == n_group
        assert len(analysis_data.distance_stds) == n_group
        assert len(analysis_data.indices) == n_group

        dist_f = self.use_dim_decorator(use_dim)(self.dist_f_map[analyze_dist])

        for group_index, (
            person_idx,
            selected_indices,
            means,
            stds,
        ) in enumerate(
            zip(
                analysis_data.people,
                analysis_data.indices,
                analysis_data.distance_means,
                analysis_data.distance_stds,
            )
        ):
            n_components, n_points_per_component = selected_indices.shape

            if n_components == 0:
                if verbose:
                    print(f"Warning: no components for group {group_index}")
                pred_scores[:, group_index] = eps
                continue

            person_data = self.data[person_idx, selected_indices]
            assert person_data.shape == (n_components, n_points_per_component, 4)

            components: ComponentGroup = self.component_groups[person_idx]

            match analyze_dist:
                case "sphere":
                    components = components.filter(lambda c: c.shape == Shape.SPHERE)
                case "plane":
                    components = components.filter(lambda c: c.shape == Shape.PLANE)

            assert len(components) == n_components

            if analyze_dist == "hausdorff":
                assert means.shape == (n_components,)
                assert stds.shape == (n_components,)
            else:
                assert means.shape == (n_components, n_points_per_component)
                assert stds.shape == (n_components, n_points_per_component)

            match analyze_dist:
                case "sphere":
                    centers = components.param1
                    radius = components.param2
                    assert centers.shape == (n_components, 4)
                    assert radius.shape == (n_components,)
                    args = [centers, radius]
                case "plane":
                    x0s = components.param1
                    normals = components.param2
                    assert x0s.shape == (n_components, 4)
                    assert normals.shape == (n_components, 4)
                    args = [x0s, normals]
                case "hausdorff":
                    args = [person_data]
                case _:
                    raise ValueError(f"Invalid value for analyze_dist: {analyze_dist}")

            # calculate scores
            group_data = data[:, selected_indices, :]
            n_people, n_components, n_points_per_component, n_dim = group_data.shape
            assert n_dim == 4
            assert group_data.shape == (n_people, n_components, n_points_per_component, n_dim)

            distances = dist_f(group_data, *args)

            if analyze_dist == "hausdorff":
                assert distances.shape == (
                    n_people,
                    n_components,
                )
            else:  # sphere or plane
                assert distances.shape == (n_people, n_components, n_points_per_component)

            scores: np.ndarray = np.exp(-(((distances - means) / (stds + eps)) ** 2))
            if analyze_dist == "hausdorff":
                assert scores.shape == (n_people, n_components)
            else:  # sphere or plane
                assert scores.shape == (n_people, n_components, n_points_per_component)

            agg_axis = (1, 2) if analyze_dist != "hausdorff" else 1

            match score_agg_method:
                case "max":
                    scores = np.amax(scores, axis=agg_axis)
                case "mean":
                    scores = np.mean(scores, axis=agg_axis)
                case "min":
                    scores = np.amin(scores, axis=agg_axis)
                case _:
                    raise ValueError(f"Invalid value for agg_method: {score_agg_method}")
            assert scores.shape == (n_people,)

            pred_scores[:, group_index] = scores

        if normalize_score:
            pred_scores /= pred_scores.sum(axis=1, keepdims=True)
        pred_label: np.ndarray = pred_scores.argmax(axis=1)
        true_label: np.ndarray = np.array(self.group_result.person2group)

        self.pred_scores = pred_scores
        self.pred_label = pred_label
        self.true_label = true_label

        return pred_scores

    def classify_with_typical(
        self,
        use_dim: use_dim_type,
        data: np.ndarray = None,
        score_agg_method: Literal["mean", "max", "min"] = "mean",
        dist_type: analyze_dist_type = "default",
        verbose: bool = False,
        normalize_score: bool = True,
    ):
        """
        Classify the data with typical data

        Parameters
        ----------
        use_dim: use_dim_type
            The dimension of the data to be classified
        data: np.ndarray
            The data to be classified, if None, self.data will be used
        score_agg_method: Literal["mean", "max", "min"]
            The method to aggregate the scores of different components
        dist_type: analyze_dist_type
            The distance type to use
        verbose: bool
            whether to print the information, by default False

        Returns
        -------
        pred_scores: np.ndarray
        """

        if dist_type == "default":
            score_sphere = self.classify_with_typical(
                use_dim=use_dim,
                data=data,
                score_agg_method=score_agg_method,
                dist_type="sphere",
                verbose=verbose,
                normalize_score=False,
            )

            score_plane = self.classify_with_typical(
                use_dim=use_dim,
                data=data,
                score_agg_method=score_agg_method,
                dist_type="plane",
                verbose=verbose,
                normalize_score=False,
            )

            match score_agg_method:
                case "mean":
                    score = score_sphere * score_plane
                case "max":
                    score = np.maximum(score_sphere, score_plane)
                case "min":
                    score = np.minimum(score_sphere, score_plane)
                case _:
                    raise ValueError(f"Invalid value for score_agg_method: {score_agg_method}")

            if normalize_score:
                score /= score.sum(axis=1, keepdims=True)

        else:
            analysis_data = getattr(self, f"{dist_type}_typical_data")[use_dim]

            score = self._classify_impl(
                use_dim=use_dim,
                analyze_dist=dist_type,
                analysis_data=analysis_data,
                data=data,
                score_agg_method=score_agg_method,
                verbose=verbose,
                normalize_score=normalize_score,
            )

        self.pred_scores = score
        self.pred_label = score.argmax(axis=1)
        self.true_label = np.array(self.group_result.person2group)

        return score

    @staticmethod
    def use_dim_decorator(use_dim: use_dim_type) -> Callable[[Callable[..., np.ndarray]], Callable[..., np.ndarray]]:
        match use_dim:
            case "4D4D":
                return lambda f: f
            case "1D1D":
                return lambda f: wraps(f)(lambda data, *args: f(*(np.mean(arr, axis=-1, keepdims=True) if arr.shape[-1] == 4 else arr for arr in (data,) + args)))
            case "1D4D":
                return lambda f: wraps(f)(lambda data, *args: f(np.broadcast_to(np.mean(data, axis=-1, keepdims=True), data.shape), *args))

    def show_classification_result(self):
        utils.show_classification_result(self.true_label, self.pred_scores)

    def split(self, n_splits: int = 10, random_state: int = 42) -> list["Core"]:
        """
        Split the data into n_splits parts

        Parameters
        ----------
        n_splits : int, optional
            the number of splits, by default 10
        random_state : int, optional
            the random state, by default 42

        Returns
        -------
        list[Core]
            the splitted data
        """

        np.random.seed(random_state)
        indices = np.random.permutation(len(self.data))
        split_indices = np.array_split(indices, n_splits)

        cores = [
            Core(
                data=self.data[split_index],
                dataType=self.dataType[split_index],
                component_groups=self.component_groups[split_index],
            )
            for split_index in split_indices
        ]

        # for attr in ('clusterer', 'cluster_result'):
        #     if hasattr(self, attr):
        #         for core in cores:
        #             setattr(core, attr, getattr(self, attr))

        return cores

    def __add__(self, other: "Core") -> "Core":
        obj = object.__new__(Core)
        obj.data = np.vstack([self.data, other.data])
        obj.component_groups = self.component_groups + other.component_groups
        obj.cluster_result = self.cluster_result + other.cluster_result
        obj.dataType = np.hstack([self.dataType, other.dataType])
        for attr in ("sphere_typical_data", "plane_typical_data", "hausdorff_typical_data"):
            if hasattr(self, attr) and hasattr(other, attr):
                setattr(obj, attr, {})
                for key in self.__annotations__[attr].__args__[0].__args__:
                    if key in getattr(self, attr) and key in getattr(other, attr):
                        getattr(obj, attr)[key] = getattr(self, attr)[key] + getattr(other, attr)[key]
        return obj


if __name__ == "__main__":
    patients_core = Core(
        data=np.load("data/AD.npy"),
        dataType="AD",
    )
    patients_core.fit_manifolds()
    patients_core.component_groups.set_distance_cache_path(
        "data/patients_dist_sheet.npy",
    )
    patients_core.cluster_by_gmm(
        n_group=4,
        feature_getter=lambda cgs: cgs.inverse_total_curvatures[:, None],
    )
    patients_core.analyze_typical(
        use_dim="4D4D",
        analyze_dist="sphere",
    )
    patients_core.classify_with_typical(
        use_dim="4D4D",
        dist_type="sphere",
    )
    patients_core.show_classification_result()
