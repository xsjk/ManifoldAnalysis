import numpy as np
from typing import Literal, Callable
from clusterer import GMMClusterer
import geometry
from grouper import GroupResult
from tqdm import trange
from joblib import Parallel, delayed
from functools import wraps
from component import (
    ComponentGroup,
    ComponentGroups,
    Shape,
)
from dataclasses import dataclass
import utils
import pickle


eps = np.finfo(np.float32).eps

@dataclass
class TypicalAnalysisData:
    people: np.ndarray
    distance_means: list[np.ndarray]
    distance_stds: list[np.ndarray]
    indices: list[np.ndarray]
    num_people: int

    @property
    def num_groups(self) -> int:
        return len(self.people)

    def __add__(self, other: "TypicalAnalysisData") -> "TypicalAnalysisData":
        return TypicalAnalysisData(
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
typical_analyzer_type = Callable[[
    ComponentGroups,
    np.ndarray[tuple[int], np.dtype[np.integer]],
    np.ndarray[tuple[int], np.dtype[np.floating]]
], np.integer | int]

feature_getter_type = Callable[[ComponentGroups], np.ndarray[tuple[int, int], np.dtype[np.floating]]]


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

    sphere_typical_data: dict[use_dim_type, TypicalAnalysisData]
    plane_typical_data: dict[use_dim_type, TypicalAnalysisData]
    hausdorff_typical_data: dict[use_dim_type, TypicalAnalysisData]

    pred_scores: np.ndarray
    pred_label: np.ndarray
    true_label: np.ndarray

    def __init__(
        self,
        data: np.ndarray,
        dataType: str | np.ndarray,
        component_groups: ComponentGroups | None = None,
    ):
        self.data = data
        if isinstance(dataType, str):
            self.dataType = np.array(
                [dataType] * len(data), dtype=f"U{len(dataType)}")
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
                        personID=i, dataType=self.dataType[i], data=self.data[i]
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
        self.clusterer = GMMClusterer(
            n_components=n_group, random_state=random_state)

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

    def analyze_typical_sphere(
        self,
        use_dim: use_dim_type,
        typical_analyzer: typical_analyzer_type | None = None,
        selectable_indices: np.ndarray | None = None,
        top_k: int = 20,
        verbose: bool = False,
    ):
        """
        Analyze the typical components for each group

        Parameters
        ----------
        use_dim : use_dim_type, optional
            the dimension to use, by default '4D4D'
            `1D1D` means 1D data and 1D center
            `1D4D` means 1D data and 4D center
            `4D4D` means 4D data and 4D center
        typical_analyzer : Callable[[ComponentGroups, np.ndarray, np.ndarray], np.integer | int], optional
            the function should take the component_groups, people indices, and the attr of cluster mean as input and return the typical person index, by default None
            If None, the function to get the typical person, by default it gets the person with the attribute ( use self.feature_getter ) closest to the cluster mean
        selectable_indices : np.ndarray, optional
            the indices that can be selected, by default None, which means all the indices can be selected
        top_k : int, optional
            the number of components to select, by default 20
        verbose : bool, optional
            whether to print the information, by default False
        """

        typical_people = self.analyze_typical_person(typical_analyzer)
        typical_distance_means = []
        typical_distance_stds = []
        typical_indices = []

        dist_f = self.use_dim_decorator(use_dim)(geometry.distance_to_spheres)

        if not hasattr(self, "cluster_result"):
            raise ValueError("The cluster_result has not been assigned yet")

        for people, mean, typical_person in zip(
            self.group_result.group2people,
            self.cluster_result.cluster_means,
            typical_people
        ):
            assert mean.ndim == 1

            typical_components: ComponentGroup = self.component_groups[typical_person]
            sphere_components: ComponentGroup = typical_components.filter(lambda c: c.shape == Shape.SPHERE)

            n_spheres = len(sphere_components)
            if n_spheres == 0:
                typical_distance_means.append(np.empty((0, top_k)))
                typical_distance_stds.append(np.empty((0, top_k)))
                typical_indices.append(np.empty((0, top_k), dtype=int))
                continue

            selected_indices = sphere_components.get_typical_indices(top_k, selectable_indices)
            assert selected_indices.shape == (n_spheres, top_k)

            if verbose:
                print(f"{n_spheres} / {len(typical_components)} of the components are spheres")
                print("number of selected indices:", selected_indices.shape)
                print()

            centers = sphere_components.param1
            radii = sphere_components.param2
            assert centers.shape == (n_spheres, 4)
            assert radii.shape == (n_spheres,)

            n_people = len(people)

            selected_data = self.data[people][:, selected_indices, :]
            assert selected_data.shape == (n_people, n_spheres, top_k, 4)

            distance = dist_f(selected_data, centers, radii)
            assert distance.shape == (n_people, n_spheres, top_k)

            means: np.ndarray = distance.mean(axis=0)
            stds: np.ndarray = distance.std(axis=0)
            assert means.shape == (n_spheres, top_k)
            assert stds.shape == (n_spheres, top_k)

            typical_distance_means.append(means)
            typical_distance_stds.append(stds)
            typical_indices.append(selected_indices)

        if not hasattr(self, "sphere_typical_data"):
            self.sphere_typical_data = {}

        self.sphere_typical_data[use_dim] = TypicalAnalysisData(
            people=typical_people,
            distance_means=typical_distance_means,
            distance_stds=typical_distance_stds,
            indices=typical_indices,
            num_people=self.group_result.n_people,
        )

    def analyze_typical_plane(
        self,
        use_dim: use_dim_type,
        typical_analyzer: typical_analyzer_type | None = None,
        selectable_indices: np.ndarray | None = None,
        top_k: int = 20,
        verbose: bool = False,
    ):
        """
        Analyze the typical components for each group

        Parameters
        ----------
        use_dim : use_dim_type, optional
            the dimension to use, by default '4D4D'
            `1D1D` means 1D data and 1D center
            `1D4D` means 1D data and 4D center
            `4D4D` means 4D data and 4D center
        typical_analyzer : Callable[[ComponentGroups, np.ndarray, np.ndarray], np.integer | int], optional
            the function should take the component_groups, people indices, and the attr of cluster mean as input and return the typical person index, by default None
            If None, the function to get the typical person, by default it gets the person with the attribute ( use self.feature_getter ) closest to the cluster mean
        selectable_indices : np.ndarray, optional
            the indices that can be selected, by default None, which means all the indices can be selected
        top_k : int, optional
            the number of components to select, by default 20
        verbose : bool, optional
            whether to print the information, by default False
        """

        typical_people = self.analyze_typical_person(typical_analyzer)
        typical_distance_means = []
        typical_distance_stds = []
        typical_indices = []

        dist_f = self.use_dim_decorator(use_dim)(geometry.distance_to_planes)

        if not hasattr(self, "cluster_result"):
            raise ValueError("The cluster_result has not been assigned yet")

        for people, mean, typical_person in zip(
            self.group_result.group2people,
            self.cluster_result.cluster_means,
            typical_people
        ):
            assert mean.ndim == 1

            typical_components: ComponentGroup = self.component_groups[typical_person]
            plane_components = typical_components.filter(lambda c: c.shape == Shape.PLANE)

            n_planes = len(plane_components)
            if n_planes == 0:
                typical_distance_means.append(np.empty((0, top_k)))
                typical_distance_stds.append(np.empty((0, top_k)))
                typical_indices.append(np.empty((0, top_k), dtype=int))
                continue

            selected_indices = plane_components.get_typical_indices(top_k, selectable_indices)
            assert selected_indices.shape == (n_planes, top_k)

            if verbose:
                print(f"{n_planes} / {len(typical_components)} of the components are planes")
                print("number of selected indices:", selected_indices.shape)
                print()

            x0s = plane_components.param1
            normals = plane_components.param2
            assert x0s.shape == (n_planes, 4)
            assert normals.shape == (n_planes, 4)

            n_people = len(people)

            selected_data = self.data[people][:, selected_indices, :]
            assert selected_data.shape == (n_people, n_planes, top_k, 4)

            distance = dist_f(selected_data, x0s, normals)
            assert distance.shape == (n_people, n_planes, top_k)

            means: np.ndarray = distance.mean(axis=0)
            stds: np.ndarray = distance.std(axis=0)
            assert means.shape == (n_planes, top_k)
            assert stds.shape == (n_planes, top_k)

            typical_distance_means.append(means)
            typical_distance_stds.append(stds)
            typical_indices.append(selected_indices)

        if not hasattr(self, "plane_typical_data"):
            self.plane_typical_data = {}

        self.plane_typical_data[use_dim] = TypicalAnalysisData(
            people=typical_people,
            distance_means=typical_distance_means,
            distance_stds=typical_distance_stds,
            indices=typical_indices,
            num_people=self.group_result.n_people,
        )

    def analyze_typical_hausdorff(
        self,
        use_dim: use_dim_type,
        typical_analyzer: typical_analyzer_type | None = None,
        selectable_indices: np.ndarray | None = None,
        top_k: int = 20,
        verbose: bool = False,
    ):

        """
        Analyze the typical components for each group

        Parameters
        ----------
        use_dim : use_dim_type, optional
            the dimension to use, by default '4D4D'
            `1D1D` means 1D data and 1D center
            `1D4D` means 1D data and 4D center
            `4D4D` means 4D data and 4D center
        typical_analyzer : Callable[[ComponentGroups, np.ndarray, np.ndarray], np.integer | int], optional
            the function should take the component_groups, people indices, and the attr of cluster mean as input and return the typical person index, by default None
            If None, the function to get the typical person, by default it gets the person with the attribute ( use self.feature_getter ) closest to the cluster mean
        selectable_indices : np.ndarray, optional
            the indices that can be selected, by default None, which means all the indices can be selected
        top_k : int, optional
            the number of components to select, by default 20
        verbose : bool, optional
            whether to print the information, by default False
        """

        typical_people = self.analyze_typical_person(typical_analyzer)
        typical_distance_means = []
        typical_distance_stds = []
        typical_indices = []

        dist_f = self.use_dim_decorator(use_dim)(geometry.distance_to_pointsets)

        if not hasattr(self, "cluster_result"):
            raise ValueError("The cluster_result has not been assigned yet")

        for people, mean, typical_person in zip(
            self.group_result.group2people,
            self.cluster_result.cluster_means,
            typical_people
        ):
            assert mean.ndim == 1

            typical_components: ComponentGroup = self.component_groups[typical_person]

            n_components = len(typical_components)

            selected_indices = typical_components.get_typical_indices(top_k, selectable_indices)
            assert selected_indices.shape == (n_components, top_k)

            n_people = len(people)

            selected_data = self.data[people][:, selected_indices, :]
            assert selected_data.shape == (n_people, n_components, top_k, 4)

            typical_person_data = self.data[typical_person, selected_indices, :]
            assert typical_person_data.shape == (n_components, top_k, 4)

            distance = dist_f(selected_data, typical_person_data)
            assert distance.shape == (n_people, n_components, )

            means: np.ndarray = distance.mean(axis=0)
            stds: np.ndarray = distance.std(axis=0)
            assert means.shape == (n_components, )
            assert stds.shape == (n_components, )

            typical_distance_means.append(means)
            typical_distance_stds.append(stds)
            typical_indices.append(selected_indices)

        if not hasattr(self, "hausdorff_typical_data"):
            self.hausdorff_typical_data = {}

        self.hausdorff_typical_data[use_dim] = TypicalAnalysisData(
            people=typical_people,
            distance_means=typical_distance_means,
            distance_stds=typical_distance_stds,
            indices=typical_indices,
            num_people=self.group_result.n_people,
        )

    def analyze_typical_person(self, typical_analyzer: typical_analyzer_type | None = None) -> np.ndarray[tuple[int], np.dtype[np.integer]]:
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
                assert hasattr(
                    self, 'feature_getter'), "The feature_getter has not been assigned yet, make sure you have run the cluster_by_gmm method"
                feature = self.feature_getter(self.component_groups)[people]
                assert feature.ndim == 2
                assert feature.shape[1] == mean.shape[0]
                distance = np.linalg.norm(feature - mean, axis=-1)
                typical_person = people[distance.argmin()]  # type: ignore
            else:
                typical_person = typical_analyzer(
                    self.component_groups,
                    people,
                    mean
                )

            typical.append(typical_person)

        return np.array(typical)


    def analyze_typical(
        self,
        use_dim: use_dim_type,
        analyze_dist: analyze_dist_type,
        **kwargs
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
        """
        match analyze_dist:
            case "default":
                self.analyze_typical_sphere(use_dim=use_dim, **kwargs)
                self.analyze_typical_plane(use_dim=use_dim, **kwargs)
            case "sphere":
                self.analyze_typical_sphere(use_dim=use_dim, **kwargs)
            case "plane":
                self.analyze_typical_plane(use_dim=use_dim, **kwargs)
            case "hausdorff":
                self.analyze_typical_hausdorff(use_dim=use_dim, **kwargs)

    def classify_with_typical_sphere(
        self,
        use_dim: use_dim_type,
        data: np.ndarray | None = None,
        score_agg_method: Literal["mean", "max", "min"] = "mean",
        verbose: bool = False,
        normalize_score: bool = True,
    ) -> np.ndarray:
        '''
        Classify the data with typical data

        Parameters
        ----------
        use_dim: use_dim_type
            The dimension of the data to be classified
        data: np.ndarray | None
            The data to be classified, if None, self.data will be used
        score_agg_method: Literal["mean", "max", "min"]
            The method to aggregate the scores of different components
        verbose: bool
            whether to print the information, by default False

        Returns
        -------
        pred_scores: np.ndarray
        '''

        if data is None:
            data = self.data

        # evaluate the classification score for every person in all groups
        n_people, n_proteins, n_dim = data.shape
        n_group = self.group_result.n_group

        pred_scores = np.empty((n_people, n_group))
        assert pred_scores.shape == (n_people, n_group)

        if use_dim not in self.sphere_typical_data:
            raise ValueError(f"Typical data for {use_dim} has not been analyzed yet")
        typical_data = self.sphere_typical_data[use_dim]

        typical = typical_data.people
        typical_distance_means = typical_data.distance_means
        typical_distance_stds = typical_data.distance_stds
        typical_indices = typical_data.indices
        assert len(typical) == n_group
        assert len(typical_distance_means) == n_group
        assert len(typical_distance_stds) == n_group
        assert len(typical_indices) == n_group

        dist_f = self.use_dim_decorator(use_dim)(geometry.distance_to_spheres)

        for group_index, (
            typical_person,
            selected_indices,
            means,
            stds,
        ) in enumerate(
            zip(
                typical,
                typical_indices,
                typical_distance_means,
                typical_distance_stds,
            )
        ):
            n_spheres, n_points_per_component = selected_indices.shape

            if n_spheres == 0:
                if verbose:
                    print(f"Warning: no components for group {group_index}")
                pred_scores[:, group_index] = eps
                continue

            typical_person_data = self.data[typical_person, selected_indices]
            assert typical_person_data.shape == (n_spheres, n_points_per_component, 4)

            typical_components: ComponentGroup = self.component_groups[typical_person]
            sphere_components = typical_components.filter(lambda c: c.shape == Shape.SPHERE)
            assert len(sphere_components) == n_spheres

            assert means.shape == (n_spheres, n_points_per_component)
            assert stds.shape == (n_spheres, n_points_per_component)

            centers =sphere_components.param1
            radius = sphere_components.param2
            assert centers.shape == (n_spheres, 4)
            assert radius.shape == (n_spheres,)

            # calculate scores
            group_data = data[:, selected_indices, :]
            n_people, n_spheres, n_points_per_component, n_dim = group_data.shape
            assert n_dim == 4
            assert group_data.shape == (n_people, n_spheres, n_points_per_component, n_dim)
            assert centers.shape == (n_spheres, n_dim)
            assert radius.shape == (n_spheres,)
            assert means.shape == (n_spheres, n_points_per_component)
            assert stds.shape == (n_spheres, n_points_per_component)

            distances = dist_f(group_data, centers, radius)
            assert distances.shape == (n_people, n_spheres, n_points_per_component)

            scores: np.ndarray = ((distances - means) / (stds + eps)) ** 2
            assert scores.shape == (n_people, n_spheres, n_points_per_component)

            match score_agg_method:
                case "max":
                    scores = np.amin(scores, axis=(1, 2))
                case "mean":
                    scores = np.mean(scores, axis=(1, 2))
                case "min":
                    scores = np.amax(scores, axis=(1, 2))
                case _:
                    raise ValueError(f"Invalid value for agg_method: {score_agg_method}")
            assert scores.shape == (n_people, )

            pred_scores[:, group_index] = np.exp(-scores)


        if normalize_score:
            pred_scores /= pred_scores.sum(axis=1, keepdims=True)
        pred_label: np.ndarray = pred_scores.argmax(axis=1)
        true_label: np.ndarray = np.array(self.group_result.person2group)

        self.pred_scores = pred_scores
        self.pred_label = pred_label
        self.true_label = true_label

        return pred_scores

    def classify_with_typical_plane(
        self,
        use_dim: use_dim_type,
        data: np.ndarray | None = None,
        score_agg_method: Literal["mean", "max", "min"] = "mean",
        verbose: bool = False,
        normalize_score: bool = True,
    ) -> np.ndarray:
        '''
        Classify the data with typical data

        Parameters
        ----------
        use_dim: use_dim_type
            The dimension of the data to be classified
        data: np.ndarray | None
            The data to be classified, if None, self.data will be used
        score_agg_method: Literal["mean", "max", "min"]
            The method to aggregate the scores of different components
        verbose: bool
            whether to print the information, by default False

        Returns
        -------
        pred_scores: np.ndarray
        '''

        if data is None:
            data = self.data

        # evaluate the classification score for every person in all groups
        n_people, n_proteins, n_dim = data.shape
        n_group = self.group_result.n_group

        pred_scores = np.empty((n_people, n_group))
        assert pred_scores.shape == (n_people, n_group)

        if use_dim not in self.plane_typical_data:
            raise ValueError(f"Typical data for {use_dim} has not been analyzed yet")
        typical_data = self.plane_typical_data[use_dim]

        typical = typical_data.people
        typical_distance_means = typical_data.distance_means
        typical_distance_stds = typical_data.distance_stds
        typical_indices = typical_data.indices
        assert len(typical) == n_group
        assert len(typical_distance_means) == n_group
        assert len(typical_distance_stds) == n_group
        assert len(typical_indices) == n_group

        dist_f = self.use_dim_decorator(use_dim)(geometry.distance_to_planes)

        for group_index, (
            typical_person,
            selected_indices,
            means,
            stds,
        ) in enumerate(
            zip(
                typical,
                typical_indices,
                typical_distance_means,
                typical_distance_stds,
            )
        ):
            n_planes, n_points_per_component = selected_indices.shape

            if n_planes == 0:
                if verbose:
                    print(f"Warning: no components for group {group_index}")
                pred_scores[:, group_index] = eps
                continue

            typical_person_data = self.data[typical_person, selected_indices]
            assert typical_person_data.shape == (n_planes, n_points_per_component, 4)

            typical_components: ComponentGroup = self.component_groups[typical_person]
            plane_components = typical_components.filter(lambda c: c.shape == Shape.PLANE)
            assert len(plane_components) == n_planes

            assert means.shape == (n_planes, n_points_per_component)
            assert stds.shape == (n_planes, n_points_per_component)

            x0s = plane_components.param1
            normals = plane_components.param2
            assert x0s.shape == (n_planes, 4)
            assert normals.shape == (n_planes, 4)

            group_data = data[:, selected_indices, :]
            assert group_data.ndim == 4, f"Invalid dimension for group_data, got {group_data.shape}"
            n_people, n_planes, n_points_per_component, n_dim = group_data.shape
            assert n_dim == 4

            assert group_data.shape == (n_people, n_planes, n_points_per_component, n_dim)
            assert x0s.shape == (n_planes, n_dim)
            assert normals.shape == (n_planes, n_dim)
            assert means.shape == (n_planes, n_points_per_component)
            assert stds.shape == (n_planes, n_points_per_component)

            distances = dist_f(group_data, x0s, normals)
            assert distances.shape == (n_people, n_planes, n_points_per_component)

            scores: np.ndarray = ((distances - means) / (stds + eps)) ** 2
            assert scores.shape == (n_people, n_planes, n_points_per_component)

            match score_agg_method:
                case "max":
                    scores = np.amin(scores, axis=(1, 2))
                case "mean":
                    scores = np.mean(scores, axis=(1, 2))
                case "min":
                    scores = np.amax(scores, axis=(1, 2))
                case _:
                    raise ValueError(f"Invalid value for agg_method: {score_agg_method}")
            assert scores.shape == (n_people, )

            pred_scores[:, group_index] = np.exp(-scores)

        if normalize_score:
            pred_scores /= pred_scores.sum(axis=1, keepdims=True)
        pred_label: np.ndarray = pred_scores.argmax(axis=1)
        true_label: np.ndarray = np.array(self.group_result.person2group)

        self.pred_scores = pred_scores
        self.pred_label = pred_label
        self.true_label = true_label

        return pred_scores

    def classify_with_typical_hausdorff(
        self,
        use_dim: use_dim_type,
        data: np.ndarray | None = None,
        score_agg_method: Literal["mean", "max", "min"] = "mean",
        verbose: bool = False,
        normalize_score: bool = True,
    ):
        '''
        Classify the data with typical data

        Parameters
        ----------
        use_dim: use_dim_type
            The dimension of the data to be classified
        data: np.ndarray | None
            The data to be classified, if None, self.data will be used
        score_agg_method: Literal["mean", "max", "min"]
            The method to aggregate the scores of different components
        verbose: bool
            whether to print the information, by default False

        Returns
        -------
        pred_scores: np.ndarray
        '''

        if data is None:
            data = self.data

        # evaluate the classification score for every person in all groups
        n_people, n_proteins, n_dim = data.shape
        n_group = self.group_result.n_group

        pred_scores = np.empty((n_people, n_group))
        assert pred_scores.shape == (n_people, n_group)

        if use_dim not in self.hausdorff_typical_data:
            raise ValueError(f"Typical data for {use_dim} has not been analyzed yet")
        typical_data = self.hausdorff_typical_data[use_dim]

        typical = typical_data.people
        typical_distance_means = typical_data.distance_means
        typical_distance_stds = typical_data.distance_stds
        typical_indices = typical_data.indices
        assert len(typical) == n_group
        assert len(typical_distance_means) == n_group
        assert len(typical_distance_stds) == n_group
        assert len(typical_indices) == n_group

        dist_f = self.use_dim_decorator(use_dim)(geometry.distance_to_pointsets)

        for group_index, (
            typical_person,
            selected_indices,
            means,
            stds,
        ) in enumerate(
            zip(
                typical,
                typical_indices,
                typical_distance_means,
                typical_distance_stds,
            )
        ):
            n_components, n_points_per_component = selected_indices.shape

            if len(selected_indices) == 0:
                if verbose:
                    print(f"Warning: no components for group {group_index}")
                pred_scores[:, group_index] = 0
                continue

            typical_person_data = self.data[typical_person, selected_indices]
            assert typical_person_data.shape == (n_components, n_points_per_component, 4)

            assert means.shape == (n_components, )
            assert stds.shape == (n_components, )

            group_data = data[:, selected_indices, :]
            assert group_data.ndim == 4, f"Invalid dimension for group_data, got {group_data.shape}"
            n_people, n_components, n_points_per_component, n_dim = group_data.shape
            assert n_dim == 4

            assert group_data.shape == (n_people, n_components, n_points_per_component, n_dim)
            assert typical_person_data.shape == (n_components, n_points_per_component, n_dim)
            assert means.shape == (n_components, )
            assert stds.shape == (n_components, )

            distances = dist_f(group_data, typical_person_data)
            assert distances.shape == (n_people, n_components)

            scores: np.ndarray = ((distances - means) / (stds + eps)) ** 2
            assert scores.shape == (n_people, n_components)

            match score_agg_method:
                case "max":
                    scores = np.amin(scores, axis=1)
                case "mean":
                    scores = np.mean(scores, axis=1)
                case "min":
                    scores = np.amax(scores, axis=1)
                case _:
                    raise ValueError(f"Invalid value for agg_method: {score_agg_method}")
            assert scores.shape == (n_people, )

            pred_scores[:, group_index] = np.exp(-scores)


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
        data: np.ndarray | None = None,
        score_agg_method: Literal["mean", "max", "min"] = "mean",
        dist_type: analyze_dist_type = "default",
        verbose: bool = False,
        normalize_score: bool = True,
    ):
        '''
        Classify the data with typical data

        Parameters
        ----------
        use_dim: use_dim_type
            The dimension of the data to be classified
        data: np.ndarray | None
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
        '''

        match dist_type:

            case "default":

                score_sphere = self.classify_with_typical_sphere(
                    use_dim=use_dim,
                    data=data,
                    score_agg_method=score_agg_method,
                    verbose=verbose,
                    normalize_score=False
                )

                score_plane = self.classify_with_typical_plane(
                    use_dim=use_dim,
                    data=data,
                    score_agg_method=score_agg_method,
                    verbose=verbose,
                    normalize_score=False
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

            case "sphere":

                score = self.classify_with_typical_sphere(
                    use_dim=use_dim,
                    data=data,
                    score_agg_method=score_agg_method,
                    verbose=verbose,
                    normalize_score=normalize_score
                )

            case "plane":

                score = self.classify_with_typical_plane(
                    use_dim=use_dim,
                    data=data,
                    score_agg_method=score_agg_method,
                    verbose=verbose,
                    normalize_score=normalize_score
                )

            case "hausdorff":

                score = self.classify_with_typical_hausdorff(
                    use_dim=use_dim,
                    data=data,
                    score_agg_method=score_agg_method,
                    verbose=verbose,
                    normalize_score=normalize_score
                )

            case _:
                raise ValueError(f"Invalid value for dist_type: {dist_type}")

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
                return lambda f: wraps(f)(lambda data, *args: f(*(np.mean(arr, axis=-1, keepdims=True) if arr.shape[-1] == 4 else arr for arr in (data, ) + args)))
            case "1D4D":
                return lambda f: wraps(f)(lambda data, *args: f(np.broadcast_to(np.mean(data, axis=-1, keepdims=True), data.shape), *args))

    def show_classification_result(self):
        utils.show_classification_result(self.true_label, self.pred_scores)

    def predict(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

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
        "data/patients_dist_sheet.npy"
    )
    patients_core.cluster_by_gmm(
        n_group=4, feature_getter=lambda cgs: cgs.inverse_total_curvatures[:, None])
    patients_core.analyze_typical_sphere(use_dim="1D4D")
    patients_core.classify_with_typical_sphere(use_dim="1D4D")
    patients_core.show_classification_result()
