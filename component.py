import json
import os
import logging
from functools import cache
from types import EllipsisType
from typing import (
    Callable,
    Iterable,
    Iterator,
    Literal,
    Sequence,
    SupportsIndex,
    overload,
)

import numpy as np
import plotly.express as px

import fitting
import geometry
import J_loss
from geometry import (
    Geometry,
    HyperPlaneConvexHull,
    HyperSphereShellCap,
)

DIMENSION = 4
FILTER_RATE = 0.9


class Shape:
    """
    An enumerated class for the shape attribute.
    """

    SPHERE = "sphere"
    PLANE = "plane"
    CYLINDER = "cyclinder"
    CONE = "cone"


class NotEnoughPointsError(RuntimeError):
    """
    Exception raised when there are not enough points to fit a shape.
    """

    pass


class Component:
    __data_dict = {
        "AD": np.load("data/AD.npy"),
        "NL": np.load("data/Normal.npy"),
    }

    @property
    def data(self) -> np.ndarray:
        try:
            return self.__data[self.dataIndex][:, self.usedim]
        except AttributeError:
            pass
        if self.dataType is None:
            raise ValueError("dataType is not specified")
        if self.dataType in ["AD", "NL"]:
            assert isinstance(self.personID, int | np.integer), "personID is not specified"
            return self.__data_dict[self.dataType][self.personID, self.dataIndex][:, self.usedim]
        else:
            raise ValueError("dataType not found")

    @data.setter
    def data(self, value: np.ndarray):
        # if not np.allclose(self.data, value):
        #     raise ValueError("data is not consistent with dataType")
        self.__data = value

    def __init__(
        self,
        box=None,
        dataIndex: list = None,
        shape: Shape = None,
        param1=None,
        param2=None,
        param3=None,
        loss=None,
        dataType=None,
        personID=...,
        usedim=...,
        data: np.ndarray = None,
    ) -> None:
        """
        Parameters
        ----------
        box : list of box index within the component
        shape : plane, sphere, cone, cylinder
        param1 : center(sphere), x0(plane), vertex(cone), center(cylinder)
        param2 : radius(sphere), normal(plane), direction(cone), direction(cylinder)
        param3 : none(sphere), none(plane), cos_theta(cone), radius(cylinder)
        dataIndex : list of data index (0~23200)
        dataType : choose from "AD", "NL", "TPAD", "TPNL"
        personID : the person ID of the component
        usedim : the used dimension of the data
        """
        if isinstance(box, list):
            self.box = box
        elif isinstance(box, int):
            self.box = [box]
        elif box is None:
            self.box = None
        else:
            raise ValueError("box must be a list or an integer")

        self.shape = shape
        self.loss = loss
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3
        self.dataIndex = dataIndex
        self.dataType = dataType
        self.personID = personID
        self.usedim = usedim
        if data is not None:
            self.data = data

    def __str__(self):
        attributes = {
            "box": self.box,
            "shape": self.shape,
            "loss": self.loss,
            "param1": self.param1,
            "param2": self.param2,
            "param3": self.param3,
            "data.shape": self.data.shape,
            "dataType": self.dataType,
            # "dataIndex": self.dataIndex,
        }
        attribute_strs = [f"{k}: {v}" for k, v in attributes.items() if v is not None]

        return "\n".join(attribute_strs)

    def setShape(self, shape):
        self.shape = shape

    def setParams(self, param1=None, param2=None, param3=None):
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3

    def setLoss(self, loss):
        self.loss = loss

    def setDataType(self, dataType, personID=...):
        assert dataType in self.__data_dict, f"dataType {dataType} not found"
        self.dataType = dataType
        self.personID = personID

    def analysis(self, data, mode="Plane&Sphere", checkbox=False, verbose=False):
        assert self.dataIndex is not None
        data_comp = np.array([data[i] for i in self.dataIndex])
        if checkbox:
            print(len(data_comp))
            dataBox = {}
            for p in data_comp:
                box = encodeData(p)
                if box not in dataBox:
                    dataBox[box] = 1
                else:
                    dataBox[box] += 1
            dataBox = dict(sorted(dataBox.items(), key=lambda x: x[1], reverse=True))
            print(dataBox)
            raise ValueError("checkbox")
        if mode == "Line":
            cLine, wLine = fitting.fit_line(data_comp)
            # print(wLine)
            # JLine = J_loss.J_line(cLine, wLine, data_comp)
            self.setParams(param1=cLine, param2=wLine)
            self.setShape(shape="line")
            # self.setLoss(loss=JLine)
            return
        if mode == "PlaneOnly":
            cPlane, wPlane = fitting.fit_plane(data_comp)
            JPlane = J_loss.J_plane(cPlane, wPlane, data_comp)
            self.setParams(param1=cPlane, param2=wPlane)
            self.setShape(shape=Shape.PLANE)
            self.setLoss(loss=JPlane)
            if verbose:
                print(f"{mode = }  {JPlane = }")
            return
        if mode == "Plane&Sphere":
            cPlane, wPlane = fitting.fit_plane(data_comp)
            JPlane = J_loss.J_plane(cPlane, wPlane, data_comp)
            cSphere, rSphere = fitting.fit_sphere(data_comp)
            JSphere = J_loss.J_sphere(cSphere, rSphere, data_comp)
            self.setParams(param1=cPlane, param2=wPlane)
            self.setShape(shape=Shape.PLANE)
            self.setLoss(loss=JPlane)
            if verbose:
                print(f"{mode = }  {JPlane = }  {JSphere = }")
            if JPlane < JSphere:
                self.setParams(param1=cPlane, param2=wPlane)
                self.setShape(shape=Shape.PLANE)
                self.setLoss(loss=JPlane)
                return
            else:
                self.setParams(param1=cSphere, param2=rSphere)
                self.setShape(shape=Shape.SPHERE)
                self.setLoss(loss=JSphere)
                return

        cPlane, wPlane = fitting.fit_plane(data_comp)
        JPlane = J_loss.J_plane(cPlane, wPlane, data_comp)
        cSphere, rSphere = fitting.fit_sphere(data_comp)
        JSphere = J_loss.J_sphere(cSphere, rSphere, data_comp)
        cCylinder, wCylinder, rCylinder = fitting.fit_cylinder(data_comp, k=5)
        JCylinder = J_loss.J_cylinder(cCylinder, wCylinder, rCylinder, data_comp)
        cCone, wCone, alphaCone = fitting.fit_cone(data_comp, k=5)
        JCone = J_loss.J_cone(cCone, wCone, alphaCone, data_comp)

        _min = min(JPlane, JSphere, JCylinder, JCone)

        if _min == JPlane:
            self.setParams(param1=cPlane, param2=wPlane)
            self.setShape(shape=Shape.PLANE)
        if _min == JSphere:
            self.setParams(param1=cSphere, param2=rSphere)
            self.setShape(shape=Shape.SPHERE)
        if _min == JCylinder:
            self.setParams(param1=cCylinder, param2=wCylinder, param3=rCylinder)
            self.setShape(shape=Shape.CYLINDER)
        if _min == JCone:
            self.setParams(param1=cCone, param2=wCone, param3=alphaCone)
            self.setShape(shape=Shape.CONE)
        self.setLoss(loss=_min)

    @property
    @cache
    def geometry(self) -> HyperPlaneConvexHull | HyperSphereShellCap:
        assert self.param1 is not None
        assert self.param2 is not None
        if self.shape == Shape.PLANE:
            return HyperPlaneConvexHull.from_points(self.data)
        elif self.shape == Shape.SPHERE:
            return HyperSphereShellCap.from_points(self.param1, self.param2, self.data)
        else:
            raise NotImplementedError

    @property
    @cache
    def curvature(self) -> float:
        return self.geometry.curvature

    @property
    def total_curvature(self) -> float:
        return self.geometry.total_curvature

    @property
    def inverse_curvature(self) -> float:
        return 1 / self.geometry.curvature if self.geometry.curvature != 0 else float("inf")

    @property
    def inverse_total_curvature(self) -> float:
        return 1 / self.geometry.total_curvature if self.geometry.total_curvature != 0 else float("inf")

    @property
    def cur_area(self) -> float:
        return self.geometry.cur_area

    def resample(self, size: int) -> np.ndarray:
        """
        Resample points on the fitted geometry
        """
        return self.geometry.sample(size)

    @property
    @cache
    def area(self) -> float:
        return self.geometry.area

    def get_typical_indices(self, top_k: int, selectable_indices: np.ndarray = None) -> np.ndarray:
        """
        Get the typical indices of the component

        Parameters
        ----------
        top_k : int
            the number of typical indices to get
        selectable_indices : np.ndarray, optional
            the indices that can be selected, by default None

        Returns
        -------
        np.ndarray
            the typical indices of the component as a 1D array of int
        """

        match self.shape:
            case Shape.SPHERE:
                # get the closest points on the sphere
                center, radius = np.array(self.param1), np.array(self.param2)
                dist_to_comp = geometry.distance_to_sphere(self.data[:, :], center[:], radius)

            case Shape.PLANE:
                # get the closest points on the plane
                x0, normal = np.array(self.param1), np.array(self.param2)
                dist_to_comp = geometry.distance_to_plane(self.data[:, :], x0[:], normal[:])

            case _:
                raise NotImplementedError

        # local index to global index
        self.dataIndex = np.array(self.dataIndex)
        selected_indices = self.dataIndex[np.argsort(dist_to_comp)]
        if selectable_indices is not None:
            selected_indices = np.intersect1d(selected_indices, selectable_indices)

        # select top k points
        selected_indices = selected_indices[:top_k]
        if len(selected_indices) < top_k:
            raise NotEnoughPointsError(f"Not enough points to select {top_k} typical indices, only {len(selected_indices)} available.")

        return selected_indices

    def to_tuple(self) -> tuple:
        return (
            self.shape,
            tuple(self.param1) if isinstance(self.param1, Iterable) else self.param1,
            tuple(self.param2) if isinstance(self.param2, Iterable) else self.param2,
            tuple(self.param3) if isinstance(self.param3, Iterable) else self.param3,
        )

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Component):
            return False
        else:
            return self.to_tuple() == o.to_tuple()

    def __hash__(self) -> int:
        return hash(self.to_tuple())


class ComponentGroup:
    def __init__(self, components: Sequence[Component]):
        self.components = components

    def __len__(self) -> int:
        return len(self.components)

    def __getitem__(self, key) -> Component:
        return self.components[key]

    def __iter__(self) -> Iterator[Component]:
        return iter(self.components)

    def resample(self, size: int, return_shape=False, random_state: int = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Sample points on the fitted geometries, with the number of points proportional to the area of the geometries

        Parameters
        ----------
        size : int
            the number of points to sample
        return_shape : bool, optional
            whether to return the shape of the sampled points, by default False

        Returns
        -------
        np.ndarray | tuple[np.ndarray, np.ndarray]
            the sampled points, or the sampled points and their shapes
        """
        resampled = []
        shapes = []
        p = np.array([c.area for c in self.components])
        p /= p.sum()
        choices = np.random.choice(np.arange(len(self.components)), size=size, p=p)
        for c, n in zip(*np.unique(choices, return_counts=True)):
            resampled.extend(self[c].resample(n))
            shapes.extend([self[c].shape] * n)
        if return_shape:
            return np.array(resampled), np.array(shapes)
        else:
            return np.array(resampled)

    def scatter(self, dims: list[int] = [0, 1, 2], size: int = 1000):
        """
        Scatter plot of the components

        Parameters
        ----------
        dims : list[int], optional
            the dimensions to plot, by default [0, 1, 2]
        size : int, optional
            the number of points to sample, by default 1000
        """
        samples, shapes = self.resample(size, return_shape=True)
        figure = px.scatter_3d(
            x=samples[:, dims[0]],
            y=samples[:, dims[1]],
            z=samples[:, dims[2]],
            color=shapes,
        )
        figure.show()

    @property
    def area(self) -> float:
        return sum(c.area for c in self.components)

    @property
    def curvature(self) -> float:
        return self.total_curvature / self.area

    @property
    def total_curvature(self) -> float:
        return sum(c.total_curvature for c in self.components)

    @property
    def inverse_curvature(self) -> float:
        return sum(c.inverse_curvature for c in self.components if c.curvature != 0) / len(self.components)

    @property
    def inverse_total_curvature(self) -> float:
        return sum(c.inverse_total_curvature for c in self.components if c.curvature != 0)

    @property
    def total_cur_area(self) -> float:
        return sum(c.cur_area for c in self.components)

    @property
    def cur_area(self) -> float:
        return self.total_cur_area / self.total_area

    @property
    def total_area(self) -> float:
        return sum(c.area for c in self.components)

    @property
    def shapes(self) -> list[Shape]:
        assert all(c.shape is not None for c in self.components)
        return [c.shape for c in self.components]  # type: ignore

    @property
    def geometries(self) -> list[Geometry]:
        return [c.geometry for c in self.components]

    @property
    def param1(self) -> np.ndarray:
        return np.array([c.param1 for c in self.components])

    @property
    def param2(self) -> np.ndarray:
        return np.array([c.param2 for c in self.components])

    @property
    def param3(self) -> np.ndarray:
        return np.array([c.param3 for c in self.components])

    @staticmethod
    def fit(data, split_config={}, analyze_config={}, component_config={}) -> "ComponentGroup":
        components = dataSplit2Components(data, **split_config, component_config=component_config)
        for comp in components:
            comp.analysis(data=data, **analyze_config)
        return ComponentGroup(components)

    def get_typical_indices(
        self,
        top_k: int,
        selectable_indices: np.ndarray = None,
    ) -> np.ndarray:
        """
        Get the typical indices of the components in the group

        Parameters
        ----------
        top_k : int
            the number of typical indices to get
        selectable_indices : np.ndarray, optional
            the indices that can be selected, by default None

        Returns
        -------
        np.ndarray
            the typical indices of the components in the group as a 2D array of int
        """
        return np.array([comp.get_typical_indices(top_k, selectable_indices) for comp in self])

    def filter(self, f: Callable[[Component], bool]) -> "ComponentGroup":
        return ComponentGroup(list(filter(f, self)))


class ComponentGroups:
    def __init__(self, arg, **kwargs):
        assert isinstance(arg, Sequence) and isinstance(arg[0], ComponentGroup)
        self.groups = np.array(arg, dtype=ComponentGroup)

    @overload
    def __getitem__(self, key: int | np.integer) -> ComponentGroup: ...
    @overload
    def __getitem__(self, key: EllipsisType | SupportsIndex) -> "ComponentGroups": ...

    def __getitem__(self, key: int | np.integer | EllipsisType | SupportsIndex) -> "ComponentGroup | ComponentGroups":
        if isinstance(key, (int, np.integer)):
            return self.groups[key]
        elif isinstance(key, (EllipsisType, SupportsIndex)):
            obj = object.__new__(ComponentGroups)
            obj.groups = self.groups[key]
            return obj
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self) -> Iterator[ComponentGroup]:
        return iter(ComponentGroup(g) for g in self.groups)

    def __add__(self, other: "ComponentGroups") -> "ComponentGroups":
        return ComponentGroups([*self.groups, *other.groups])

    @property
    def usedim(self):
        return self.groups[0][0].usedim

    @usedim.setter
    def usedim(self, value):
        for g in self.groups:
            for c in g:
                c.usedim = value

    @property
    def dataType(self) -> Literal["AD", "NL", "TPAD", "TPNL"]:
        return self.groups[0][0].dataType

    @dataType.setter
    def dataType(self, value: Literal["AD", "NL", "TPAD", "TPNL"]):
        for g in self.groups:
            for c in g:
                c.dataType = value

    @property
    def areas(self) -> np.ndarray:
        return np.array([cs.area for cs in self])

    @property
    def curvatures(self) -> np.ndarray:
        return np.array([cs.curvature for cs in self])

    @property
    def total_curvatures(self) -> np.ndarray:
        return np.array([cs.total_curvature for cs in self])

    @property
    def inverse_curvatures(self) -> np.ndarray:
        return np.array([cs.inverse_curvature for cs in self])

    @property
    def inverse_total_curvatures(self) -> np.ndarray:
        return np.array([cs.inverse_total_curvature for cs in self])

    @property
    def total_cur_areas(self) -> np.ndarray:
        return np.array([cs.total_cur_area for cs in self])

    @property
    def cur_areas(self) -> np.ndarray:
        return np.array([cs.cur_area for cs in self])

    @property
    def shapes(self) -> list[list[Shape]]:
        return [cs.shapes for cs in self]

    @property
    def geometries(self) -> list[list[Geometry]]:
        return [cs.geometries for cs in self]

    def set_distance_cache_path(self, path: str):
        self.distance_cache_path = path

    distance_cache_path = "data/dist_sheet.npy"


def encodeData(point, l=-5, r=15) -> int:
    # no scaling
    if point.shape[0] == DIMENSION + 1:
        # point = point - np.array([l, l, l, l, 0])
        point = point[:-1]
    # elif point.shape[0] == DIMENSION:
    point = point - np.full((DIMENSION,), l)

    _range = r - l
    index = 0
    for i in range(DIMENSION):
        index += int(point[i]) * _range ** (DIMENSION - i - 1)
    # index = int(point[0]) * _range**3 \
    #     + int(point[1]) * _range**2 \
    #     + int(point[2]) * _range \
    #     + int(point[3])
    return int(index)


def dataSplit2Components(data, l=-5, r=15, constraint=100, scale=1, component_config: dict = {}) -> list[Component]:
    r *= scale
    l *= scale
    comp_list_data = [[] for _ in range(((r - l) ** DIMENSION))]
    for j, d in enumerate(data):
        # print(d, encodeData(d, l, r))
        index = encodeData(d, l, r)
        if index >= len(comp_list_data) or index < 0:
            continue
        comp_list_data[index].append(j)  # the index
        # comp_list_data[index].append((d, index))
        # comp_list_data[index].append(d)
    # print(comp_list)
    comp_list_data = [(i, lst) for i, lst in enumerate(comp_list_data) if len(lst) >= constraint]
    # print(comp_list_data[0][0])
    comp_list = [Component(lst[0], lst[1], **component_config) for lst in comp_list_data]
    # print(comp_list[1].__dict__)
    return comp_list
