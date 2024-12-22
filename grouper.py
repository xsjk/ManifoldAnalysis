import numpy as np
import pandas as pd
from typing import overload
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap

class GroupResult:

    person2group: list[int]
    group2people: list[list[int]]
    group2boolean: list[list[bool]]

    @overload
    def __init__(self, res: list[int] | np.ndarray):
        '''
        Parameters
        ----------
        person2group: list[int] | np.ndarray
            person2group[i] is the group id of the i-th person
        '''
        ...

    @overload
    def __init__(self, res: list[list[int]] | list[np.ndarray]):
        '''
        Parameters
        ----------
        group2people: list[list[int]] | list[np.ndarray]
            group2people[i] is the list of people in group i.
        '''
        ...

    @overload
    def __init__(self, res: list[list[bool]]):
        '''
        Parameters
        ----------
        group2boolean: list[list[bool]]
            group2boolean[i][j] represents whether the j-th person belong to the i-th group.
        '''
        ...

    def __init__(self, res: np.ndarray | list[int] | list[list[int]] | list[np.ndarray] | list[list[bool]]):
        if isinstance(res[0], int | np.integer):
            self.person2group = list(res) # type: ignore
            self.group2people = [[i for i, g in enumerate(self.person2group) if g == j] for j in range(max(self.person2group)+1)]
            self.group2boolean = [[g == j for g in self.person2group] for j in range(max(self.person2group)+1)]
        elif isinstance(res[0][0], bool | np.bool_): # type: ignore
            self.person2group = [0] * len(res[0]) # type: ignore
            for i, g in enumerate(res):
                for j, p in enumerate(g): # type: ignore
                    if p:
                        self.person2group[j] = i
            self.group2people = [[i for i, g in enumerate(self.person2group) if g == j] for j in range(max(self.person2group)+1)]
            self.group2boolean = res # type: ignore
        elif isinstance(res[0][0], int | np.integer): # type: ignore
            self.person2group = [0] * (max(map(max, res)) + 1) # type: ignore
            for i, g in enumerate(res):
                for p in g: # type: ignore
                    self.person2group[p] = i
            self.group2people = res # type: ignore
            self.group2boolean = [[g == j for g in self.person2group] for j in range(max(self.person2group)+1)]

    def __repr__(self) -> str:
        return f'GroupResult({self.group2people})'

    @property
    def n_group(self) -> int:
        return len(self.group2people)

    @property
    def n_people(self) -> int:
        return len(self.person2group)

    @staticmethod
    def display_relation(result1: 'GroupResult', result2: 'GroupResult', ax=None,
                         cmap: Colormap | str = 'viridis') -> plt.Axes: # type: ignore
        '''
        Display the relation between two GroupResult.

        Parameters
        ----------
        result1 : GroupResult
            The first group result.
        result2 : GroupResult
            The second group result.
        ax : matplotlib.axes.Axes, optional
            The axes to plot on, by default None.
        cmap : str, optional
            The colormap to use, by default 'viridis'

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.
        '''

        m = np.zeros((result2.n_group, result1.n_group), dtype=int)
        for i in range(result2.n_group):
            for j in range(result1.n_group):
                m[i, j] = len(set(result1.group2people[j]) & set(result2.group2people[i]))
        if ax is None:
            im = plt.imshow(m, cmap=cmap)
            plt.title('Split Matrix')
            plt.colorbar(im, label='Number of People')
            return im # type: ignore
        else:
            return ax.imshow(m, vmin=0, vmax=13)

    @staticmethod
    def similarity(result1: 'GroupResult', result2: 'GroupResult') -> float:
        '''
        Calculate the similarity between two GroupResult.
        (the probability that two people in the same group of `result1` are in the same group of `result2`)

        Parameters
        ----------
        result1 : GroupResult
            The first group result.
        result2 : GroupResult
            The second group result.

        Returns
        -------
        float
            The similarity between two GroupResult.
        '''
        return sum(max(len(set(result1.group2people[i]) & set(result2.group2people[j]))
                       for j in range(result2.n_group) if result2.group2people[j])
                       for i in range(result1.n_group) if result1.group2people[i]) / result1.n_people

    @staticmethod
    def jaccard_similarity(result1: 'GroupResult', result2: 'GroupResult') -> float:
        '''
        Calculate the Jaccard similarity between two GroupResult.

        Parameters
        ----------
        result1 : GroupResult
            The first group result.
        result2 : GroupResult
            The second group result.

        Returns
        -------
        float
            The Jaccard similarity between two GroupResult.
        '''
        return sum(max(len(set(result1.group2people[i]) & set(result2.group2people[j])) / len(set(result1.group2people[i]) | set(result2.group2people[j]))
                       for j in range(result2.n_group) if result2.group2people[j])
                       for i in range(result1.n_group) if result1.group2people[i]) / result1.n_group


    @classmethod
    def register_protein_map(cls, protein_map: pd.Series):
        cls.protein_map = protein_map
        cls.all_proteins = sorted(set.union(*map(set, cls.protein_map.values)))

    def hist_protein(self) -> pd.DataFrame:
        hists = pd.DataFrame(0, index=self.all_proteins, columns=range(self.n_group))
        for col, group in zip(hists.columns, self.group2people):
            for person in group:
                hists[col][self.protein_map[person]] += 1
            hists[col] /= len(group)
        hists.plot.bar(xticks=[], rot=90, subplots=True, figsize=(20, 20), ylim=(0, 1), grid=True)
        return hists

    @staticmethod
    def split_by(values, splits) -> 'GroupResult':
        '''
        Create GroupResult by splitting values at given splits (splits[i] to splits[i+1] as group i)

        Parameters
        ----------
        values : np.ndarray
            Values to split.
        splits : list
            Split points.
        '''
        intervals = [(splits[i], splits[i+1]) for i in range(len(splits)-1)]
        values = np.array(values)
        label = np.empty(len(values), dtype=int)
        for i, (lb, ub) in enumerate(intervals):
            label[(values >= lb) & (values < ub)] = i
        return GroupResult(label)

    def __add__(self, other: 'GroupResult') -> 'GroupResult':
        return GroupResult(
            [np.array(p) for p in self.group2people] +
            [np.array(p) + self.n_people for p in other.group2people]
        )
