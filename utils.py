from typing import Literal
import numpy as np
import pandas as pd
import xarray as xr
import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.colors import Colormap, Normalize
from matplotlib.axes import Axes
from plotly import graph_objects as go
from itertools import product
from sklearn.metrics import auc, roc_curve, precision_recall_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, jaccard_score, balanced_accuracy_score, average_precision_score
from sklearn.preprocessing import label_binarize


def file2data(file: str, sheet=0, withIndex=True) -> np.ndarray:
    """
    load data from excel file

    Parameters
    ----------
    file : str
        file name
    sheet : int, optional
        sheet number, by default 0
    withIndex : bool, optional
        whether to add index column, by default True

    Returns
    -------
    np.ndarray (RNA, feature)

    Examples
    --------
    >>> data = file2data("./Desktop/AD/AMPAD_MSSM_0000003566.xlsx")
    >>> data.shape
    (23201, 4)
    """
    data = pd.read_excel(file, sheet_name=sheet).iloc[:, [1, 3, 4]].to_numpy()
    if withIndex:
        data = np.hstack((data, np.arange(len(data)).reshape(-1, 1)))
    return data


def csvfile2data(file: str, withIndex=True) -> np.ndarray:
    data = pd.read_csv(file).iloc[:, 1:].to_numpy()
    if withIndex:
        data = np.hstack((data, np.arange(len(data)).reshape(-1, 1)))
    return data


def file2title(file: str, sheet=0):
    title = pd.read_excel(file, sheet_name=sheet).iloc[:, :1].to_numpy()
    return title


def traverse_dir(url=".") -> list:
    """
    traverse directory

    Parameters
    ----------
    url : str, optional
        directory url, by default "."
    Returns
    -------
    list
        file names
    """
    return [file[:-5] for file in os.listdir(url) if file.startswith("A")]


def load_all(file: str) -> np.ndarray:
    """
    load all data from binary file

    Parameters
    ----------
    file : str
        file name

    Returns
    -------
    np.ndarray (person, RNA, feature)

    Examples
    --------
    >>> AD = load_all("AD")
    >>> AD.shape
    (86, 23201, 4)
    >>> Normal = load_all("Normal")
    >>> Normal.shape
    (36, 23201, 4)
    """
    return np.load(f"./data/{file}.npy")


def getResampledData(file):
    with open(file, "r") as f:
        data = f.read().strip()

    _all_points = []
    for i, line in enumerate(data.split(",\n")):
        component_dict = json.loads(line.strip(','))
        # # print(component_dict)
        dda = np.array(component_dict[str(i)])
        # concatenate all dda
        _all_points.append(dda)

    # print(all_points.shape)
    return _all_points


def plot_matrix(mat: np.ndarray, classes: list[str], *, normalize=False, ax: Axes | None = None, cmap="Blues") -> Axes:
    '''
    This function prints and plots the confusion matrix.

    Parameters
    ----------
    mat : np.ndarray
        matrix
    classes : list[str]
        class names
    normalize : bool, optional
        whether to normalize, by default False
    ax : plt.Axes, optional
        the axis to plot on, by default None
    cmap : plt.cm, optional
        color map, by default plt.cm.Blues

    Returns
    -------
    plt.Axes
        axis to plot on
    '''
    if normalize:
        mat = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]

    tick_marks = np.arange(len(classes))

    ax = ax or plt.gca()
    im = ax.imshow(mat, interpolation='nearest', cmap=cmap)
    plt.colorbar(ax=ax, mappable=im)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    if normalize:
        im.set_clim(0, 1)

    fmt = '.2f' if normalize else 'd'
    thresh = mat.max() / 2.
    for i, j in product(range(mat.shape[0]), range(mat.shape[1])):
        ax.text(j, i, format(mat[i, j], fmt),
                horizontalalignment="center",
                color="white" if mat[i, j] > thresh else "black")
    return ax


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, *, normalize: Literal["true", "pred", "all"] | None = "true", ax: Axes | None = None, cmap='Blues', names: list[str] | None = None) -> Axes:
    """
    Plot confusion matrix for the given classifier.

    Parameters:
    ----------
    y_true: np.ndarray
        True labels.
    y_pred: np.ndarray
        Predicted labels.
    normalize: bool, optional
        Whether to normalize, by default True.
    ax: Axes, optional
        The axis to plot on, by default None.
    cmap: str, optional
        Colormap for confusion matrix, by default 'Blues'.
    names: list[str], optional
        Name of the classes, by default None.

    Returns
    -------
    plt.Axes
        axis to plot on
    """
    ax = plot_matrix(confusion_matrix(y_true, y_pred, normalize=normalize), classes=[str(
        c) for c in np.unique(y_true)], normalize=normalize is not None, ax=ax, cmap=cmap)
    if names is not None:
        ax.set_xticks(range(len(names)), names)
        ax.set_yticks(range(len(names)), names, rotation=90, va='center')
    return ax


def plot_confusion_matrix_hetro(y_true: np.ndarray, y_pred: np.ndarray, *, normalize: Literal["true", "pred", "all"] | None = "true", ax: Axes | None = None, cmap: str | Colormap = 'Blues', names: list[str] | None = None) -> Axes:
    """
    Plot confusion matrix for the given classifier.

    Parameters:
    ----------
    y_true: np.ndarray
        True labels.
    y_pred: np.ndarray
        Predicted labels.
    normalize: bool, optional
        Whether to normalize, by default True.
    ax: Axes, optional
        The axis to plot on, by default None.
    cmap: str, optional
        Colormap for confusion matrix, by default 'Blues'.
    names: list[str], optional
        Name of the classes, by default None.

    Returns
    -------
    plt.Axes
        axis to plot on
    """

    n = len(np.unique(y_true))

    cm = confusion_matrix(y_true, y_pred)

    fmt = '.2f' if normalize else 'd'

    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)

    x_nodes = np.linspace(0, 1, n + 1)
    y_nodes = np.linspace(0, 1, n + 1)
    if normalize == 'all':
        x_nodes = np.cumsum(np.append([0], col_sums)) / col_sums.sum()
        y_nodes = np.cumsum(np.append([0], row_sums)) / row_sums.sum()
        cm = cm / cm.sum()
    elif normalize == 'pred':
        y_nodes = np.cumsum(np.append([0], row_sums)) / row_sums.sum()
        cm = cm / cm.sum(axis=0)
    elif normalize == 'true':
        x_nodes = np.cumsum(np.append([0], col_sums)) / col_sums.sum()
        cm = cm / cm.sum(axis=1)[:, np.newaxis]

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    ax = ax or plt.gca()

    max_value = cm.max()
    min_value = cm.min()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]

            if normalize is None or normalize == 'all':
                color_intensity = (value - min_value) / (max_value - min_value)
                color = cmap(color_intensity)
            else:
                color = cmap(value)

            x, y = x_nodes[j], y_nodes[i]

            width = x_nodes[j + 1] - x_nodes[j]
            height = y_nodes[i + 1] - y_nodes[i]

            rect = Rectangle((x, y), width, height,
                             facecolor=color, edgecolor=None)
            ax.add_patch(rect)

            ax.text(x + width / 2, y + height / 2, format(value, fmt),
                    horizontalalignment='center', verticalalignment='center', color='black')

    if names is None:
        names = [str(i) for i in range(n)]

    ax.set_xticks(x_nodes[:-1] + np.diff(x_nodes) / 2, names)
    ax.set_yticks(y_nodes[:-1] + np.diff(y_nodes) /
                  2, names, rotation=90, va='center')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.invert_yaxis()

    plt.colorbar(
        plt.cm.ScalarMappable(
            norm=Normalize(vmin=min_value, vmax=max_value),
            cmap=cmap
        ),
        ax=ax,
        orientation='vertical',
        shrink=1,
        aspect=20
    )

    return ax


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, *, ax: Axes | None = None, axes: list[Axes] | None = None, subplot=False, cmap='tab10') -> Axes | list[Axes]:
    """
    Plot ROC curve for the given classifier.

    Parameters
    ----------
    y_true: np.ndarray
        True labels.
    y_score: np.ndarray
        Predicted probabilities.
    ax: Axes, optional
        The axis to plot on, used when subplot is False, by default None.
    axes: list[Axes], optional
        The axes to plot on, used when subplot is True, by default None.
    subplot: bool, optional
        Whether to plot ROC curve for each class separately, by default True.
    cmap: str, optional
        Colormap for multi-class ROC curve, by default 'tab10'.

    Returns
    -------
    plt.Axes | list[plt.Axes]
        axis to plot on
    """
    unique_classes = np.unique(y_true)
    n_classes = len(unique_classes)\

    if n_classes == 2:
        ax = ax or plt.gca()
        true_label_binarized = y_true
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc="lower right")
        return ax
    elif n_classes == 1:
        raise ValueError("Only one class is present in the true labels.")

    true_label_binarized = label_binarize(y_true, classes=unique_classes)
    assert isinstance(true_label_binarized, np.ndarray)

    fpr, tpr, roc_auc = {}, {}, {}
    for index in unique_classes:
        fpr[index], tpr[index], _ = roc_curve(
            true_label_binarized[:, index], y_score[:, index])
        roc_auc[index] = auc(fpr[index], tpr[index])

    if subplot:
        if axes is None:
            axes = [plt.subplot(1, n_classes, i + 1) for i in range(n_classes)]

        n_axes = len(axes)
        assert n_axes == n_classes, "Number of axes must be equal to number of classes."
        for ax, index in zip(axes, unique_classes):
            ax.plot(fpr[index], tpr[index], color='darkorange',
                    lw=2, label=f'ROC curve (area = {roc_auc[index]:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.05)
            ax.legend(loc="lower right")
            ax.set_title(f'Class {index} ROC curve')
        return axes

    else:
        ax = ax or plt.gca()
        colors = cm.get_cmap(cmap, n_classes)
        for i, index in enumerate(unique_classes):
            ax.plot(fpr[index], tpr[index], lw=2, color=colors(
                i), label=f'ROC curve of class {index} (area = {roc_auc[index]:.2f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc="lower right")
        return ax


def plot_precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray, *, ax: Axes | None = None, axes: list[Axes] | None = None, subplot=False, cmap='tab10') -> Axes | list[Axes]:
    """
    Plot precision-recall curve for the given classifier.

    Parameters
    ----------
    y_true: np.ndarray
        True labels.
    y_score: np.ndarray
        Predicted probabilities.
    ax: Axes, optional
        The axis to plot on, used when subplot is False, by default None.
    axes: list[Axes], optional
        The axes to plot on, used when subplot is True, by default None.
    subplot: bool, optional
        Whether to plot precision-recall curve for each class separately, by default True.
    cmap: str, optional
        Colormap for multi-class precision-recall curve, by default 'tab10'.

    Returns
    -------
    plt.Axes | list[plt.Axes]
        axis to plot on
    """
    unique_classes = np.unique(y_true)
    n_classes = len(unique_classes)

    if n_classes == 2:
        ax = ax or plt.gca()
        true_label_binarized = y_true
        precision, recall, _ = precision_recall_curve(y_true, y_score[:, 1])
        average_precision = average_precision_score(y_true, y_score[:, 1])
        ax.plot(recall, precision, lw=2,
                label=f'Precision-Recall curve (area = {average_precision:.2f})')
        ax.plot([0, 1], [0.5, 0.5], 'k--', lw=2)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(loc="lower right")
        return ax
    elif n_classes == 1:
        raise ValueError("Only one class is present in the true labels.")

    true_label_binarized = label_binarize(y_true, classes=unique_classes)
    assert isinstance(true_label_binarized, np.ndarray)

    precision, recall, average_precision = {}, {}, {}
    for index in unique_classes:
        precision[index], recall[index], _ = precision_recall_curve(
            true_label_binarized[:, index], y_score[:, index])
        average_precision[index] = average_precision_score(
            true_label_binarized[:, index], y_score[:, index])

    if subplot:
        if axes is None:
            axes = [plt.subplot(1, n_classes, i + 1) for i in range(n_classes)]

        n_axes = len(axes)
        assert n_axes == n_classes, "Number of axes must be equal to number of classes."
        for ax, index in zip(axes, unique_classes):
            ax.plot(recall[index], precision[index], color='darkorange',
                    lw=2, label=f'Precision-Recall curve (area = {average_precision[index]:.2f})')
            ax.plot([0, 1], [0.5, 0.5], 'k--', lw=2)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.05)
            ax.legend(loc="lower right")
            ax.set_title(f'Class {index} Precision-Recall curve')
        return axes

    else:
        ax = ax or plt.gca()
        colors = cm.get_cmap(cmap, n_classes)
        for i, index in enumerate(unique_classes):
            ax.plot(recall[index], precision[index], lw=2, color=colors(
                i), label=f'Precision-Recall curve of class {index} (area = {average_precision[index]:.2f})')
        ax.plot([0, 1], [0.5, 0.5], 'k--', lw=2)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(loc="lower right")
        return ax


def show_classification_result(true_label: np.ndarray, pred_scores: np.ndarray, pred_label: np.ndarray | None = None, *, axes: None | tuple[Axes, Axes, Axes] = None, normalize: Literal["true", "pred", "all"] | None = "true", names: list[str] | None = None) -> tuple[Axes, Axes, Axes]:
    '''
    Show classification result

    Parameters
    ----------
    true_label : np.ndarray
        True labels
    pred_scores : np.ndarray
        Predicted scores
    pred_label : np.ndarray, optional
        Predicted labels, by default None, if None, will be calculated from pred_scores use argmax
    axes : tuple[plt.Axes, plt.Axes, plt.Axes], optional
        the axes to plot the confusion matrix, ROC curve and precision-recall curve, by default None
    normalize : Literal["true", "pred", "all"], optional
        whether to normalize, by default "true"
    names : list[str], optional
        the names of the classes, by default None

    Returns
    -------
    tuple[plt.Axes, plt.Axes, plt.Axes]
        the axes to plot the confusion matrix and ROC curve
    '''

    if pred_label is None:
        pred_label = pred_scores.argmax(axis=1)
    assert isinstance(pred_label, np.ndarray)

    num_class = len(np.unique(true_label))
    if num_class == 2:
        average = 'binary'
    else:
        average = 'macro'

    accuracy = accuracy_score(true_label, pred_label)
    balanced_accuracy = balanced_accuracy_score(true_label, pred_label)
    precision = precision_score(true_label, pred_label, average=average)
    recall = recall_score(true_label, pred_label, average=average)
    f1 = f1_score(true_label, pred_label, average=average)
    jaccard = jaccard_score(true_label, pred_label, average=average)

    print(f"         Accuracy: {accuracy:.3f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.3f}")
    print(f"        Precision: {precision:.3f}")
    print(f"           Recall: {recall:.3f}")
    print(f"         F1 Score: {f1:.3f}")
    print(f"    Jaccard Score: {jaccard:.3f}")

    if axes is None:
        figsize = (13, 3) if num_class < 6 else (20, 5)
        plt.figure(figsize=figsize)
        gs = GridSpec(1, 3, width_ratios=[5, 4, 4])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])
    else:
        ax1, ax2, ax3 = axes

    plot_confusion_matrix(true_label, pred_label, ax=ax1,
                          normalize=normalize, names=names)
    plot_roc_curve(true_label, pred_scores, ax=ax2, subplot=False)
    plot_precision_recall_curve(true_label, pred_scores, ax=ax3, subplot=False)
    return ax1, ax2, ax3


def plot_bincount(labels: np.ndarray, ax: Axes | None = None, normalize: bool = False, title: str = "Class Distribution") -> Axes:
    '''
    Plot the bin count of the labels

    Parameters
    ----------
    labels : np.ndarray
        the labels to plot
    ax : plt.Axes, optional
        the axis to plot on, by default None

    Returns
    -------
    plt.Axes
        the axis to plot on
    '''
    ax = ax or plt.gca()
    unique_labels, counts = np.unique(labels, return_counts=True)
    if normalize:
        counts = counts / counts.sum()
    ax.bar(unique_labels, counts)
    ax.set_xlabel("Class")
    ax.set_ylabel("Percentage" if normalize else "Count")
    ax.set_xticks(unique_labels)
    ax.set_title(title)
    return ax


def plot_voxel(array: xr.DataArray, x_label: str, y_label: str, z_label: str) -> go.Figure:
    """
    Plot 3D voxel plot

    Parameters
    ----------
    array : xr.DataArray
        The array to plot.
    x_label : str
        The name of the column for the x-axis from the data array.
    y_label : str
        The name of the column for the y-axis from the data array.
    z_label : str
        The name of the column for the z-axis from the data array.

    Returns
    -------
    go.Figure
        The plotly figure object.
    """
    coords = array.coords

    x, y, z = np.meshgrid(coords[x_label], coords[y_label], coords[z_label])

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x.ravel(),
                y=y.ravel(),
                z=z.ravel(),
                mode='markers',
                marker=dict(
                    size=5,
                    color=array.values.ravel(),
                    colorscale='Viridis',
                    opacity=0.6,
                    colorbar=dict(title=array.name)
                )
            ),
            go.Volume(
                x=x.ravel(),
                y=y.ravel(),
                z=z.ravel(),
                value=array.values.ravel(),
                isomin=array.values.min(),
                isomax=array.values.max(),
                opacity=0.1, # needs to be small to see through all surfaces
                surface_count=17, # needs to be a large number for good volume rendering
                colorscale='Viridis'
            )
        ],
        layout=dict(
            scene=dict(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title=z_label,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=800,
            width=800,
        )
    )

    return fig


def load_cv_result(path: str) -> xr.DataArray:
    '''
    Load the cross-validation result

    Parameters
    ----------
    path : str
        the path of the dataarray to the cross-validation result or the directory containing the cross-validation result

    Returns
    -------
    xr.DataArray
        the cross-validation result
    '''
    if os.path.isdir(path):
        files = os.listdir(path)
        cv_scores = xr.concat([xr.open_dataarray(os.path.join(path, f)) for f in files], dim='fold')
    else:
        cv_scores = xr.open_dataarray(path)

    precision = cv_scores.loc[..., 'test_precision', :]
    precision_macro = cv_scores.loc[..., 'test_precision_macro', :]
    precision_neg = precision_macro * 2 - precision
    recall = cv_scores.loc[..., 'test_recall', :]
    recall_macro = cv_scores.loc[..., 'test_recall_macro', :]
    recall_neg = recall_macro * 2 - recall

    F = 1 / (1 + 1 / (1 / precision + 1 / recall - 2) + 1 / (1 / recall_neg + 1 / precision_neg - 2))
    T = 1 - F
    TP = (1 / (1 / precision + 1 / recall - 2)) * F
    TN = (1 / (1 / precision_neg + 1 / recall_neg - 2)) * F
    FP = (1 / precision - 1) * TP
    FN = (1 / recall - 1) * TP

    FNR = FN / (FN + TP)
    FPR = FP / (FP + TN)
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    FP = FP.assign_coords(metric='FP')
    FN = FN.assign_coords(metric='FN')
    TP = TP.assign_coords(metric='TP')
    TN = TN.assign_coords(metric='TN')

    FNR = FNR.assign_coords(metric='FNR')
    FPR = FPR.assign_coords(metric='FPR')
    TPR = TPR.assign_coords(metric='TPR')
    TNR = TNR.assign_coords(metric='TNR')
    MCC = MCC.assign_coords(metric='MCC')

    other_metrics = xr.concat(
        [TP, FN, FP, TN, TPR, FNR, FPR, TNR, MCC],
        dim=pd.Index(['test_TP', 'test_FN', 'test_FP', 'test_TN', 'test_TPR', 'test_FNR', 'test_FPR', 'test_TNR', 'test_MCC'], name='metric')
    )
    cv_scores = xr.concat([cv_scores, other_metrics], dim='metric')

    return cv_scores


def get_top_k_hyperparameters(cv_scores: xr.DataArray, top_n: int, important_metrics: list[str] = [
    "f1_macro",
    "jaccard_macro",
    "recall_macro",
    "roc_auc"
]) -> pd.DataFrame:
    '''
    Get the top k hyperparameters

    Parameters
    ----------
    cv_scores : xr.DataArray
        the cross-validation scores
    top_n : int
        the number of top hyperparameters to get
    important_metrics : list[str], optional
        the important metrics to consider, by default ["f1_macro", "jaccard_macro", "recall_macro", "roc_auc"]

    Returns
    -------
    pd.DataFrame
        the top k hyperparameters with the corresponding scores
    '''


    *hyperparameter_names, metric_name, fold_name = cv_scores.dims
    assert metric_name == 'metric'
    assert fold_name == 'fold'
    for metric in important_metrics:
        assert 'test_' + metric in cv_scores.coords['metric'].values

    best_indices = {}
    common_best_indices = []
    indices_orders = pd.DataFrame(columns=important_metrics)


    for metric in important_metrics:
        cv_score_means = cv_scores.loc[..., 'test_' + metric, :].mean(dim='fold')
        indices_orders[metric] = cv_score_means.values.ravel().argsort()[::-1]
        best_indices[metric] = set()

    # loop until we find top_n common best hyperparams
    while len(common_best_indices) < top_n:
        for metric in important_metrics:
            new_index = indices_orders[metric][len(best_indices[metric])]
            if all(new_index in best_indices[other_metric]
                   for other_metric in important_metrics
                   if other_metric != metric):
                common_best_indices.append(new_index)
            best_indices[metric].add(new_index)

    common_best_hyperparams = list(zip(*(
        cv_scores.coords[param_name].values[indices]
        for indices, param_name in zip(
            np.unravel_index(common_best_indices, cv_scores.shape[:-2]),
            hyperparameter_names
        )
    )))

    df = pd.DataFrame(index=pd.MultiIndex.from_tuples(common_best_hyperparams, names=hyperparameter_names)).sort_index()

    for p in common_best_hyperparams:
        for metric in cv_scores.coords['metric'].values:
            if metric.startswith("test_"):
                df.loc[p, metric[5:]] = cv_scores.loc[*p, metric].mean(dim='fold').values

    return df