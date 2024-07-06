""" This module contains the functions used to plot comparison graphs.

Functions
------
plot_precision_recall_comparison
    Plots precision-recall curves for different methods.
plot_purity_comparison
    Plots purity (precision) graph depending on the momentum of the particles.
plot_efficiency_comparison
    Plots efficiency (recall) graph depending on the momentum of the particles.
"""
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

from pdi.evaluate import get_interval_purity_efficiency


def plot_precision_recall_comparison(target_name, data_dict, pt_range, pt_percent, save_dir=None):
    plt.figure(figsize=(6, 6))
    plt.axis('equal')
    ax = plt.gca()
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    axins = ax.inset_axes([0.05, 0.05, 0.7, 0.7], zorder=0)
    for method_name, results in data_dict.items():
        targets = results["targets"]
        preds = results["predictions"]
        momentum = results["momentum"]

        mask = (momentum >= pt_range[0]) & (momentum <= pt_range[1])
        targets = targets[mask]
        preds = preds[mask]

        precision, recall, _ = precision_recall_curve(targets, preds)
        area = auc(recall, precision)
        ax.plot(recall, precision, label=method_name + f" AUC: {area:.3f}")
        axins.plot(recall, precision)

    # subregion of the original image
    if "kaon" in target_name:
       x1, x2, y1, y2 = 0.7, 1, 0.7, 1 
    else:
        x1, x2, y1, y2 = 0.9, 1.01, 0.9, 1.01
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])

    ax.indicate_inset_zoom(axins, edgecolor="black")

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title(f"{target_name} classification \n" + "$p_{T} \in$" + f"{pt_range} (GeV/c)")
    plt.legend(loc="lower left")
    if save_dir is not None:
        plt.savefig(f"{save_dir}/precision_recall_{pt_percent}.png")
    plt.show()
    plt.close()


def plot_purity_comparison(target_name, data_dict, intervals, save_dir=None):
    plt.figure()
    for method_name, results in data_dict.items():
        targets = results["targets"]
        preds = results["predictions"]
        fP = results["momentum"]
        threshold = results["threshold"]

        selected = preds > threshold

        purities_p_plot, _, confidence_intervals, momenta_avg = get_interval_purity_efficiency(
            targets, selected, fP, intervals)

        p = plt.plot(momenta_avg, purities_p_plot, label=method_name)
        plt.fill_between(
            momenta_avg,
            confidence_intervals["purity_lower"],
            confidence_intervals["purity_upper"],
            color=p[0].get_color(),
            alpha=0.2,
        )

    plt.xlabel("p (GeV/c)")
    plt.ylabel("Purity")
    plt.ylim([0, 1.1])

    plt.title(f"{target_name} classification")
    plt.legend(loc="lower left")
    if save_dir is not None:
        plt.savefig(f"{save_dir}/p_purity_optimized_threshold.png")
    plt.show()
    plt.close()


def plot_efficiency_comparison(target_name,
                               data_dict,
                               intervals,
                               save_dir=None):
    plt.figure()
    for method_name, results in data_dict.items():
        targets = results["targets"]
        preds = results["predictions"]
        fP = results["momentum"]
        threshold = results["threshold"]

        selected = preds > threshold

        _, efficiencies_p_plot, confidence_intervals, momenta_avg = get_interval_purity_efficiency(
            targets, selected, fP, intervals)

        p = plt.plot(momenta_avg, efficiencies_p_plot, label=method_name)
        plt.fill_between(
            momenta_avg,
            confidence_intervals["efficiency_lower"],
            confidence_intervals["efficiency_upper"],
            color=p[0].get_color(),
            alpha=0.2,
        )

    plt.xlabel("p (GeV/c)")
    plt.ylabel("Efficiency")
    plt.ylim([0, 1.1])
    plt.title(f"{target_name} classification")
    plt.legend(loc="lower left")
    if save_dir is not None:
        plt.savefig(f"{save_dir}/p_efficiency_optimized_threshold.png")
    plt.show()
    plt.close()