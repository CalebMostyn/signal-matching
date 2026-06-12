import os

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np 

from signal_matcher.match import ResultSet

def calculate_average_compute_time(data: ResultSet) -> float:
    if not len(data.data) > 0:
        return 0.0

    sum: float = 0
    for _, result in data.data.items():
        sum += result.time_to_compute
    return sum / len(data.data)

def calculate_accuracy(ground_truth: ResultSet, data: ResultSet) -> float:
    total: int = 0
    correct: int = 0
    for gt_sample, gt_result in ground_truth.data.items():
        # this could be a little smarter
        for sample, result in data.data.items():
            if sample.name == gt_sample.name:
                total += 2
                if (result.best_match.reference.name ==
                        gt_result.best_match.reference.name):
                    correct += 1
                if (result.second_best_match.reference.name ==
                        gt_result.second_best_match.reference.name):
                    correct += 1
    return correct / total

def plot_compute_times(data: dict[str, ResultSet], filepath: str = '') -> None:
    labels = []
    averages = []
    for name, results in data.items():
        labels.append(name.split('.')[0])
        averages.append(calculate_average_compute_time(results) / 1_000_000)

    sorted_pairs = sorted(zip(averages, labels))
    averages, labels = zip(*sorted_pairs)


    fig, ax = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    bars = ax.bar(labels, averages, color=colors)
    ax.bar_label(bars, padding=3)
    
    ax.set_ylim(0, max(averages) * 1.2)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Average Compute Time (ms)")
    plt.title("Average Compute Time per Method (ms)")
    plt.tight_layout()
    
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(filepath)
    else:
        plt.show()
    plt.close(fig)

def plot_accuracy(ground_truth: (str, ResultSet), data: dict[str, ResultSet],
                  filepath: str = '') -> None:
    labels = []
    accuracies = []
    for name, results in data.items():
        labels.append(name.split('.')[0])
        accuracies.append(calculate_accuracy(ground_truth[1], results) * 100)

    sorted_pairs = sorted(zip(accuracies, labels))
    accuracies, labels = zip(*sorted_pairs)


    fig, ax = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    bars = ax.bar(labels, accuracies, color=colors)
    ax.bar_label(bars, padding=3)
    
    ax.set_ylim(0, max(accuracies) * 1.2)
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Matching Accuracy vs 'Ground Truth' {ground_truth[0]}")
    plt.ylabel("Accuracy (%)")
    plt.tight_layout()
    
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(filepath)
    else:
        plt.show()
    plt.close(fig)

if __name__ == "__main__":
    # load all results from CSVs
    data: dict[str, ResultSet] = {}
    with os.scandir('results/') as files:
        for file in files:
            if file.is_file():
                results: ResultSet = ResultSet()
                results.from_csv(file.path)
                data[os.path.basename(file.path)] = results

    plot_compute_times(data, 'plots/average_compute_time.png')
    plot_accuracy(('nrmse', data['nrmse.csv']), data, 'plots/accuracy.png')
