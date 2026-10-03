import matplotlib.pyplot as plt
import numpy as np

def plot_tumour(x_df, y_df):
    """Plots a sample tumourous slice from split arrays."""
    if isinstance(y_df, np.ndarray):
        tumour_idx = int(np.where(y_df == 1)[0][0])
    else:
        tumour_idx = int(np.where(y_df.to_numpy() == 1)[0][0])

    sample_img = x_df[tumour_idx]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    for ax, channel in zip(axes[:3], [sample_img[:, :, 0], sample_img[:, :, 1], sample_img[:, :, 2]]):
        ax.imshow(channel, cmap="gray")
        ax.axis("off")

    fig.suptitle(f"Tumourous sample index: {tumour_idx}")
    plt.tight_layout()
    plt.show()

def plot_training_history(history, accuracy_metric, loss_metric):
    """Plots the training history for given accuracy and loss metric
    Args:
        history: model history
        accuracy_metric (string): accuracy metric
        loss_metric (string): loss metric"""
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history.history[f"val_{accuracy_metric}"]) + 1)
    best_epoch = np.argmax(history.history[f"val_{accuracy_metric}"]) + 1
    plot_history(history, axes, epochs, best_epoch, 0, accuracy_metric)
    plot_history(history, axes, epochs, best_epoch, 1, loss_metric)
    plt.show()
    
def plot_history(history, axes, epochs, best_epoch, axes_index, metric):
    """Plots the history for the training and validation sets from model training"""
    axes[axes_index].plot(epochs, history.history[metric], label=f"Training {metric}")
    axes[axes_index].plot(epochs, history.history[f"val_{metric}"], label=f"Validation {metric}")
    axes[axes_index].axvline(best_epoch, color="red", linestyle="--", label=f"Best Epoch: {best_epoch}")
    axes[axes_index].set_title(f"Training vs. Validation {metric}")
    axes[axes_index].set_xlabel("Epoch")
    axes[axes_index].set_ylabel(metric)
    axes[axes_index].legend()
    axes[axes_index].grid(True, alpha=0.3)