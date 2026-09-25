import matplotlib.pyplot as plt
import numpy as np

def plot_tumor(df):
    """Plots a sample of each MRI image for a tumourous slice"""
    sample = next(s for s in df if s["is_tumorous"])
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, key in zip(axes[:3], ["t1", "t1c", "t2"]):
        ax.imshow(sample[key], cmap="gray")
        ax.set_title(key.upper())
        ax.axis("off")

    axes[3].imshow(sample["t1c"], cmap="gray")
    axes[3].imshow(np.array(sample["tumor_mask"]), cmap="jet", alpha=0.4, vmin=0, vmax=4)
    axes[3].set_title("Tumor Mask Overlay")
    axes[3].axis("off")

    fig.suptitle(
        f"{sample['volume_id']} | Slice {sample['slice_id']} | "
        f"{sample['tumor_type']} (WHO grade {sample['who_grade']})"
    )
    plt.tight_layout()
    plt.show()

def plot_training_history(history, accuracy_metric, loss_metric):
    """Plots the training history for given accuracy and loss metric
    Args:
        history: model history
        accuracy_metric (string): accuracy metric
        loss_metric (string): loss metric"""
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history.history[accuracy_metric]) + 1)
    best_epoch = np.argmin(history.history[loss_metric]) + 1
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