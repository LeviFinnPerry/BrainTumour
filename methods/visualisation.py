import matplotlib.pyplot as plt
import numpy as np

def plot_tumour(x_df, y_df):
    """Plots a sample tumourous slice from split arrays for classification."""
    tumour_idx = int(np.where(y_df == 1)[0][0])

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

def compute_iou(pred, true):
    """Computes the IoU in the segmentation prediction"""
    pred = pred.astype(bool)
    true = true.astype(bool)
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return intersection / union if union > 0 else 1.0    
    
def plot_segmentation_prediction(x, y_true, model, n_samples=4, threshold=0.5, binary=True):
    """Plots each MRI image (t1, t1c, t2), the tumour_mask and the prediction with the computed IoU"""
    indices = np.random.choice(len(x), size=n_samples, replace=False)
    y_pred = model.predict(x[indices])
    if binary:
        y_pred_outcome = (y_pred > threshold).astype(np.float32)
    
    _, axes = plt.subplots(n_samples, 5, figsize=(15, n_samples * 3))
    for i, idx in enumerate(indices):
        iou = compute_iou(y_pred[i].squeeze(), y_true[idx].squeeze())
        axes[i, 0].imshow(x[idx][..., 0], cmap="grey")
        axes[i, 0].set_title("t1")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(x[idx][..., 1], cmap="grey")
        axes[i, 1].set_title("t1c")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(x[idx][..., 2], cmap="grey")
        axes[i, 2].set_title("t2")
        axes[i, 2].axis("off")
        
        axes[i, 3].imshow(y_true[idx].squeeze(), cmap="grey")
        axes[i, 3].set_title("Tumour Mask")
        axes[i, 3].axis("off")
        
        axes[i, 4].imshow(y_pred_outcome[i].squeeze(), cmap="grey")
        axes[i, 4].set_title(f"Prediction\nIoU: {iou:.3f}")
        axes[i, 4].axis("off")
        
    plt.tight_layout()
    plt.show()
        
        