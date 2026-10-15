import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, jaccard_score
from PIL import Image


def segmentation_metrics(
    y_true,
    y_pred,
    num_classes=None,
    ignore_index=None,
):
    """
    Compute common segmentation metrics for semantic segmentation.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    if ignore_index is not None:
        valid = y_true != ignore_index
        y_true = y_true[valid]
        y_pred = y_pred[valid]

    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max()) + 1)

    labels = list(range(num_classes))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class_precision = precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class_recall = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class_f1 = f1_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class_iou = jaccard_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    overall_accuracy = (y_true == y_pred).mean()

    metrics = {
        "confusion_matrix": cm,
        "overall_accuracy": float(overall_accuracy),
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "per_class_iou": per_class_iou,
        "mean_precision": float(np.mean(per_class_precision)),
        "mean_recall": float(np.mean(per_class_recall)),
        "mean_f1": float(np.mean(per_class_f1)),
        "mean_iou": float(np.mean(per_class_iou)),
    }

    return metrics


def binary_foreground_metrics(y_true, y_pred, background_class=0, ignore_index=None):
    """
    Collapse multiclass segmentation into binary:
        background_class -> 0
        all other classes -> 1
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

    if ignore_index is not None:
        valid = y_true != ignore_index
        y_true = y_true[valid]
        y_pred = y_pred[valid]

    y_true_bin = (y_true != background_class).astype(np.uint8)
    y_pred_bin = (y_pred != background_class).astype(np.uint8)

    cm = confusion_matrix(y_true_bin.reshape(-1), y_pred_bin.reshape(-1), labels=[0, 1])

    precision = precision_score(y_true_bin.reshape(-1), y_pred_bin.reshape(-1), zero_division=0)
    recall = recall_score(y_true_bin.reshape(-1), y_pred_bin.reshape(-1), zero_division=0)
    f1 = f1_score(y_true_bin.reshape(-1), y_pred_bin.reshape(-1), zero_division=0)
    iou = jaccard_score(y_true_bin.reshape(-1), y_pred_bin.reshape(-1), zero_division=0)
    accuracy = (y_true_bin == y_pred_bin).mean()

    return {
        "confusion_matrix": cm,
        "accuracy": float(accuracy),
        "precision_foreground": float(precision),
        "recall_foreground": float(recall),
        "f1_foreground": float(f1),
        "iou_foreground": float(iou),
    }

def binary_fault_metrics(y_true, y_pred, fault_classes=(1, 3, 6), ignore_index=None):
    """
    Collapse multiclass segmentation into binary fault vs non-fault:
        fault_classes (e.g. 1, 3, 6) -> 1 (fault)
        all other classes            -> 0 (non-fault)

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and predicted label maps (any matching shape).
    fault_classes : iterable of int
        Class indices that together make up the "fault" class.
    ignore_index : int or None
        If set, pixels where y_true == ignore_index are excluded.

    Returns
    -------
    dict with confusion matrix and fault-class accuracy/precision/recall/f1/iou.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

    if ignore_index is not None:
        valid = y_true != ignore_index
        y_true = y_true[valid]
        y_pred = y_pred[valid]

    fault_classes = np.asarray(list(fault_classes), dtype=np.int64)

    # 1 where the label is any of the fault classes, 0 otherwise
    y_true_bin = np.isin(y_true, fault_classes).astype(np.uint8)
    y_pred_bin = np.isin(y_pred, fault_classes).astype(np.uint8)

    yt = y_true_bin.reshape(-1)
    yp = y_pred_bin.reshape(-1)

    cm = confusion_matrix(yt, yp, labels=[0, 1])
    precision = precision_score(yt, yp, zero_division=0)
    recall = recall_score(yt, yp, zero_division=0)
    f1 = f1_score(yt, yp, zero_division=0)
    iou = jaccard_score(yt, yp, zero_division=0)
    accuracy = (yt == yp).mean()

    return {
        "confusion_matrix": cm,
        "fault_classes": fault_classes.tolist(),
        "accuracy": float(accuracy),
        "precision_fault": float(precision),
        "recall_fault": float(recall),
        "f1_fault": float(f1),
        "iou_fault": float(iou),
    }

def save_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    # Normalise each row to percentage (true label = row)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct   = np.where(row_sums > 0, cm / row_sums * 100, 0.0)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_pct, interpolation="nearest", vmin=0, vmax=100)
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label("(%)")

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    thresh = 50.0   # fixed threshold at 50% for white/black text
    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            plt.text(
                j, i, f"{cm_pct[i, j]:.1f}%",
                ha="center", va="center",
                color="white" if cm_pct[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def save_confusion_matrix_original(cm, class_names, save_path, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def append_metrics_to_csv(csv_path, row_dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


# Configuration
# borehole_names = ["BDR_C6", "BDR_C2", "BCL11", "BCS_D4","BSE_P26", "BSE_P12", "BFS_B15", "BPF3"]
borehole_names = ["BCL11"]
# encoder_names = ["resnet34", "efficientnet-b4", "mit_b2"]
encoder_names = ["resnet34"]
mode = "final_pred_originalsetting"
# width = 900
num_classes = 8
class_names = [
    "Background", "Tectonic fault", "Bedding", "Fault zone boundary",
    "EDZ", "UNclassified", "Fault gouge", "Induced cracks"
]

gt_dir = "data/general_borehole_testing/annotation"
pred_dir = f"data/general-boreholes-output/{mode}"
output_dir = "data/general_borehole_testing/"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "encoder_compare.csv")

# Optional: toggle confusion matrix saving
SAVE_CONFUSION_MATRICES = False


def evaluate_single(borehole_name, encoder_name):
    """Evaluate a single (borehole, encoder) pair and append metrics to CSV."""
    gt_path = os.path.join(gt_dir, f"{borehole_name}.png")
    pred_path = os.path.join(
        pred_dir,
        # f"{borehole_name}_two_step_pred_celoss_multiclass_trainboth_{encoder_name}_cr.png"
        f"{borehole_name}.png"
    )

    # Skip if files don't exist
    if not os.path.exists(gt_path):
        print(f"[SKIP] GT not found: {gt_path}")
        return
    if not os.path.exists(pred_path):
        print(f"[SKIP] Prediction not found: {pred_path}")
        return

    image_name = os.path.splitext(os.path.basename(gt_path))[0]
    gt = np.array(Image.open(gt_path))
    pred = np.array(Image.open(pred_path))

    metrics = segmentation_metrics(
        gt, pred, num_classes=num_classes, ignore_index=255,
    )
    metrics_bin = binary_foreground_metrics(
        gt, pred, background_class=0, ignore_index=255,
    )
    metrics_fault = binary_fault_metrics(
        gt, pred, fault_classes=[1, 3, 6], ignore_index=255
    )


    # Optional confusion matrices
    if SAVE_CONFUSION_MATRICES:
        multiclass_fig_path = os.path.join(
            output_dir, f"{image_name}_{encoder_name}_multiclass_confusion.png"
        )
        save_confusion_matrix(
            metrics["confusion_matrix"],
            class_names=class_names,
            save_path=multiclass_fig_path,
            title=f"Multiclass Confusion Matrix - {image_name} ({encoder_name})",
        )
        binary_fig_path = os.path.join(
            output_dir, f"{image_name}_{encoder_name}_binary_confusion.png"
        )
        save_confusion_matrix(
            metrics_bin["confusion_matrix"],
            class_names=["background", "foreground"],
            save_path=binary_fig_path,
            title=f"Binary Confusion Matrix - {image_name} ({encoder_name})",
        )

    # Build CSV row
    row = {
        "image_name": image_name,
        "mode": mode,
        # "width": width,
        "overall_accuracy": metrics["overall_accuracy"],
        "mean_precision": metrics["mean_precision"],
        "mean_recall": metrics["mean_recall"],
        "mean_f1": metrics["mean_f1"],
        "mean_iou": metrics["mean_iou"],
        "binary_accuracy": metrics_bin["accuracy"],
        "binary_precision_foreground": metrics_bin["precision_foreground"],
        "binary_recall_foreground": metrics_bin["recall_foreground"],
        "binary_f1_foreground": metrics_bin["f1_foreground"],
        "binary_iou_foreground": metrics_bin["iou_foreground"],
        "fault_accuracy": metrics_fault["accuracy"],
        "fault_precision": metrics_fault["precision_fault"],
        "fault_recall": metrics_fault["recall_fault"],
        "fault_f1": metrics_fault["f1_fault"],
        "fault_iou": metrics_fault["iou_fault"],
    }
    for i in range(num_classes):
        row[f"class_{i}_precision"] = float(metrics["per_class_precision"][i])
        row[f"class_{i}_recall"] = float(metrics["per_class_recall"][i])
        row[f"class_{i}_f1"] = float(metrics["per_class_f1"][i])
        row[f"class_{i}_iou"] = float(metrics["per_class_iou"][i])

    append_metrics_to_csv(csv_path, row)

    # Concise console summary
    print(
        f"[OK] {image_name:20s} | {encoder_name:18s} | "
        f"OA={metrics['overall_accuracy']:.4f} "
        f"mIoU={metrics['mean_iou']:.4f} "
        f"binF1={metrics_bin['f1_foreground']:.4f} "
        f"binIoU={metrics_bin['iou_foreground']:.4f} "
        f"faultF1={metrics_fault['f1_fault']:.4f} "
        f"faultIoU={metrics_fault['iou_fault']:.4f}"
    )


def main():
    print(f"Evaluating {len(borehole_names)} boreholes × {len(encoder_names)} encoders "
          f"= {len(borehole_names) * len(encoder_names)} runs\n")

    for borehole_name in borehole_names:
        for encoder_name in encoder_names:
            try:
                evaluate_single(borehole_name, encoder_name)
            except Exception as e:
                print(f"[ERROR] {borehole_name} / {encoder_name}: {e}")

    print(f"\nAll metrics appended to: {csv_path}")


if __name__ == "__main__":
    main()

# =========================
# Main evaluation
# =========================
# borehole_name = "BCS_D4" # "BPF3" or"BSE_P12" #"BSE_P26" # "BFS_B15_bottom" # "BDR_C6" # "BDR_C2"
# encoder_name =  "efficientnet-b4" # "resnet34" or "efficientnet-b4" or "mit_b2" 
# width = 1440
# gt_path = f"data/general_borehole_testing/annotation/{borehole_name}.png"
# # pred_path = f"data/direct_multi/{borehole_name}_{width}_onlyPEskipfusion_pred_multi_dcn_celoss.png"
# pred_path = f"data/general-boreholes-output/final_pred_encoder_comparison/{borehole_name}_{width}_two_step_pred_celoss_multiclass_trainboth_stride_{width}_{encoder_name}.png"
# # data/general-boreholes-output/final_pred_encoder_comparison/BPF3_1440_two_step_pred_celoss_multiclass_trainboth_stride_1440_resnet34.png
# output_dir = "data/general_borehole_testing/"
# os.makedirs(output_dir, exist_ok=True)

# image_name = os.path.splitext(os.path.basename(gt_path))[0]

# gt = np.array(Image.open(gt_path))
# pred = np.array(Image.open(pred_path))

# num_classes = 8 
# class_names = ["Background","Tectonic fault","Bedding","Fault zone boundary", "EDZ","UNclassified","Fault gouge", "Induced cracks"]#,"Desiccation"

# metrics = segmentation_metrics(
#     gt,
#     pred,
#     num_classes=num_classes,
#     ignore_index=255,
# )

# metrics_bin = binary_foreground_metrics(
#     gt,
#     pred,
#     background_class=0,
#     ignore_index=255,   # or None
# )

# # Save multiclass confusion matrix figure
# # multiclass_fig_path = os.path.join(output_dir, f"{image_name}_multiclass_confusion.png")
# # save_confusion_matrix(
# #     metrics["confusion_matrix"],
# #     class_names=class_names,
# #     save_path=multiclass_fig_path,
# #     title=f"Multiclass Confusion Matrix - {image_name}"
# # )

# # # Save binary confusion matrix figure
# # binary_fig_path = os.path.join(output_dir, f"{image_name}_binary_confusion.png")
# # save_confusion_matrix(
# #     metrics_bin["confusion_matrix"],
# #     class_names=["background", "foreground"],
# #     save_path=binary_fig_path,
# #     title=f"Binary Confusion Matrix - {image_name}"
# # )

# # Build CSV row
# row = {
#     "image_name": image_name,
#     "encoder": encoder_name,  # or "EfficientNet-B4" depending on the model used
#     "overall_accuracy": metrics["overall_accuracy"],
#     "mean_precision": metrics["mean_precision"],
#     "mean_recall": metrics["mean_recall"],
#     "mean_f1": metrics["mean_f1"],
#     "mean_iou": metrics["mean_iou"],
#     "binary_accuracy": metrics_bin["accuracy"],
#     "binary_precision_foreground": metrics_bin["precision_foreground"],
#     "binary_recall_foreground": metrics_bin["recall_foreground"],
#     "binary_f1_foreground": metrics_bin["f1_foreground"],
#     "binary_iou_foreground": metrics_bin["iou_foreground"],
#     # "multiclass_confusion_figure": multiclass_fig_path,
#     # "binary_confusion_figure": binary_fig_path,
# }

# # Add per-class metrics into CSV row
# for i in range(num_classes):
#     row[f"class_{i}_precision"] = float(metrics["per_class_precision"][i])
#     row[f"class_{i}_recall"] = float(metrics["per_class_recall"][i])
#     row[f"class_{i}_f1"] = float(metrics["per_class_f1"][i])
#     row[f"class_{i}_iou"] = float(metrics["per_class_iou"][i])

# csv_path = os.path.join(output_dir, "encoder_metrics.csv")
# append_metrics_to_csv(csv_path, row)

# # Optional console summary
# # print("Saved multiclass confusion matrix to:", multiclass_fig_path)
# # print("Saved binary confusion matrix to:", binary_fig_path)
# print("Appended metrics to:", csv_path)

# print("Binary foreground recall:", metrics_bin["recall_foreground"])
# print("Binary foreground precision:", metrics_bin["precision_foreground"])
# print("Binary foreground F1:", metrics_bin["f1_foreground"])
# print("Binary foreground IoU:", metrics_bin["iou_foreground"])

# print("Overall accuracy:", metrics["overall_accuracy"])
# print("Mean recall:", metrics["mean_recall"])
# print("Mean IoU:", metrics["mean_iou"])

# for i, name in enumerate(class_names):
#     print(
#         f"{name:12s} "
#         f"Precision={metrics['per_class_precision'][i]:.4f} "
#         f"Recall={metrics['per_class_recall'][i]:.4f} "
#         f"F1={metrics['per_class_f1'][i]:.4f} "
#         f"IoU={metrics['per_class_iou'][i]:.4f}"
#     )