"""
Utils for the training library

author: Louca Malerba
"""
# Standard imports
import os

# Imports
import torch
import numpy as np
import tqdm as tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


######################################################################
# training loop
######################################################################
def train(model, loader, f_loss, optimizer, device, class_names=None,
          grad_accum_steps=1, grad_clip_norm=None):
    """
    Train one epoch of the model
    """
    
    model.train()

    total_loss = 0
    total_cls_loss = 0
    total_seg_loss = 0
    num_samples = 0
    num_correct = 0
    all_preds = []
    all_targets = []
    all_seg_probs = []   # for Dice / IoU
    all_seg_masks = []   # for Dice / IoU
    all_probs = []
    optimizer.zero_grad()
    pbar = tqdm.tqdm(loader, desc="  Train", leave=False)
    for step_idx, batch in enumerate(pbar):

        # Support both (inputs, targets) and (inputs, targets, masks)
        if len(batch) == 3:
            inputs, targets, masks = batch
            inputs  = inputs.to(device)
            targets = targets.to(device)
            masks   = masks.to(device)
        else:
            inputs, targets = batch
            inputs  = inputs.to(device)
            targets = targets.to(device)
            masks   = None

        # Forward pass
        outputs = model(inputs)

        # Support single-output models and dual-head (cls, seg) models
        if isinstance(outputs, tuple):
            cls_logits, seg_logits = outputs
            loss_out = f_loss(cls_logits, seg_logits, targets, masks)
            # f_loss returns (total, cls_loss, seg_loss)
            loss, cls_l, seg_l = loss_out
            total_cls_loss += inputs.shape[0] * cls_l.item()
            total_seg_loss += inputs.shape[0] * seg_l.item()
            # Accumulate for seg metrics (detach + CPU to save GPU memory)
            all_seg_probs.append(torch.sigmoid(seg_logits).detach().cpu())
            if masks is not None:
                all_seg_masks.append(masks.detach().cpu())
            else:
                all_seg_masks.append(torch.zeros_like(seg_logits).cpu())
        else:
            cls_logits = outputs
            probs = torch.softmax(cls_logits, dim=1)[:, 1]  # P(positive class), assumes binary
            all_probs.append(probs.detach().cpu())
            loss = f_loss(cls_logits, targets)
            total_cls_loss += inputs.shape[0] * loss.item()

        # Scale loss for gradient accumulation
        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()

        # Step optimizer every grad_accum_steps, or at the last batch
        if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(loader):
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=grad_clip_norm,
                )
            optimizer.step()
            optimizer.zero_grad()

        # Update the metrics (use unscaled loss for logging)
        total_loss += inputs.shape[0] * loss.item()
        num_samples += inputs.shape[0]

        # Accuracy (always based on cls_logits)
        preds = cls_logits.argmax(dim=1)
        num_correct += (preds == targets).sum().item()
        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())
        all_probs.append(probs.detach().cpu())  

        acc = num_correct / num_samples
        pbar.set_postfix(loss=f"{total_loss/num_samples:.3f}", acc=f"{100*acc:.1f}%")

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy() 

    accuracy = num_correct / num_samples if num_samples > 0 else 0.0
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    avg_cls_loss = total_cls_loss / num_samples if num_samples > 0 else 0.0
    avg_seg_loss = total_seg_loss / num_samples if num_samples > 0 else 0.0

    # Compute sklearn metrics
    labels = sorted(np.unique(np.concatenate([all_preds, all_targets])))
    f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    prec_macro = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    rec_macro = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    f1_per = f1_score(all_targets, all_preds, average=None, labels=labels, zero_division=0)
    try:                                                    
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        # e.g. only one class present in this epoch's predictions/targets
        auc = float("nan")

    seg_metrics = _seg_dice_iou(all_seg_probs, all_seg_masks) if all_seg_probs else {"dice": None, "iou": None}

    return {
        "loss": avg_loss,
        "cls_loss": avg_cls_loss,
        "seg_loss": avg_seg_loss,
        "seg_dice": seg_metrics["dice"],
        "seg_iou":  seg_metrics["iou"],
        "accuracy": accuracy,
        "balanced_accuracy": bal_acc,
        "auc": auc,
        "f1_macro": f1_macro,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_per_class": {(class_names[i] if class_names and i < len(class_names) else str(i)): float(f) for i, f in zip(labels, f1_per)},
    }

######################################################################
# Intra-training test funciton
######################################################################

def test(model, loader, f_loss, device, class_names=None):
    """
    Test a model over the loader.
    """

    model.eval()

    total_loss = 0
    total_cls_loss = 0
    total_seg_loss = 0
    num_samples = 0
    num_correct = 0
    all_preds = []
    all_targets = []
    all_seg_probs = []   # for Dice / IoU
    all_seg_masks = []   # for Dice / IoU
    all_probs = []

    pbar = tqdm.tqdm(loader, desc="  Valid", leave=False)
    with torch.no_grad():
        for batch in pbar:

            # Support both (inputs, targets) and (inputs, targets, masks)
            if len(batch) == 3:
                inputs, targets, masks = batch
                inputs  = inputs.to(device)
                targets = targets.to(device)
                masks   = masks.to(device)
            else:
                inputs, targets = batch
                inputs  = inputs.to(device)
                targets = targets.to(device)
                masks   = None

            # Forward pass
            outputs = model(inputs)

            # Support single-output and dual-head (cls, seg) models
            if isinstance(outputs, tuple):
                cls_logits, seg_logits = outputs
                loss_out = f_loss(cls_logits, seg_logits, targets, masks)
                loss, cls_l, seg_l = loss_out
                total_cls_loss += inputs.shape[0] * cls_l.item()
                total_seg_loss += inputs.shape[0] * seg_l.item()
                all_seg_probs.append(torch.sigmoid(seg_logits).cpu())
                if masks is not None:
                    all_seg_masks.append(masks.cpu())
                else:
                    all_seg_masks.append(torch.zeros_like(seg_logits).cpu())
            else:
                cls_logits = outputs
                loss = f_loss(cls_logits, targets)
                total_cls_loss += inputs.shape[0] * loss.item()

            # Update the metrics
            total_loss += inputs.shape[0] * loss.item()
            num_samples += inputs.shape[0]

            preds = cls_logits.argmax(dim=1)
            probs = torch.softmax(cls_logits, dim=1)[:, 1]
            num_correct += (preds == targets).sum().item()
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_probs.append(probs.cpu())

            acc = num_correct / num_samples
            pbar.set_postfix(loss=f"{total_loss/num_samples:.3f}", acc=f"{100*acc:.1f}%")

    # end with torch.no_grad

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()

    accuracy = num_correct / num_samples if num_samples > 0 else 0.0
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    avg_cls_loss = total_cls_loss / num_samples if num_samples > 0 else 0.0
    avg_seg_loss = total_seg_loss / num_samples if num_samples > 0 else 0.0

    # Compute sklearn metrics for standard predictions
    labels = sorted(np.unique(np.concatenate([all_preds, all_targets])))
    f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    prec_macro = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    rec_macro = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    f1_per = f1_score(all_targets, all_preds, average=None, labels=labels, zero_division=0)
    report = classification_report(
        all_targets, all_preds,
        target_names=class_names,
        zero_division=0,
    )
    cm = confusion_matrix(all_targets, all_preds, labels=labels)
    try:                                          
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = float("nan")

    seg_metrics = _seg_dice_iou(all_seg_probs, all_seg_masks) if all_seg_probs else {"dice": None, "iou": None}

    return {
        "loss": avg_loss,
        "cls_loss": avg_cls_loss,
        "seg_loss": avg_seg_loss,
        "seg_dice": seg_metrics["dice"],
        "seg_iou":  seg_metrics["iou"],
        "accuracy": accuracy,
        "balanced_accuracy": bal_acc,
        "auc": auc,
        "f1_macro": f1_macro,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_per_class": {(class_names[i] if class_names and i < len(class_names) else str(i)): float(f) for i, f in zip(labels, f1_per)},
        "classification_report": report,
        "confusion_matrix": cm,
    }

######################################################################
# Metrics
######################################################################

def _seg_dice_iou(
    all_seg_probs: list[torch.Tensor],
    all_masks: list[torch.Tensor],
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> dict:
    """Compute epoch-level mean Dice and IoU for binary segmentation.

    Args:
        all_seg_probs: list of (B, 1, D, H, W) sigmoid probabilities (CPU tensors).
        all_masks:     list of (B, 1, D, H, W) binary GT masks (CPU tensors).
        threshold:     binarization threshold for predictions.
        smooth:        Laplace smoothing to avoid division by zero.

    Returns:
        dict with keys 'dice' and 'iou' (floats, averaged over all samples).
    """
    dice_scores = []
    iou_scores  = []
    for probs, masks in zip(all_seg_probs, all_masks):
        pred = (probs >= threshold).float()      # (B, 1, D, H, W)
        gt   = (masks  >= 0.5).float()
        # Flatten spatial dims → (B, N)
        pred_f = pred.view(pred.shape[0], -1)
        gt_f   = gt.view(gt.shape[0], -1)
        inter  = (pred_f * gt_f).sum(dim=1)
        sum_pg = pred_f.sum(dim=1) + gt_f.sum(dim=1)
        union  = sum_pg - inter
        dice_scores.append(((2.0 * inter + smooth) / (sum_pg  + smooth)))
        iou_scores.append( ((       inter + smooth) / (union   + smooth)))
    all_dice = torch.cat(dice_scores)  # (N_samples,)
    all_iou  = torch.cat(iou_scores)
    return {
        "dice": float(all_dice.mean()),
        "iou":  float(all_iou.mean()),
    }

######################################################################
# General utils
######################################################################

##### Logging utils
def generate_unique_logpath(logdir, raw_run_name):
    """
    Generate a unique directory name
    Argument:
        logdir: the prefix directory
        raw_run_name(str): the base name
    Returns:
        log_path: a non-existent path like logdir/raw_run_name_xxxx
                  where xxxx is an int
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1

def make_logdir(logging_config, logname):
    logdir = generate_unique_logpath(logging_config["logdir"], logname)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    return logdir

##### Model checkpointing
class ModelCheckpoint(object):
    """
    Early stopping callback
    """

    def __init__(
        self,
        model: torch.nn.Module,
        savepath,
        min_is_best: bool = True,
    ) -> None:
        self.model = model
        self.savepath = savepath
        self.best_score = None
        if min_is_best:
            self.is_better = self.lower_is_better
        else:
            self.is_better = self.higher_is_better

    def lower_is_better(self, score):
        return self.best_score is None or score < self.best_score

    def higher_is_better(self, score):
        return self.best_score is None or score > self.best_score

    def update(self, score):
        if self.is_better(score):
            torch.save(self.model.state_dict(), self.savepath)
            self.best_score = score
            return True
        return False

##### Training curves
def _plot_single_curve(ax_loss, epochs, train_loss, val_loss, train_bal_acc, val_bal_acc, title):
    """
    generated by Claude Opus 4.6
    """
     
    """Helper: draw loss + balanced accuracy on a dual-axis subplot."""
    color_tl = "#1f77b4"  # blue
    color_vl = "#ff7f0e"  # orange
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.plot(epochs, train_loss, color=color_tl, linestyle="-", label="Train Loss")
    ax_loss.plot(epochs, val_loss, color=color_vl, linestyle="-", label="Val Loss")
    ax_loss.tick_params(axis="y")
    ax_loss.grid(True, alpha=0.3)

    ax_acc = ax_loss.twinx()
    color_ta = "#2ca02c"  # green
    color_va = "#d62728"  # red
    ax_acc.set_ylabel("Balanced Accuracy (%)")
    ax_acc.plot(epochs, [100 * v for v in train_bal_acc], color=color_ta, linestyle="--", label="Train BalAcc")
    ax_acc.plot(epochs, [100 * v for v in val_bal_acc], color=color_va, linestyle="--", label="Val BalAcc")
    ax_acc.tick_params(axis="y")

    # Combined legend
    lines1, labels1 = ax_loss.get_legend_handles_labels()
    lines2, labels2 = ax_acc.get_legend_handles_labels()
    ax_loss.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax_loss.set_title(title)

def plot_training_curves(history, savepath):
    """
    generated by Claude Opus 4.6
    """
    n = len(history["train_loss"])
    epochs_all = list(range(1, n + 1))
    has_lr = "lr" in history and len(history["lr"]) > 0

    if has_lr:
        fig, axes = plt.subplots(2, 2, figsize=(18, 12),
                                 gridspec_kw={"height_ratios": [2, 1]})
        ax_top = axes[0]   # row 0: loss+acc plots
        ax_bot = axes[1]   # row 1: LR plot
    else:
        fig, ax_top = plt.subplots(1, 2, figsize=(18, 6))

    # --- Top-left: all epochs ---
    _plot_single_curve(
        ax_top[0], epochs_all,
        history["train_loss"], history["val_loss"],
        history["train_bal_acc"], history["val_bal_acc"],
        title="Training Curves (all epochs)",
    )

    # --- Top-right: from epoch 2 onwards ---
    if n >= 2:
        epochs_skip = list(range(2, n + 1))
        _plot_single_curve(
            ax_top[1], epochs_skip,
            history["train_loss"][1:], history["val_loss"][1:],
            history["train_bal_acc"][1:], history["val_bal_acc"][1:],
            title="Training Curves (from epoch 2)",
        )
    else:
        ax_top[1].set_title("Training Curves (from epoch 2)\n(waiting for epoch 2\u2026)")
        ax_top[1].grid(True, alpha=0.3)

    # --- Bottom: Learning Rate ---
    if has_lr:
        # Merge bottom-left and bottom-right into one wide subplot
        ax_bot[0].remove()
        ax_bot[1].remove()
        ax_lr = fig.add_subplot(2, 1, 2)
        ax_lr.plot(epochs_all, history["lr"], color="#9467bd", linewidth=1.5)
        ax_lr.set_xlabel("Epoch")
        ax_lr.set_ylabel("Learning Rate")
        ax_lr.set_title("Learning Rate Schedule")
        ax_lr.grid(True, alpha=0.3)
        ax_lr.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    fig.tight_layout()
    fig.savefig(str(savepath), dpi=150)
    plt.close(fig)
