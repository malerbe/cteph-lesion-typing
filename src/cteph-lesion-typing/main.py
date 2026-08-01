# Standard imports
import logging
import sys
import yaml
import pathlib

# External imports
import torch

# Local imports
from . import data
from . import models
from . import optim
from . import utils

CLASS_NAMES = ["acute", "chronic"]


######################################################################
# train and helper functions
######################################################################
def train(config):
    """
    Main entry point to train based on the provided config.
    """
    if config["data"]["task"] in ["classification", "multihead_classification"]:
        _train_patch_classification(config)


def _train_patch_classification(config):
    """
    Train a classification model based on patches
    """

    logging.info("Training patch classification model...")

    ######################################
    # Config
    ######################################
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    data_config = config["data"]
    model_config = config["model"]
    loss_config = config["loss"]
    logging_config = config["logging"]
    
    ######################################
    # Dataloaders
    ######################################
    logging.info("= Building dataloaders...")

    train_loader, valid_loader, input_size, num_classes = data.get_dataloaders(data_config, use_cuda)

    ######################################
    # Model
    ######################################
    logging.info("= Building model...")

    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)

    ######################################
    # Loss
    ######################################
    logging.info("= Building loss...")
    
    loss = optim.get_loss(loss_config)

    ######################################
    # Logging
    ######################################
    logging.info("= Configuring logging...")
    logname = model_config["class"]
    logdir = utils.make_logdir(logging_config, logname)

    logging.info(f"Will be logging into {logdir}")

    ##### Copy the config file into the logdir
    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as file:
        yaml.dump(config, file)

    ######################################
    # Model checkpoints
    ######################################
    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=True
    )

    ######################################
    # Phases
    ######################################
    phases = []
    if "phase1" in config:
        phases.append(("Phase 1 (frozen)", config["phase1"], True))
    else:
        raise ValueError("At least on phase must be configured in the training file. phase1 was not found")
    if "phase2" in config:
        phases.append(("Phase 2 (unfrozen)", config["phase2"], False))
    # could be more than 2 phases, but only support 2 for now 

    ######################################
    # Training
    ######################################
    # History for training curves
    curve_history = {"train_loss": [], "val_loss": [], "train_bal_acc": [], "val_bal_acc": [], "lr": []}

    global_epoch = 0
    for phase_name, phase_cfg, freeze_backbone in phases:
        nepochs = phase_cfg["nepochs"]
        
        # freeze or unfreeze backbone
        if hasattr(model, "freeze_backbone"):
            if freeze_backbone:
                model.freeze_backbone()
                logging.info(f"\n{'='*60}")
                logging.info(f"{phase_name}: backbone FROZEN")
            else:
                model.unfreeze_backbone()
                logging.info(f"\n{'='*60}")
                logging.info(f"{phase_name}: backbone UNFROZEN")
            logging.info(f"{'='*60}")
        else:
            logging.info(f"\n{'='*60}")
            logging.info(f"{phase_name} | Model not supporting freezing/unfreezing")
            logging.info(f"{'='*60}")

        ################
        # Optimizer + scheduler + grad acc + clipping for this phase
        ################
        phase_optim_cfg = phase_cfg["optim"]
        optimizer = optim.get_optimizer(phase_optim_cfg, filter(lambda p: p.requires_grad, model.parameters()))
        logging.info(f"Optimizer: {phase_optim_cfg['algo']} (lr={phase_optim_cfg['params'].get('lr', '?')})")
        logging.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        logging.info(f"Total params:     {sum(p.numel() for p in model.parameters()):,}")

        scheduler_cfg = phase_cfg.get("scheduler", None)
        scheduler = optim.get_scheduler(scheduler_cfg, optimizer)
        if scheduler is not None:
            logging.info(f"Scheduler: {scheduler_cfg['algo']} {scheduler_cfg.get('params', {})}")

        grad_accum_steps = phase_cfg.get("grad_accum_steps", 1)
        grad_clip_norm = phase_cfg.get("grad_clip_norm", None)
        if grad_accum_steps > 1:
            logging.info(f"Gradient accumulation: {grad_accum_steps} steps")
        if grad_clip_norm is not None:
            logging.info(f"Gradient clipping: max_norm={grad_clip_norm}")

        ################
        # Training loop for this phase
        ################
        for e in range(nepochs):
            logging.info(f"\n###### {phase_name} | Epoch {e+1}/{nepochs} (global {global_epoch+1}) ######")
            
            # train 1 epoch
            train_results = utils.train(model, train_loader, loss, optimizer, device,
                                        class_names=CLASS_NAMES,
                                        grad_accum_steps=grad_accum_steps,
                                        grad_clip_norm=grad_clip_norm)

            # test
            test_results = utils.test(model, valid_loader, loss, device,
                                      class_names=CLASS_NAMES)

            # update checkpoint
            updated = model_checkpoint.update(test_results["loss"])

            # Update and save training curves
            curve_history["train_loss"].append(train_results["loss"])
            curve_history["val_loss"].append(test_results["loss"])
            curve_history["train_bal_acc"].append(train_results["balanced_accuracy"])
            curve_history["val_bal_acc"].append(test_results["balanced_accuracy"])
            curve_history["lr"].append(optimizer.param_groups[0]["lr"])
            utils.plot_training_curves(curve_history, logdir / "training_curves.png")

            
            ################
            # Log metrics
            ################
        
            # train metrics
            train_f1_parts = " | ".join(f"{k}: {v:.3f}" for k, v in train_results["f1_per_class"].items())
            logging.info(
                "  Train loss: %.3f | Acc: %.2f%% | BalAcc: %.2f%% | F1: %.3f | Prec: %.3f | Rec: %.3f"
                % (train_results["loss"], 100.0 * train_results["accuracy"], 100.0 * train_results["balanced_accuracy"], train_results["f1_macro"], train_results["precision_macro"], train_results["recall_macro"])
            )
            if train_results.get("seg_loss", 0.0) > 0.0:
                seg_dice_str = f"{train_results['seg_dice']:.4f}" if train_results.get("seg_dice") is not None else "n/a"
                seg_iou_str  = f"{train_results['seg_iou']:.4f}"  if train_results.get("seg_iou")  is not None else "n/a"
                logging.info(
                    "  Train cls_loss: %.4f | seg_loss: %.4f | seg_Dice: %s | seg_IoU: %s"
                    % (train_results["cls_loss"], train_results["seg_loss"], seg_dice_str, seg_iou_str)
                )
            logging.info(f"  Train F1/class: {train_f1_parts}")
        
            # test metrics
            test_f1_parts = " | ".join(f"{k}: {v:.3f}" for k, v in test_results["f1_per_class"].items())
            logging.info(
                "  Val   loss: %.3f | Acc: %.2f%% | BalAcc: %.2f%% | F1: %.3f | Prec: %.3f | Rec: %.3f %s"
                % (test_results["loss"], 100.0 * test_results["accuracy"], 100.0 * test_results["balanced_accuracy"],
                    test_results["f1_macro"], test_results["precision_macro"], test_results["recall_macro"],
                    "[>> BETTER <<]" if updated else "")
            )
            logging.info(f"  Val   F1/class: {test_f1_parts}")
            if test_results.get("seg_loss", 0.0) > 0.0:
                val_dice_str = f"{test_results['seg_dice']:.4f}" if test_results.get("seg_dice") is not None else "n/a"
                val_iou_str  = f"{test_results['seg_iou']:.4f}"  if test_results.get("seg_iou")  is not None else "n/a"
                logging.info(
                    "  Val   cls_loss: %.4f | seg_loss: %.4f | seg_Dice: %s | seg_IoU: %s"
                    % (test_results["cls_loss"], test_results["seg_loss"], val_dice_str, val_iou_str)
                )

            # Update dashboard
             # Update the dashboard
            metrics = {
                "epoch": global_epoch,
                "train_loss": train_results["loss"],
                "train_acc": train_results["accuracy"],
                "train_bal_acc": train_results["balanced_accuracy"],
                "train_f1_macro": train_results["f1_macro"],
                "train_precision_macro": train_results["precision_macro"],
                "train_recall_macro": train_results["recall_macro"],
                "val_loss": test_results["loss"],
                "val_acc": test_results["accuracy"],
                "val_bal_acc": test_results["balanced_accuracy"],
                "val_f1_macro": test_results["f1_macro"],
                "val_precision_macro": test_results["precision_macro"],
                "val_recall_macro": test_results["recall_macro"],
                "val_acc_tta": test_results.get("accuracy_tta", None),
                "val_bal_acc_tta": test_results.get("balanced_accuracy_tta", None),
                "val_f1_macro_tta": test_results.get("f1_macro_tta", None),
                "val_precision_macro_tta": test_results.get("precision_macro_tta", None),
                "val_recall_macro_tta": test_results.get("recall_macro_tta", None),
            }

            for k, v in train_results["f1_per_class"].items():
                metrics[f"train_f1_{k}"] = v
            for k, v in test_results["f1_per_class"].items():
                metrics[f"val_f1_{k}"] = v

            ################
            # Scheduler step
            ################
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(test_results["loss"])
                else:
                    scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            logging.info(f"  LR: {current_lr:.2e}")

            global_epoch += 1

    logging.info(f"\nTraining complete. Best model saved to {model_checkpoint.savepath}")


    
if __name__ == "__main__":
    # Read arguments
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|evaluate>")
        sys.exit(-1)

    logging.info("Loading {} configuration file".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))

    command = sys.argv[2]
    eval(f"{command}(config)")
