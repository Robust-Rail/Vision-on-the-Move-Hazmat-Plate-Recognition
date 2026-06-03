# src/un_detector/training/trainers.py
import contextlib
import io
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Optional/lazy imports for pycocotools
_pycocotools_available = False
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    _pycocotools_available = True
except ImportError:
    pass

from un_detector.training.train_utils import create_directory, save_epoch_data


class FasterRCNNTrainer:
    """
    Trainer class for fine-tuning Faster R-CNN object detection models.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        device: torch.device,
        scaler: Optional[GradScaler] = None,
        checkpoint_dir: str = "data/models",
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.scaler = scaler if scaler is not None else GradScaler()
        self.checkpoint_dir = checkpoint_dir
        self.active_dir: Optional[str] = None

    def train_one_epoch(self, data_loader: DataLoader) -> Tuple[float, float, float, float, float]:
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_classifier_loss = 0.0
        total_box_reg_loss = 0.0
        total_objectness_loss = 0.0
        total_rpn_box_reg_loss = 0.0

        progress_bar = tqdm(data_loader, desc="Training", leave=True)

        for images, targets in progress_bar:
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Wrap the forward pass in autocast for mixed precision
            with autocast():
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            self.scaler.scale(losses).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate losses
            total_loss += losses.item()
            total_classifier_loss += loss_dict["loss_classifier"].item()
            total_box_reg_loss += loss_dict["loss_box_reg"].item()
            total_objectness_loss += loss_dict["loss_objectness"].item()
            total_rpn_box_reg_loss += loss_dict["loss_rpn_box_reg"].item()

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "Loss": f"{losses.item():.4f}",
                    "Classifier": f"{loss_dict['loss_classifier'].item():.4f}",
                    "BoxReg": f"{loss_dict['loss_box_reg'].item():.4f}",
                }
            )

        n_batches = len(data_loader)
        return (
            total_loss / n_batches,
            total_classifier_loss / n_batches,
            total_box_reg_loss / n_batches,
            total_objectness_loss / n_batches,
            total_rpn_box_reg_loss / n_batches,
        )

    def _convert_to_coco_format(self, outputs: List[Dict[str, torch.Tensor]], image_ids: List[int]) -> List[Dict[str, Any]]:
        coco_results = []
        for output, image_id in zip(outputs, image_ids):
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                coco_results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                        "score": float(score),
                    }
                )
        return coco_results

    def validate(self, data_loader: DataLoader, coco_gt: Any) -> List[float]:
        """
        Validate the model and compute COCO evaluation metrics.
        """
        if not _pycocotools_available:
            raise ImportError(
                "pycocotools is required for validation. Install it via 'pip install pycocotools'."
            )

        self.model.eval()
        results = []

        progress_bar = tqdm(data_loader, desc="Validation", leave=True)

        with torch.no_grad():
            for images, targets in progress_bar:
                images = list(image.to(self.device) for image in images)
                outputs = self.model(images)

                image_ids = [target["image_id"].item() for target in targets]
                coco_results = self._convert_to_coco_format(outputs, image_ids)
                results.extend(coco_results)

                progress_bar.set_postfix({"Processed": len(results)})

        if not results:
            print("⚠️ No predictions generated during validation. Skipping evaluation.")
            return [0.0] * 6  # Return zeroed metrics if no predictions

        # Suppress COCOeval stdout output to avoid cluttering logs
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(results)
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        # coco_eval.stats contains [mAP@0.50:0.95, mAP@0.50, mAP@0.75, mAP_small, mAP_medium, mAP_large]
        return coco_eval.stats.tolist()

    def fit(
        self,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        coco_val_gt: Any,
    ) -> Dict[str, Any]:
        """
        Run the complete fine-tuning pipeline.
        """
        if self.active_dir is None:
            self.active_dir = create_directory(self.checkpoint_dir)

        print(f"🏋️ Starting Faster R-CNN training for {epochs} epochs...")
        print(f"💾 Checkpoints will be saved in: {self.active_dir}")

        train_metrics_list = []
        best_val_map = float("-inf")

        for epoch in range(epochs):
            start_time = time.time()

            # Train one epoch
            train_loss, train_clf_loss, train_box_loss, train_obj_loss, train_rpn_loss = (
                self.train_one_epoch(train_loader)
            )

            # Validate
            val_metrics = self.validate(val_loader, coco_val_gt)
            val_map = val_metrics[0]  # mAP@0.50:0.95

            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(int(elapsed_time), 60)

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Save metrics
            epoch_data = {
                "epoch": epoch + 1,
                "time_elapsed": (minutes, seconds),
                "learning_rate": current_lr,
                "train_loss": train_loss,
                "classifier_loss": train_clf_loss,
                "box_reg_loss": train_box_loss,
                "objectness_loss": train_obj_loss,
                "rpn_box_reg_loss": train_rpn_loss,
                "val_metrics": val_metrics,
            }
            train_metrics_list.append(epoch_data)

            # Print summary
            print(f"📊 Epoch {epoch + 1} | ⏳ Time: {minutes}m {seconds}s | 🔄 LR: {current_lr:.6f}")
            print(f"📉 Train Loss: {train_loss:.4f} | 🎯 Classifier: {train_clf_loss:.4f} | 📦 Box Reg: {train_box_loss:.4f}")
            print(f"🧪 mAP | 🟢 mAP@IoU=0.50:0.95: {val_metrics[0]:.4f} | 🔵 mAP@IoU=0.50: {val_metrics[1]:.4f}")

            # Log to text file
            save_epoch_data(self.active_dir, epoch_data)

            # Step scheduler
            self.lr_scheduler.step()

            # Save checkpoints
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_map": val_map,
                "train_metrics_list": train_metrics_list,
            }
            torch.save(checkpoint, os.path.join(self.active_dir, "latest_model.pth"))

            if val_map > best_val_map:
                best_val_map = val_map
                torch.save(checkpoint, os.path.join(self.active_dir, "best_model.pth"))
                print(f"🏆 New best model saved with validation mAP: {val_map:.4f}")

        return {
            "metrics": train_metrics_list,
            "best_map": best_val_map,
            "save_dir": self.active_dir,
        }
