"""
DebugLogger - PyTorch Lightning Callback for Comprehensive Training Analytics

This module provides a PyTorch Lightning callback that captures detailed metrics
during training including:
- Epoch-level metrics (loss, accuracy, precision, recall, F1)
- Batch-level metrics (accuracy, loss per batch)
- Class-wise metrics (per-class precision, recall, F1)
- Gradient norms (for detecting vanishing/exploding gradients)
- Learning rates (per epoch)
- Training time (per epoch and total)
- Input data statistics (mean, std, min, max)
- Misclassified samples with images in base64 format

All data is saved to a single comprehensive JSON file with timestamp.
"""

import pytorch_lightning as pl
import torch
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import json
import os
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import time
import hashlib
import matplotlib.pyplot as plt


class DebugLogger(pl.Callback):
    """
    PyTorch Lightning callback for comprehensive training analytics and debugging.
    
    Captures and logs detailed metrics during training in a single JSON file.
    
    Args:
        save_dir (str): Directory to save training logs. Default: './logs'
        distortion_model_path (str | None): Optional ultralytics model checkpoint path
            used to predict distortion labels (e.g. blur/noise/pixelate/jpeg) for
            misclassified images.
        distortion_classes (dict | None): Optional mapping of class indices to label
            strings for the distortion model.
        distortion_conf_threshold (float): Confidence threshold to consider a
            distortion prediction valid (default: 0.01).
    
    Attributes:
        metrics (list): Epoch-level metrics
        batch_metrics (list): Batch-level metrics
        class_metrics (list): Class-wise metrics per epoch
        gradient_norms (list): Gradient norms per epoch
        learning_rates (list): Learning rates per epoch
        epoch_times (list): Training time per epoch
        input_stats (list): Input data statistics per batch
        misclassified_data (list): Misclassified samples with base64 images
    """
    
    def __init__(
        self,
        save_dir='./logs',
        distortion_model_path=None,
        distortion_classes=None,
        distortion_conf_threshold=0.01,
    ):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # dataset metadata
        self.metadata = {}
        self.summary = {}
        # Epoch level metrics
        self.metrics = []
        self.all_predictions = []
        self.all_labels = []
        self.misclassified_data = []
        
        # Batch level metrics
        self.batch_metrics = []
        
        # Class-wise metrics
        self.class_metrics = []
        
        # Timing
        self.epoch_times = []
        self.epoch_start_time = None
        
        # Gradient norms
        self.gradient_norms = []
        
        # Learning rates
        self.learning_rates = []
        
        # Input data statistics
        self.input_stats = []
        
        # Correct predictions per epoch
        self.correct_predictions_per_epoch = []
        # Track misclassification counts by image hash
        self.misclassified_counts = {}

        # Distortion model (optional)
        self.distortion_model_path = distortion_model_path
        self.distortion_model = None
        self.distortion_classes = distortion_classes
        self.distortion_conf_threshold = distortion_conf_threshold
        if self.distortion_model_path:
            try:
                from ultralytics import YOLO

                self.distortion_model = YOLO(self.distortion_model_path)
                if self.distortion_classes is None and hasattr(self.distortion_model.model, 'names'):
                    self.distortion_classes = self.distortion_model.model.names
            except Exception as e:
                print(f"⚠️ Failed to load distortion model '{self.distortion_model_path}': {e}")
                self.distortion_model = None
        
        # Batch level metrics
        self.batch_metrics = []
        
        # Class-wise metrics
        self.class_metrics = []
        
        # Timing
        self.epoch_times = []
        self.epoch_start_time = None
        
        # Gradient norms
        self.gradient_norms = []
        
        # Learning rates
        self.learning_rates = []
        
        # Input data statistics
        self.input_stats = []
        
        # Correct predictions per epoch
        self.correct_predictions_per_epoch = []
        # Track misclassification counts by image hash
        self.misclassified_counts = {}
    def on_train_start(self, trainer, pl_module):
        # 1. Programmatically get class names from the dataloader
        # This works for CIFAR10, ImageNet, or any standard FolderDataset
        train_dataloader = trainer.train_dataloader
        dataset = train_dataloader.dataset
        
        if hasattr(dataset, 'classes'):
            self.metadata['classes'] = {i: name for i, name in enumerate(dataset.classes)}
        
        # 2. Get image resolution from a sample batch
        try:
            sample_batch = next(iter(train_dataloader))
            images, _ = sample_batch
            self.metadata['resolution'] = f"{images.shape[2]}x{images.shape[3]}"
        except Exception:
            self.metadata['resolution'] = "Unknown"
        
        # 3. Save to the summary section of your JSON
        self.summary['dataset_info'] = self.metadata
    def on_train_epoch_start(self, trainer, pl_module):
        """Record epoch start time"""
        self.epoch_start_time = time.time()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Called at the end of each validation batch to collect predictions"""
        x, y = batch
        
        # Collect input statistics
        self.input_stats.append({
            'batch_idx': batch_idx,
            'epoch': trainer.current_epoch,
            'input_mean': float(x.mean().item()),
            'input_std': float(x.std().item()),
            'input_min': float(x.min().item()),
            'input_max': float(x.max().item()),
        })
        
        logits = pl_module(x)
        preds = torch.argmax(logits, dim=1)
        
        # Store all predictions and labels
        self.all_predictions.extend(preds.cpu().numpy().tolist())
        self.all_labels.extend(y.cpu().numpy().tolist())
        
        # Batch-wise metrics
        batch_accuracy = (preds == y).float().mean()
        batch_loss = torch.nn.functional.cross_entropy(logits, y)
        self.batch_metrics.append({
            'epoch': trainer.current_epoch,
            'batch_idx': batch_idx,
            'batch_accuracy': float(batch_accuracy.item()),
            'batch_loss': float(batch_loss.item()),
        })
        
        # Identify misclassified instances and convert to base64
        misclassified_mask = (preds != y)
        if misclassified_mask.any():
            misclassified_images = x[misclassified_mask].cpu()
            misclassified_preds = preds[misclassified_mask].cpu().numpy()
            misclassified_labels = y[misclassified_mask].cpu().numpy()
            
            for img, pred, true_label in zip(misclassified_images, misclassified_preds, misclassified_labels):
                # Convert tensor to PIL Image
                img_np = img.numpy().transpose(1, 2, 0)  # CHW to HWC
                img_np = ((img_np + 1) / 2 * 255).astype(np.uint8)  # Denormalize
                img_pil = Image.fromarray(img_np)

                # Compute a hash so we can count repeated misclassifications
                img_bytes = img.cpu().numpy().tobytes()
                img_hash = hashlib.sha1(img_bytes).hexdigest()

                # Convert to base64
                buffer = BytesIO()
                img_pil.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode()

                # determine distortion type using metadata if available (fallback to class name)
                distortion = self.metadata.get('classes', {}).get(int(pred), str(pred))

                # If we have an external distortion model (e.g. best.pt), run it on the misclassified image
                distortion_pred = None
                distortion_conf = None
                if self.distortion_model is not None:
                    try:
                        # models expect HWC uint8 images
                        img_for_model = img_np
                        res = self.distortion_model(img_for_model, conf=self.distortion_conf_threshold, verbose=False)
                        if len(res) > 0 and hasattr(res[0], 'boxes') and len(res[0].boxes) > 0:
                            # pick highest confidence detection
                            confs = res[0].boxes.conf.cpu()
                            best_idx = int(confs.argmax())
                            cls_idx = int(res[0].boxes.cls[best_idx].cpu())
                            distortion_pred = (
                                self.distortion_classes.get(cls_idx, str(cls_idx))
                                if isinstance(self.distortion_classes, dict)
                                else self.distortion_classes[cls_idx]
                            )
                            distortion_conf = float(confs[best_idx].item())
                    except Exception as e:
                        distortion_pred = f"error:{e}"

                entry = {
                    'hash': img_hash,
                    'image_base64': img_base64,
                    'true_label': int(true_label),
                    'predicted_label': int(pred),
                    # keep legacy distortion label (dataset class) for backwards compatibility
                    'distortion_type': distortion,
                    # add distortion model prediction if available
                    'distortion_predicted': distortion_pred,
                    'distortion_confidence': distortion_conf,
                    'epoch': trainer.current_epoch
                }
                self.misclassified_data.append(entry)
                # increment counter
                self.misclassified_counts[img_hash] = self.misclassified_counts.get(img_hash, 0) + 1
        
        # Capture learning rate
        
        # Capture learning rate
        for param_group in trainer.optimizers[0].param_groups:
            self.learning_rates.append({
                'epoch': trainer.current_epoch,
                'learning_rate': float(param_group['lr'])
            })
        
        # Capture gradient norms
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.gradient_norms.append({
            'epoch': trainer.current_epoch,
            'gradient_norm': float(total_norm)
        })
        
        metrics = trainer.callback_metrics

        # Determine if this is the last batch in the validation epoch
        try:
            num_val_batches = trainer.num_val_batches[dataloader_idx]
            is_last_batch = batch_idx == num_val_batches - 1
        except Exception:
            is_last_batch = False

        if is_last_batch:
            # Count correct predictions across the epoch
            correct_count = len([m for m in self.batch_metrics if m['epoch'] == trainer.current_epoch and m['batch_accuracy'] == 1.0])
            self.correct_predictions_per_epoch.append({
                'epoch': trainer.current_epoch,
                'total_correct_batches': correct_count
            })

            # Calculate class-wise metrics for the epoch
            if self.all_predictions:
                precision, recall, f1, support = precision_recall_fscore_support(
                    self.all_labels, self.all_predictions, average=None, zero_division=0
                )
                self.class_metrics.append({
                    'epoch': trainer.current_epoch,
                    'precision_per_class': precision.tolist(),
                    'recall_per_class': recall.tolist(),
                    'f1_per_class': f1.tolist(),
                    'support_per_class': support.tolist(),
                })
            else:
                precision = recall = f1 = np.array([])

            # Overall metrics for the epoch
            overall_accuracy = float(np.mean(np.array(self.all_predictions) == np.array(self.all_labels))) if len(self.all_predictions) > 0 else 0.0

            # Track epoch time (only add once per epoch)
            epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
            self.epoch_times.append({
                'epoch': trainer.current_epoch,
                'time_seconds': epoch_time
            })

            epoch_data = {
                'epoch': trainer.current_epoch,
                'train_loss': float(metrics.get('train_loss', 0).item() if hasattr(metrics.get('train_loss', 0), 'item') else metrics.get('train_loss', 0)),
                'val_loss': float(metrics.get('val_loss', 0).item() if hasattr(metrics.get('val_loss', 0), 'item') else metrics.get('val_loss', 0)),
                'accuracy': float(metrics.get('accuracy', 0).item() if hasattr(metrics.get('accuracy', 0), 'item') else metrics.get('accuracy', 0)),
                'overall_accuracy': overall_accuracy,
                'precision': float(np.mean(precision)) if len(precision) > 0 else 0,
                'recall': float(np.mean(recall)) if len(recall) > 0 else 0,
                'f1_score': float(np.mean(f1)) if len(f1) > 0 else 0,
                'num_misclassified': len([m for m in self.misclassified_data if m['epoch'] == trainer.current_epoch]),
                'epoch_time': epoch_time,
                'learning_rate': self.learning_rates[-1]['learning_rate'] if self.learning_rates else 0,
                'gradient_norm': self.gradient_norms[-1]['gradient_norm'] if self.gradient_norms else 0,
            }
            self.metrics.append(epoch_data)
            print(f"Epoch {trainer.current_epoch}: {epoch_data}")

            # Clear epoch-level buffers for the next epoch
            self.all_predictions = []
            self.all_labels = []

    def on_train_end(self, trainer, pl_module):
        """Save all metrics and data to a single comprehensive JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Combine all data
        all_data = {
            'summary': {
                # Merge the dataset_info (classes, resolution) captured at runtime
                **self.summary,
                'timestamp': timestamp,
                'total_epochs': len(self.metrics),
                'best_accuracy': max([m['accuracy'] for m in self.metrics]) if self.metrics else 0,
                'best_epoch': max(range(len(self.metrics)), key=lambda i: self.metrics[i]['accuracy']) if self.metrics else 0,
                'final_train_loss': self.metrics[-1]['train_loss'] if self.metrics else 0,
                'final_val_loss': self.metrics[-1]['val_loss'] if self.metrics else 0,
                'total_misclassified': len(self.misclassified_data),
                'total_time_seconds': sum([t['time_seconds'] for t in self.epoch_times]),
                'avg_gradient_norm': float(np.mean([g['gradient_norm'] for g in self.gradient_norms])) if self.gradient_norms else 0,
            },
            'epochs': self.metrics,
            'batch_metrics': self.batch_metrics,
            'class_metrics': self.class_metrics,
            'gradient_norms': self.gradient_norms,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
            'input_statistics': self.input_stats,
            'misclassified_samples': self.misclassified_data
        }
        
        # before saving, reorganize misclassified samples by frequency
        unique = {}
        for entry in self.misclassified_data:
            h = entry['hash']
            if h not in unique:
                unique[h] = {**entry, 'count': 0}
            unique[h]['count'] += 1
        # sort by count descending
        sorted_mis = sorted(unique.values(), key=lambda e: e['count'], reverse=True)
        all_data['misclassified_samples'] = sorted_mis
        
        # also generate a distribution graph of distortion types (prefer model prediction if available)
        distortion_counts = {}
        for entry in self.misclassified_data:
            dist = entry.get('distortion_predicted') or entry.get('distortion_type', 'unknown')
            distortion_counts[dist] = distortion_counts.get(dist, 0) + 1
        total = sum(distortion_counts.values())
        if total > 0:
            percentages = {k: v / total * 100 for k, v in distortion_counts.items()}
            plt.figure(figsize=(8, 6))
            plt.bar(percentages.keys(), percentages.values(), color='skyblue')
            plt.ylabel('Percentage of misclassified images')
            plt.title('Distortion type distribution')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            graph_path = os.path.join(self.save_dir, 'distortion_distribution.png')
            plt.savefig(graph_path)
            plt.close()
            print(f"📈 Distortion distribution graph saved to: {graph_path}")
        
        # Save consolidated JSON file
        output_file = os.path.join(self.save_dir, f'training_log_{timestamp}.json')
        with open(output_file, 'w') as f:
            json.dump(all_data, f, indent=2)
        
        print(f"\n✓ Training log saved to: {output_file}")
        
        # Log dataset info for verification
        dataset_info = all_data['summary'].get('dataset_info', {})
        print(f"📊 Dataset Info: {dataset_info.get('resolution')} resolution, {len(dataset_info.get('classes', {}))} classes")
        
        print(f"\n📊 Final Summary:")
        print(f"  • Total Epochs: {all_data['summary']['total_epochs']}")
        print(f"  • Best Accuracy: {all_data['summary']['best_accuracy']:.4f} (Epoch {all_data['summary']['best_epoch']})")
        print(f"  • Final Train Loss: {all_data['summary']['final_train_loss']:.4f}")
        print(f"  • Final Val Loss: {all_data['summary']['final_val_loss']:.4f}")
        print(f"  • Total Misclassified: {all_data['summary']['total_misclassified']}")
        print(f"  • Total Training Time: {all_data['summary']['total_time_seconds']:.2f}s")
        print(f"  • Avg Gradient Norm: {all_data['summary']['avg_gradient_norm']:.6f}")