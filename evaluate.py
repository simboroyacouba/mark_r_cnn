"""
Évaluation complète du modèle Mask R-CNN
Métriques calculées:
- mAP (mean Average Precision)
- mAP@50 (IoU threshold = 0.5)
- mAP@50:95 (IoU thresholds de 0.5 à 0.95)
- Precision, Recall, F1-Score
- IoU moyen (boîtes et masques)
- Matrice de confusion
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T
from torchvision.ops import box_iou
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask_utils
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from collections import defaultdict
from tqdm import tqdm
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

# Charger les classes depuis le fichier YAML
def load_classes(yaml_path="classes.yaml"):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data['classes']

CONFIG = {
    # Chemins (à adapter)
    "images_dir": os.getenv("SEGMENTATION_DATASET_IMAGES_DIR"),
    "annotations_file": os.getenv("SEGMENTATION_DATASET_ANNOTATIONS_FILE"),
    "classes_file": os.getenv("CLASSES_FILE", "classes.yaml"),
    "model_path": "./output/best_model.pth",
    "output_dir": "./evaluation",
    
    # Classes
   "classes": load_classes(os.getenv("CLASSES_FILE", "classes.yaml")),
    
    # Paramètres d'évaluation
    "score_threshold": 0.5,      # Seuil de confiance pour les prédictions
    "iou_thresholds": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    "batch_size": 1,
    "num_workers": 0,
}


# =============================================================================
# DATASET
# =============================================================================

class EvalDataset(torch.utils.data.Dataset):
    """Dataset pour l'évaluation"""
    
    def __init__(self, images_dir, annotations_file):
        self.images_dir = images_dir
        self.coco = COCO(annotations_file)
        self.image_ids = list(self.coco.imgs.keys())
        
        self.cat_ids = self.coco.getCatIds()
        self.cat_mapping = {cat_id: idx + 1 for idx, cat_id in enumerate(self.cat_ids)}
        self.reverse_cat_mapping = {v: k for k, v in self.cat_mapping.items()}
        
        print(f"Dataset d'évaluation: {len(self.image_ids)} images")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        image_tensor = T.ToTensor()(image)
        
        # Récupérer les annotations ground truth
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        masks = []
        areas = []
        
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_mapping[ann['category_id']])
            
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):
                    rles = coco_mask_utils.frPyObjects(
                        ann['segmentation'],
                        img_info['height'],
                        img_info['width']
                    )
                    rle = coco_mask_utils.merge(rles)
                    mask = coco_mask_utils.decode(rle)
                else:
                    mask = coco_mask_utils.decode(ann['segmentation'])
                masks.append(mask)
            
            areas.append(ann.get('area', w * h))
        
        if len(boxes) == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, img_info['height'], img_info['width']), dtype=torch.uint8),
                "image_id": img_id,
                "area": torch.zeros((0,), dtype=torch.float32),
            }
        else:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "masks": torch.as_tensor(np.array(masks), dtype=torch.uint8),
                "image_id": img_id,
                "area": torch.as_tensor(areas, dtype=torch.float32),
            }
        
        return image_tensor, target


# =============================================================================
# MODÈLE
# =============================================================================

def get_model(num_classes):
    """Créer le modèle Mask R-CNN"""
    model = maskrcnn_resnet50_fpn_v2(weights=None)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    
    return model


def load_model(model_path, num_classes, device):
    """Charger le modèle entraîné"""
    model = get_model(num_classes)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Modèle chargé: {model_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    return model


# =============================================================================
# CALCUL DES MÉTRIQUES
# =============================================================================

def calculate_iou_boxes(box1, box2):
    """Calculer IoU entre deux boîtes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def calculate_iou_masks(mask1, mask2):
    """Calculer IoU entre deux masques binaires"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def calculate_ap(recalls, precisions):
    """Calculer Average Precision (aire sous la courbe PR)"""
    # Ajouter les points sentinelles
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])
    
    # Assurer que la précision est décroissante
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Trouver les points où le recall change
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    
    # Calculer l'aire
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    
    return ap


class MetricsCalculator:
    """Classe pour calculer toutes les métriques"""
    
    def __init__(self, num_classes, class_names, iou_thresholds):
        self.num_classes = num_classes
        self.class_names = class_names
        self.iou_thresholds = iou_thresholds
        
        # Stockage des prédictions et ground truths
        self.all_predictions = []  # Liste de dicts par image
        self.all_ground_truths = []
        
        # Métriques par classe et par seuil IoU
        self.reset()
    
    def reset(self):
        """Réinitialiser les compteurs"""
        self.tp_per_class = defaultdict(lambda: defaultdict(int))  # [class][iou_thresh]
        self.fp_per_class = defaultdict(lambda: defaultdict(int))
        self.fn_per_class = defaultdict(lambda: defaultdict(int))
        
        self.all_predictions = []
        self.all_ground_truths = []
        
        self.box_ious = []
        self.mask_ious = []
    
    def add_batch(self, predictions, targets):
        """Ajouter un batch de prédictions et ground truths"""
        
        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_masks = pred['masks'].cpu().numpy()
            
            gt_boxes = target['boxes'].cpu().numpy()
            gt_labels = target['labels'].cpu().numpy()
            gt_masks = target['masks'].cpu().numpy()
            
            self.all_predictions.append({
                'boxes': pred_boxes,
                'labels': pred_labels,
                'scores': pred_scores,
                'masks': pred_masks
            })
            
            self.all_ground_truths.append({
                'boxes': gt_boxes,
                'labels': gt_labels,
                'masks': gt_masks
            })
            
            # Calculer les métriques pour cette image
            self._evaluate_image(
                pred_boxes, pred_labels, pred_scores, pred_masks,
                gt_boxes, gt_labels, gt_masks
            )
    
    def _evaluate_image(self, pred_boxes, pred_labels, pred_scores, pred_masks,
                        gt_boxes, gt_labels, gt_masks):
        """Évaluer une image"""
        
        for iou_thresh in self.iou_thresholds:
            # Pour chaque classe
            for class_id in range(1, self.num_classes):  # Ignorer background
                # Filtrer par classe
                pred_mask_cls = pred_labels == class_id
                gt_mask_cls = gt_labels == class_id
                
                pred_boxes_cls = pred_boxes[pred_mask_cls]
                pred_scores_cls = pred_scores[pred_mask_cls]
                pred_masks_cls = pred_masks[pred_mask_cls]
                
                gt_boxes_cls = gt_boxes[gt_mask_cls]
                gt_masks_cls = gt_masks[gt_mask_cls]
                
                n_pred = len(pred_boxes_cls)
                n_gt = len(gt_boxes_cls)
                
                if n_gt == 0 and n_pred == 0:
                    continue
                
                if n_gt == 0:
                    # Toutes les prédictions sont des FP
                    self.fp_per_class[class_id][iou_thresh] += n_pred
                    continue
                
                if n_pred == 0:
                    # Tous les GT sont des FN
                    self.fn_per_class[class_id][iou_thresh] += n_gt
                    continue
                
                # Calculer la matrice IoU
                iou_matrix = np.zeros((n_pred, n_gt))
                for i, (pb, pm) in enumerate(zip(pred_boxes_cls, pred_masks_cls)):
                    for j, (gb, gm) in enumerate(zip(gt_boxes_cls, gt_masks_cls)):
                        # IoU des boîtes
                        box_iou_val = calculate_iou_boxes(pb, gb)
                        # IoU des masques
                        if pm.shape[0] > 0 and gm.shape[0] > 0:
                            mask_iou_val = calculate_iou_masks(pm[0] > 0.5, gm > 0)
                            iou_matrix[i, j] = (box_iou_val + mask_iou_val) / 2
                        else:
                            iou_matrix[i, j] = box_iou_val
                        
                        # Stocker les IoUs pour statistiques globales
                        if iou_thresh == 0.5:  # Éviter les doublons
                            self.box_ious.append(box_iou_val)
                            if pm.shape[0] > 0 and gm.shape[0] > 0:
                                self.mask_ious.append(mask_iou_val)
                
                # Matching glouton (greedy)
                matched_gt = set()
                matched_pred = set()
                
                # Trier par score décroissant
                sorted_indices = np.argsort(-pred_scores_cls)
                
                for pred_idx in sorted_indices:
                    best_iou = 0
                    best_gt = -1
                    
                    for gt_idx in range(n_gt):
                        if gt_idx in matched_gt:
                            continue
                        if iou_matrix[pred_idx, gt_idx] > best_iou:
                            best_iou = iou_matrix[pred_idx, gt_idx]
                            best_gt = gt_idx
                    
                    if best_iou >= iou_thresh:
                        matched_gt.add(best_gt)
                        matched_pred.add(pred_idx)
                        self.tp_per_class[class_id][iou_thresh] += 1
                    else:
                        self.fp_per_class[class_id][iou_thresh] += 1
                
                # FN = GT non matchés
                self.fn_per_class[class_id][iou_thresh] += n_gt - len(matched_gt)
    
    def compute_metrics(self):
        """Calculer toutes les métriques finales"""
        
        results = {
            'per_class': {},
            'overall': {},
            'iou_stats': {}
        }
        
        # Métriques par classe
        for class_id in range(1, self.num_classes):
            class_name = self.class_names[class_id]
            results['per_class'][class_name] = {}
            
            for iou_thresh in self.iou_thresholds:
                tp = self.tp_per_class[class_id][iou_thresh]
                fp = self.fp_per_class[class_id][iou_thresh]
                fn = self.fn_per_class[class_id][iou_thresh]
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                results['per_class'][class_name][f'iou_{iou_thresh}'] = {
                    'TP': tp,
                    'FP': fp,
                    'FN': fn,
                    'Precision': precision,
                    'Recall': recall,
                    'F1': f1
                }
        
        # Métriques globales
        for iou_thresh in self.iou_thresholds:
            total_tp = sum(self.tp_per_class[c][iou_thresh] for c in range(1, self.num_classes))
            total_fp = sum(self.fp_per_class[c][iou_thresh] for c in range(1, self.num_classes))
            total_fn = sum(self.fn_per_class[c][iou_thresh] for c in range(1, self.num_classes))
            
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results['overall'][f'iou_{iou_thresh}'] = {
                'TP': total_tp,
                'FP': total_fp,
                'FN': total_fn,
                'Precision': precision,
                'Recall': recall,
                'F1': f1
            }
        
        # Calculer mAP@50
        results['mAP50'] = results['overall']['iou_0.5']['Precision']
        
        # Calculer mAP@50:95 (moyenne des précisions sur tous les seuils)
        precisions_all_thresholds = [
            results['overall'][f'iou_{t}']['Precision'] 
            for t in self.iou_thresholds
        ]
        results['mAP50_95'] = np.mean(precisions_all_thresholds)
        
        # Calculer mAP par classe
        results['mAP_per_class'] = {}
        for class_id in range(1, self.num_classes):
            class_name = self.class_names[class_id]
            precisions = [
                results['per_class'][class_name][f'iou_{t}']['Precision']
                for t in self.iou_thresholds
            ]
            results['mAP_per_class'][class_name] = {
                'AP50': results['per_class'][class_name]['iou_0.5']['Precision'],
                'AP50_95': np.mean(precisions)
            }
        
        # Statistiques IoU
        if self.box_ious:
            results['iou_stats']['box_iou_mean'] = np.mean(self.box_ious)
            results['iou_stats']['box_iou_std'] = np.std(self.box_ious)
            results['iou_stats']['box_iou_median'] = np.median(self.box_ious)
        
        if self.mask_ious:
            results['iou_stats']['mask_iou_mean'] = np.mean(self.mask_ious)
            results['iou_stats']['mask_iou_std'] = np.std(self.mask_ious)
            results['iou_stats']['mask_iou_median'] = np.median(self.mask_ious)
        
        return results


# =============================================================================
# ÉVALUATION COCO OFFICIELLE
# =============================================================================

def evaluate_coco(model, dataset, device):
    """Évaluation avec les métriques COCO officielles"""
    
    model.eval()
    coco_gt = dataset.coco
    coco_results = []
    
    print("\n📊 Évaluation COCO officielle...")
    
    for idx in tqdm(range(len(dataset)), desc="Inférence"):
        image, target = dataset[idx]
        image_id = target['image_id']
        
        with torch.no_grad():
            predictions = model([image.to(device)])
        
        pred = predictions[0]
        
        # Convertir les prédictions au format COCO
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        masks = pred['masks'].cpu().numpy()
        
        for i in range(len(boxes)):
            if scores[i] < CONFIG['score_threshold']:
                continue
            
            # Boîte au format COCO [x, y, width, height]
            box = boxes[i]
            coco_box = [
                float(box[0]), 
                float(box[1]), 
                float(box[2] - box[0]), 
                float(box[3] - box[1])
            ]
            
            # Masque RLE
            mask_binary = (masks[i, 0] > 0.5).astype(np.uint8)
            rle = coco_mask_utils.encode(np.asfortranarray(mask_binary))
            rle['counts'] = rle['counts'].decode('utf-8')
            
            # Convertir label local vers COCO
            coco_cat_id = dataset.reverse_cat_mapping.get(labels[i], labels[i])
            
            coco_results.append({
                'image_id': image_id,
                'category_id': coco_cat_id,
                'bbox': coco_box,
                'score': float(scores[i]),
                'segmentation': rle
            })
    
    if len(coco_results) == 0:
        print("⚠️  Aucune prédiction au-dessus du seuil!")
        return None
    
    # Évaluation bbox
    print("\n📦 Évaluation des boîtes englobantes (bbox):")
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval_bbox = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval_bbox.evaluate()
    coco_eval_bbox.accumulate()
    coco_eval_bbox.summarize()
    
    # Évaluation segmentation
    print("\n🎭 Évaluation de la segmentation (mask):")
    coco_eval_segm = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval_segm.evaluate()
    coco_eval_segm.accumulate()
    coco_eval_segm.summarize()
    
    return {
        'bbox': {
            'AP': coco_eval_bbox.stats[0],
            'AP50': coco_eval_bbox.stats[1],
            'AP75': coco_eval_bbox.stats[2],
            'AP_small': coco_eval_bbox.stats[3],
            'AP_medium': coco_eval_bbox.stats[4],
            'AP_large': coco_eval_bbox.stats[5],
            'AR_1': coco_eval_bbox.stats[6],
            'AR_10': coco_eval_bbox.stats[7],
            'AR_100': coco_eval_bbox.stats[8],
        },
        'segm': {
            'AP': coco_eval_segm.stats[0],
            'AP50': coco_eval_segm.stats[1],
            'AP75': coco_eval_segm.stats[2],
            'AP_small': coco_eval_segm.stats[3],
            'AP_medium': coco_eval_segm.stats[4],
            'AP_large': coco_eval_segm.stats[5],
            'AR_1': coco_eval_segm.stats[6],
            'AR_10': coco_eval_segm.stats[7],
            'AR_100': coco_eval_segm.stats[8],
        }
    }


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_metrics(results, output_dir):
    """Créer les graphiques des métriques"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Graphique des métriques par classe (AP50 et AP50:95)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    class_names = list(results['mAP_per_class'].keys())
    ap50_values = [results['mAP_per_class'][c]['AP50'] for c in class_names]
    ap50_95_values = [results['mAP_per_class'][c]['AP50_95'] for c in class_names]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    axes[0].bar(x - width/2, ap50_values, width, label='AP@50', color='steelblue')
    axes[0].bar(x + width/2, ap50_95_values, width, label='AP@50:95', color='coral')
    axes[0].set_xlabel('Classes')
    axes[0].set_ylabel('Average Precision')
    axes[0].set_title('AP par classe')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # 2. Precision, Recall, F1 par classe (IoU=0.5)
    precisions = [results['per_class'][c]['iou_0.5']['Precision'] for c in class_names]
    recalls = [results['per_class'][c]['iou_0.5']['Recall'] for c in class_names]
    f1s = [results['per_class'][c]['iou_0.5']['F1'] for c in class_names]
    
    width = 0.25
    axes[1].bar(x - width, precisions, width, label='Precision', color='green')
    axes[1].bar(x, recalls, width, label='Recall', color='blue')
    axes[1].bar(x + width, f1s, width, label='F1-Score', color='red')
    axes[1].set_xlabel('Classes')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Precision / Recall / F1 par classe (IoU=0.5)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_per_class.png'), dpi=150)
    plt.close()
    
    # 3. Métriques globales vs seuil IoU
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iou_thresholds = CONFIG['iou_thresholds']
    global_precisions = [results['overall'][f'iou_{t}']['Precision'] for t in iou_thresholds]
    global_recalls = [results['overall'][f'iou_{t}']['Recall'] for t in iou_thresholds]
    global_f1s = [results['overall'][f'iou_{t}']['F1'] for t in iou_thresholds]
    
    ax.plot(iou_thresholds, global_precisions, 'o-', label='Precision', linewidth=2, markersize=8)
    ax.plot(iou_thresholds, global_recalls, 's-', label='Recall', linewidth=2, markersize=8)
    ax.plot(iou_thresholds, global_f1s, '^-', label='F1-Score', linewidth=2, markersize=8)
    
    ax.set_xlabel('Seuil IoU')
    ax.set_ylabel('Score')
    ax.set_title('Métriques globales vs Seuil IoU')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.set_xlim(0.45, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_vs_iou.png'), dpi=150)
    plt.close()
    
    # 4. Distribution des IoU
    if results.get('iou_stats'):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        if 'box_iou_mean' in results['iou_stats']:
            # Simuler une distribution (on n'a que les stats)
            axes[0].axvline(results['iou_stats']['box_iou_mean'], color='red', 
                           linestyle='--', linewidth=2, label=f"Moyenne: {results['iou_stats']['box_iou_mean']:.3f}")
            axes[0].axvline(results['iou_stats']['box_iou_median'], color='blue',
                           linestyle='--', linewidth=2, label=f"Médiane: {results['iou_stats']['box_iou_median']:.3f}")
            axes[0].set_xlabel('IoU')
            axes[0].set_title('Statistiques IoU des boîtes')
            axes[0].legend()
            axes[0].set_xlim(0, 1)
        
        if 'mask_iou_mean' in results['iou_stats']:
            axes[1].axvline(results['iou_stats']['mask_iou_mean'], color='red',
                           linestyle='--', linewidth=2, label=f"Moyenne: {results['iou_stats']['mask_iou_mean']:.3f}")
            axes[1].axvline(results['iou_stats']['mask_iou_median'], color='blue',
                           linestyle='--', linewidth=2, label=f"Médiane: {results['iou_stats']['mask_iou_median']:.3f}")
            axes[1].set_xlabel('IoU')
            axes[1].set_title('Statistiques IoU des masques')
            axes[1].legend()
            axes[1].set_xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'iou_stats.png'), dpi=150)
        plt.close()
    
    print(f"📊 Graphiques sauvegardés dans: {output_dir}")


def generate_report(results, coco_results, output_dir):
    """Générer un rapport complet"""
    
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("   RAPPORT D'ÉVALUATION - MASK R-CNN CADASTRAL\n")
        f.write("=" * 70 + "\n")
        f.write(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        # Résumé principal
        f.write("📊 RÉSUMÉ DES MÉTRIQUES PRINCIPALES\n")
        f.write("-" * 50 + "\n")
        f.write(f"   mAP@50:        {results['mAP50']:.4f} ({results['mAP50']*100:.2f}%)\n")
        f.write(f"   mAP@50:95:     {results['mAP50_95']:.4f} ({results['mAP50_95']*100:.2f}%)\n")
        f.write(f"\n   Precision@50:  {results['overall']['iou_0.5']['Precision']:.4f}\n")
        f.write(f"   Recall@50:     {results['overall']['iou_0.5']['Recall']:.4f}\n")
        f.write(f"   F1-Score@50:   {results['overall']['iou_0.5']['F1']:.4f}\n")
        
        # IoU stats
        if results.get('iou_stats'):
            f.write(f"\n   IoU moyen (boîtes):  {results['iou_stats'].get('box_iou_mean', 0):.4f}\n")
            f.write(f"   IoU moyen (masques): {results['iou_stats'].get('mask_iou_mean', 0):.4f}\n")
        
        # COCO metrics si disponibles
        if coco_results:
            f.write("\n\n📦 MÉTRIQUES COCO OFFICIELLES - BBOX\n")
            f.write("-" * 50 + "\n")
            f.write(f"   AP (IoU=0.50:0.95):  {coco_results['bbox']['AP']:.4f}\n")
            f.write(f"   AP@50:               {coco_results['bbox']['AP50']:.4f}\n")
            f.write(f"   AP@75:               {coco_results['bbox']['AP75']:.4f}\n")
            f.write(f"   AR (maxDets=100):    {coco_results['bbox']['AR_100']:.4f}\n")
            
            f.write("\n\n🎭 MÉTRIQUES COCO OFFICIELLES - SEGMENTATION\n")
            f.write("-" * 50 + "\n")
            f.write(f"   AP (IoU=0.50:0.95):  {coco_results['segm']['AP']:.4f}\n")
            f.write(f"   AP@50:               {coco_results['segm']['AP50']:.4f}\n")
            f.write(f"   AP@75:               {coco_results['segm']['AP75']:.4f}\n")
            f.write(f"   AR (maxDets=100):    {coco_results['segm']['AR_100']:.4f}\n")
        
        # Métriques par classe
        f.write("\n\n📋 MÉTRIQUES PAR CLASSE (IoU=0.5)\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Classe':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AP50':>10}\n")
        f.write("-" * 65 + "\n")
        
        for class_name in results['per_class']:
            metrics = results['per_class'][class_name]['iou_0.5']
            ap50 = results['mAP_per_class'][class_name]['AP50']
            f.write(f"{class_name:<25} {metrics['Precision']:>10.4f} {metrics['Recall']:>10.4f} "
                   f"{metrics['F1']:>10.4f} {ap50:>10.4f}\n")
        
        # Détails TP/FP/FN
        f.write("\n\n📈 DÉTAILS TP/FP/FN PAR CLASSE (IoU=0.5)\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Classe':<25} {'TP':>8} {'FP':>8} {'FN':>8}\n")
        f.write("-" * 50 + "\n")
        
        for class_name in results['per_class']:
            metrics = results['per_class'][class_name]['iou_0.5']
            f.write(f"{class_name:<25} {metrics['TP']:>8} {metrics['FP']:>8} {metrics['FN']:>8}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"📄 Rapport sauvegardé: {report_path}")
    return report_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("   ÉVALUATION MASK R-CNN - Segmentation des Toitures")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    
    # Créer le dossier de sortie
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # Charger le dataset
    print("\n📂 Chargement du dataset...")
    dataset = EvalDataset(
        CONFIG["images_dir"],
        CONFIG["annotations_file"]
    )
    
    # Charger le modèle
    print("\n🧠 Chargement du modèle...")
    num_classes = len(CONFIG["classes"])
    model = load_model(CONFIG["model_path"], num_classes, device)
    
    # Initialiser le calculateur de métriques
    metrics_calc = MetricsCalculator(
        num_classes=num_classes,
        class_names=CONFIG["classes"],
        iou_thresholds=CONFIG["iou_thresholds"]
    )
    
    # Évaluation manuelle
    print("\n📊 Calcul des métriques...")
    model.eval()
    
    for idx in tqdm(range(len(dataset)), desc="Évaluation"):
        image, target = dataset[idx]
        
        with torch.no_grad():
            predictions = model([image.to(device)])
        
        # Filtrer par score
        pred = predictions[0]
        keep = pred['scores'] > CONFIG['score_threshold']
        
        filtered_pred = {
            'boxes': pred['boxes'][keep],
            'labels': pred['labels'][keep],
            'scores': pred['scores'][keep],
            'masks': pred['masks'][keep]
        }
        
        # Convertir target pour le batch
        target_batch = {
            'boxes': target['boxes'],
            'labels': target['labels'],
            'masks': target['masks']
        }
        
        metrics_calc.add_batch([filtered_pred], [target_batch])
    
    # Calculer les métriques finales
    results = metrics_calc.compute_metrics()
    
    # Évaluation COCO officielle
    coco_results = None
    try:
        coco_results = evaluate_coco(model, dataset, device)
    except Exception as e:
        print(f"⚠️  Erreur lors de l'évaluation COCO: {e}")
    
    # Affichage des résultats
    print("\n" + "=" * 70)
    print("   📊 RÉSULTATS DE L'ÉVALUATION")
    print("=" * 70)
    
    print(f"\n🎯 MÉTRIQUES PRINCIPALES")
    print(f"   {'─' * 40}")
    print(f"   mAP@50:        {results['mAP50']:.4f} ({results['mAP50']*100:.2f}%)")
    print(f"   mAP@50:95:     {results['mAP50_95']:.4f} ({results['mAP50_95']*100:.2f}%)")
    print(f"\n   Precision@50:  {results['overall']['iou_0.5']['Precision']:.4f}")
    print(f"   Recall@50:     {results['overall']['iou_0.5']['Recall']:.4f}")
    print(f"   F1-Score@50:   {results['overall']['iou_0.5']['F1']:.4f}")
    
    if results.get('iou_stats'):
        print(f"\n   IoU moyen (boîtes):  {results['iou_stats'].get('box_iou_mean', 0):.4f}")
        print(f"   IoU moyen (masques): {results['iou_stats'].get('mask_iou_mean', 0):.4f}")
    
    print(f"\n📋 PAR CLASSE (IoU=0.5)")
    print(f"   {'─' * 40}")
    for class_name in results['per_class']:
        metrics = results['per_class'][class_name]['iou_0.5']
        print(f"   {class_name}:")
        print(f"      Precision: {metrics['Precision']:.4f} | Recall: {metrics['Recall']:.4f} | F1: {metrics['F1']:.4f}")
    
    # Sauvegarder les résultats
    results_path = os.path.join(CONFIG["output_dir"], "metrics.json")
    
    # Convertir les defaultdicts en dicts normaux pour JSON
    def convert_to_serializable(obj):
        if isinstance(obj, defaultdict):
            return dict(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    results_serializable = json.loads(
        json.dumps(results, default=convert_to_serializable)
    )
    
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\n💾 Métriques sauvegardées: {results_path}")
    
    # Générer les graphiques
    plot_metrics(results, CONFIG["output_dir"])
    
    # Générer le rapport
    generate_report(results, coco_results, CONFIG["output_dir"])
    
    print("\n" + "=" * 70)
    print("   ✅ ÉVALUATION TERMINÉE")
    print("=" * 70)


if __name__ == "__main__":
    main()
