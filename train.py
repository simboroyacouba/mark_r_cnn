"""
Entraînement Mask R-CNN pour segmentation des toitures cadastrales
Dataset: Images aériennes annotées avec CVAT (format COCO)
Classes: toiture_tole_ondulee, toiture_tole_bac, toiture_tuile, toiture_dalle
"""

import os
import json
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import time
from datetime import datetime, timedelta
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


    "output_dir": "./output",
    
    # Classes (dans l'ordre de CVAT)
   "classes": load_classes(os.getenv("CLASSES_FILE", "classes.yaml")),
    
    # Hyperparamètres
    "num_epochs": 25,
    "batch_size": 2,           # 2-4 pour GPU 8GB, augmenter si plus de VRAM
    "learning_rate": 0.005,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "lr_step_size": 8,
    "lr_gamma": 0.1,
    
    # Dataset
    "train_split": 0.85,       # 85% train, 15% validation
    "num_workers": 2,
    
    # Sauvegarde
    "save_every": 5,           # Sauvegarder tous les N epochs
}


# =============================================================================
# UTILITAIRES TEMPS
# =============================================================================

def format_time(seconds):
    """Formater les secondes en format lisible HH:MM:SS"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


class TrainingTimer:
    """Classe pour gérer le suivi du temps d'entraînement"""
    
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        self.start_time = None
        self.epoch_times = []
        self.epoch_start = None
        
    def start_training(self):
        """Démarrer le chronomètre global"""
        self.start_time = time.time()
        self.training_start_datetime = datetime.now()
        
    def start_epoch(self):
        """Démarrer le chronomètre pour une epoch"""
        self.epoch_start = time.time()
        
    def end_epoch(self, epoch):
        """Terminer une epoch et calculer les statistiques"""
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        
        # Calculs
        total_elapsed = time.time() - self.start_time
        avg_epoch_time = np.mean(self.epoch_times)
        remaining_epochs = self.num_epochs - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs
        estimated_total = total_elapsed + estimated_remaining
        eta = datetime.now() + timedelta(seconds=estimated_remaining)
        
        return {
            'epoch_time': epoch_time,
            'total_elapsed': total_elapsed,
            'avg_epoch_time': avg_epoch_time,
            'estimated_remaining': estimated_remaining,
            'estimated_total': estimated_total,
            'eta': eta,
            'progress_percent': ((epoch + 1) / self.num_epochs) * 100
        }
    
    def get_final_stats(self):
        """Obtenir les statistiques finales"""
        total_time = time.time() - self.start_time
        return {
            'total_time': total_time,
            'total_time_formatted': format_time(total_time),
            'avg_epoch_time': np.mean(self.epoch_times),
            'avg_epoch_time_formatted': format_time(np.mean(self.epoch_times)),
            'min_epoch_time': np.min(self.epoch_times),
            'min_epoch_time_formatted': format_time(np.min(self.epoch_times)),
            'max_epoch_time': np.max(self.epoch_times),
            'max_epoch_time_formatted': format_time(np.max(self.epoch_times)),
            'std_epoch_time': np.std(self.epoch_times),
            'epoch_times': self.epoch_times,
            'start_datetime': self.training_start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            'end_datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


# =============================================================================
# DATASET
# =============================================================================

class CadastralDataset(torch.utils.data.Dataset):
    """Dataset pour segmentation cadastrale depuis annotations COCO/CVAT"""
    
    def __init__(self, images_dir, annotations_file, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms
        
        # Charger annotations COCO
        self.coco = COCO(annotations_file)
        self.image_ids = list(self.coco.imgs.keys())
        
        # Mapping catégories COCO -> indices locaux
        self.cat_ids = self.coco.getCatIds()
        self.cat_mapping = {cat_id: idx + 1 for idx, cat_id in enumerate(self.cat_ids)}
        
        print(f"Dataset chargé: {len(self.image_ids)} images")
        print(f"Catégories: {[self.coco.cats[c]['name'] for c in self.cat_ids]}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        
        # Charger l'image
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        
        # Récupérer les annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        masks = []
        areas = []
        
        for ann in anns:
            # Ignorer les annotations invalides
            if ann.get('iscrowd', 0):
                continue
            
            # Boîte englobante [x, y, width, height] -> [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            
            # Classe (remapper vers indices locaux)
            labels.append(self.cat_mapping[ann['category_id']])
            
            # Masque
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):
                    # Polygone -> RLE -> masque binaire
                    rles = coco_mask_utils.frPyObjects(
                        ann['segmentation'],
                        img_info['height'],
                        img_info['width']
                    )
                    rle = coco_mask_utils.merge(rles)
                    mask = coco_mask_utils.decode(rle)
                else:
                    # Déjà en RLE
                    mask = coco_mask_utils.decode(ann['segmentation'])
                masks.append(mask)
            
            # Aire
            areas.append(ann.get('area', w * h))
        
        # Gérer le cas sans annotations
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, img_info['height'], img_info['width']), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            areas = torch.as_tensor(areas, dtype=torch.float32)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        # Convertir image en tensor
        image = T.ToTensor()(image)
        
        # Appliquer transformations
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target


# =============================================================================
# TRANSFORMATIONS (Augmentation)
# =============================================================================

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, target):
        if np.random.random() < self.prob:
            image = torch.flip(image, [-1])
            
            # Ajuster les boîtes
            if "boxes" in target and len(target["boxes"]) > 0:
                width = image.shape[-1]
                boxes = target["boxes"]
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes
            
            # Ajuster les masques
            if "masks" in target and len(target["masks"]) > 0:
                target["masks"] = torch.flip(target["masks"], [-1])
        
        return image, target


class RandomVerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, target):
        if np.random.random() < self.prob:
            image = torch.flip(image, [-2])
            
            if "boxes" in target and len(target["boxes"]) > 0:
                height = image.shape[-2]
                boxes = target["boxes"]
                boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
                target["boxes"] = boxes
            
            if "masks" in target and len(target["masks"]) > 0:
                target["masks"] = torch.flip(target["masks"], [-2])
        
        return image, target


def get_transforms(train=True):
    transforms = []
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
        transforms.append(RandomVerticalFlip(0.5))
    return Compose(transforms)


# =============================================================================
# MODÈLE
# =============================================================================

def get_model(num_classes):
    """Créer un Mask R-CNN fine-tuné pour N classes"""
    
    # Charger le modèle pré-entraîné sur COCO
    model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
    
    # Remplacer le classificateur de boîtes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Remplacer le prédicteur de masques
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    
    return model


# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

def collate_fn(batch):
    """Fonction de collation pour DataLoader"""
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Entraîner une epoch"""
    model.train()
    
    total_loss = 0
    loss_classifier = 0
    loss_box_reg = 0
    loss_mask = 0
    loss_objectness = 0
    loss_rpn_box = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Accumuler les pertes
        total_loss += losses.item()
        loss_classifier += loss_dict.get('loss_classifier', torch.tensor(0)).item()
        loss_box_reg += loss_dict.get('loss_box_reg', torch.tensor(0)).item()
        loss_mask += loss_dict.get('loss_mask', torch.tensor(0)).item()
        loss_objectness += loss_dict.get('loss_objectness', torch.tensor(0)).item()
        loss_rpn_box += loss_dict.get('loss_rpn_box_reg', torch.tensor(0)).item()
        
        pbar.set_postfix({
            'loss': f"{losses.item():.4f}",
            'mask': f"{loss_dict.get('loss_mask', torch.tensor(0)).item():.4f}"
        })
    
    n = len(data_loader)
    return {
        'total': total_loss / n,
        'classifier': loss_classifier / n,
        'box_reg': loss_box_reg / n,
        'mask': loss_mask / n,
        'objectness': loss_objectness / n,
        'rpn_box': loss_rpn_box / n
    }


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Évaluer sur le set de validation"""
    model.eval()
    
    total_loss = 0
    
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # En mode eval, on doit passer en mode train temporairement pour avoir les pertes
        model.train()
        loss_dict = model(images, targets)
        model.eval()
        
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
    
    return total_loss / len(data_loader)


def save_checkpoint(model, optimizer, epoch, loss, path, time_stats=None):
    """Sauvegarder un checkpoint avec informations de temps"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if time_stats:
        checkpoint['time_stats'] = time_stats
    torch.save(checkpoint, path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("   MASK R-CNN - Segmentation des Toitures Cadastrales")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Créer le dossier de sortie
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # Dataset
    print("\n📂 Chargement du dataset...")
    full_dataset = CadastralDataset(
        CONFIG["images_dir"],
        CONFIG["annotations_file"],
        transforms=None
    )
    
    # Split train/val
    train_size = int(CONFIG["train_split"] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"   Train: {len(train_dataset)} images")
    print(f"   Val: {len(val_dataset)} images")
    
    # Appliquer les transformations
    train_dataset.dataset.transforms = get_transforms(train=True)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Modèle
    print("\n🧠 Création du modèle...")
    num_classes = len(CONFIG["classes"])
    model = get_model(num_classes)
    model.to(device)
    print(f"   Classes: {CONFIG['classes']}")
    
    # Optimiseur
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=CONFIG["learning_rate"],
        momentum=CONFIG["momentum"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    # Scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=CONFIG["lr_step_size"],
        gamma=CONFIG["lr_gamma"]
    )
    
    # Historique des pertes et temps
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'epoch_times': [],
        'cumulative_times': []
    }
    
    best_val_loss = float('inf')
    
    # Initialiser le timer
    timer = TrainingTimer(CONFIG["num_epochs"])
    
    # Entraînement
    print("\n" + "=" * 70)
    print("   🚀 DÉBUT DE L'ENTRAÎNEMENT")
    print(f"   📅 Démarré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   📊 Epochs: {CONFIG['num_epochs']} | Batch size: {CONFIG['batch_size']}")
    print("=" * 70)
    
    timer.start_training()
    
    for epoch in range(CONFIG["num_epochs"]):
        timer.start_epoch()
        
        # Train
        train_losses = train_one_epoch(model, optimizer, train_loader, device, epoch)
        
        # Validation
        val_loss = evaluate(model, val_loader, device)
        
        # Scheduler
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Obtenir les stats de temps
        time_stats = timer.end_epoch(epoch)
        
        # Historique
        history['train_loss'].append(train_losses['total'])
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        history['epoch_times'].append(time_stats['epoch_time'])
        history['cumulative_times'].append(time_stats['total_elapsed'])
        
        # Affichage détaillé
        print(f"\n{'─' * 70}")
        print(f"📈 Epoch {epoch+1}/{CONFIG['num_epochs']} | Progression: {time_stats['progress_percent']:.1f}%")
        print(f"{'─' * 70}")
        print(f"   📉 Train Loss: {train_losses['total']:.4f} (mask: {train_losses['mask']:.4f})")
        print(f"   📊 Val Loss:   {val_loss:.4f}")
        print(f"   📐 LR:         {current_lr:.6f}")
        print(f"{'─' * 70}")
        print(f"   ⏱️  Temps epoch:       {format_time(time_stats['epoch_time'])}")
        print(f"   ⏱️  Temps total:       {format_time(time_stats['total_elapsed'])}")
        print(f"   ⏱️  Temps moyen/epoch: {format_time(time_stats['avg_epoch_time'])}")
        print(f"   ⏳ Temps restant:      {format_time(time_stats['estimated_remaining'])}")
        print(f"   🏁 ETA:                {time_stats['eta'].strftime('%H:%M:%S')}")
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(CONFIG["output_dir"], "best_model.pth"),
                time_stats={'epoch_time': time_stats['epoch_time'], 'total_elapsed': time_stats['total_elapsed']}
            )
            print(f"   ✅ Meilleur modèle sauvegardé!")
        
        # Sauvegardes périodiques
        if (epoch + 1) % CONFIG["save_every"] == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(CONFIG["output_dir"], f"checkpoint_epoch_{epoch+1}.pth"),
                time_stats={'epoch_time': time_stats['epoch_time'], 'total_elapsed': time_stats['total_elapsed']}
            )
            print(f"   💾 Checkpoint epoch {epoch+1} sauvegardé")
    
    # Statistiques finales de temps
    final_time_stats = timer.get_final_stats()
    
    # Ajouter les stats de temps à l'historique
    history['time_stats'] = final_time_stats
    
    # Sauvegarder le modèle final
    save_checkpoint(
        model, optimizer, CONFIG["num_epochs"]-1, val_loss,
        os.path.join(CONFIG["output_dir"], "final_model.pth"),
        time_stats=final_time_stats
    )
    
    # Sauvegarder l'historique complet
    with open(os.path.join(CONFIG["output_dir"], "history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot des courbes de perte
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Courbe de perte
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Courbes de perte - Mask R-CNN Cadastral')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Courbe de temps par epoch
    axes[1].bar(range(1, len(history['epoch_times']) + 1), history['epoch_times'], 
                color='steelblue', alpha=0.7, label='Temps par epoch')
    axes[1].axhline(y=final_time_stats['avg_epoch_time'], color='red', 
                    linestyle='--', linewidth=2, label=f"Moyenne: {final_time_stats['avg_epoch_time_formatted']}")
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Temps (secondes)')
    axes[1].set_title('Temps par epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "training_curves.png"), dpi=150)
    plt.close()
    
    # Rapport final
    print("\n" + "=" * 70)
    print("   🎉 ENTRAÎNEMENT TERMINÉ")
    print("=" * 70)
    print(f"\n📊 RÉSUMÉ DES PERFORMANCES")
    print(f"   {'─' * 50}")
    print(f"   Meilleure Val Loss: {best_val_loss:.4f}")
    print(f"   Train Loss finale:  {history['train_loss'][-1]:.4f}")
    print(f"   Val Loss finale:    {history['val_loss'][-1]:.4f}")
    
    print(f"\n⏱️  RAPPORT DE TEMPS")
    print(f"   {'─' * 50}")
    print(f"   Début:              {final_time_stats['start_datetime']}")
    print(f"   Fin:                {final_time_stats['end_datetime']}")
    print(f"   {'─' * 50}")
    print(f"   ⏱️  Temps total:       {final_time_stats['total_time_formatted']}")
    print(f"   ⏱️  Temps moyen/epoch: {final_time_stats['avg_epoch_time_formatted']}")
    print(f"   ⏱️  Epoch la + rapide: {final_time_stats['min_epoch_time_formatted']}")
    print(f"   ⏱️  Epoch la + lente:  {final_time_stats['max_epoch_time_formatted']}")
    print(f"   📈 Écart-type:         {final_time_stats['std_epoch_time']:.2f}s")
    
    print(f"\n💾 FICHIERS SAUVEGARDÉS")
    print(f"   {'─' * 50}")
    print(f"   📁 Dossier: {CONFIG['output_dir']}")
    print(f"   ├── best_model.pth")
    print(f"   ├── final_model.pth")
    print(f"   ├── checkpoint_epoch_*.pth")
    print(f"   ├── history.json")
    print(f"   └── training_curves.png")
    print("=" * 70)
    
    # Sauvegarder le rapport en fichier texte
    report_path = os.path.join(CONFIG["output_dir"], "training_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("   RAPPORT D'ENTRAÎNEMENT - MASK R-CNN CADASTRAL\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-" * 50 + "\n")
        for key, value in CONFIG.items():
            f.write(f"   {key}: {value}\n")
        
        f.write("\nPERFORMANCES\n")
        f.write("-" * 50 + "\n")
        f.write(f"   Meilleure Val Loss: {best_val_loss:.4f}\n")
        f.write(f"   Train Loss finale:  {history['train_loss'][-1]:.4f}\n")
        f.write(f"   Val Loss finale:    {history['val_loss'][-1]:.4f}\n")
        
        f.write("\nTEMPS D'ENTRAÎNEMENT\n")
        f.write("-" * 50 + "\n")
        f.write(f"   Début:               {final_time_stats['start_datetime']}\n")
        f.write(f"   Fin:                 {final_time_stats['end_datetime']}\n")
        f.write(f"   Temps total:         {final_time_stats['total_time_formatted']}\n")
        f.write(f"   Temps moyen/epoch:   {final_time_stats['avg_epoch_time_formatted']}\n")
        f.write(f"   Epoch la + rapide:   {final_time_stats['min_epoch_time_formatted']}\n")
        f.write(f"   Epoch la + lente:    {final_time_stats['max_epoch_time_formatted']}\n")
        f.write(f"   Écart-type:          {final_time_stats['std_epoch_time']:.2f}s\n")
        
        f.write("\nTEMPS PAR EPOCH\n")
        f.write("-" * 50 + "\n")
        for i, t in enumerate(final_time_stats['epoch_times']):
            f.write(f"   Epoch {i+1:3d}: {format_time(t)}\n")
    
    print(f"\n📄 Rapport sauvegardé: {report_path}")


if __name__ == "__main__":
    main()
