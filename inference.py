"""
Inférence Mask R-CNN - Prédiction sur nouvelles images
Segmentation des toitures cadastrales
"""

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import v2 as T
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import argparse
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

CLASSES = [
    "__background__",
    "toiture_tole_ondulee",
    "toiture_tole_bac", 
    "toiture_tuile",
    "toiture_dalle"
]

# Couleurs pour chaque classe (RGB)
COLORS = {
    "toiture_tole_ondulee": (255, 0, 0),      # Rouge
    "toiture_tole_bac": (0, 255, 0),          # Vert
    "toiture_tuile": (0, 0, 255),             # Bleu
    "toiture_dalle": (255, 165, 0),           # Orange
}


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


def load_model(checkpoint_path, device):
    """Charger le modèle depuis un checkpoint"""
    model = get_model(len(CLASSES))
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Modèle chargé depuis: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}, Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    
    return model


# =============================================================================
# INFÉRENCE
# =============================================================================

def predict(model, image_path, device, score_threshold=0.5):
    """Prédire sur une image"""
    
    # Charger l'image
    image = Image.open(image_path).convert("RGB")
    image_tensor = T.ToTensor()(image)
    
    # Inférence
    with torch.no_grad():
        predictions = model([image_tensor.to(device)])
    
    pred = predictions[0]
    
    # Filtrer par score
    keep = pred['scores'] > score_threshold
    
    result = {
        'boxes': pred['boxes'][keep].cpu().numpy(),
        'labels': pred['labels'][keep].cpu().numpy(),
        'scores': pred['scores'][keep].cpu().numpy(),
        'masks': pred['masks'][keep].cpu().numpy(),
    }
    
    return image, result


def calculate_surface(mask, pixel_size_m2=None):
    """
    Calculer la surface d'un masque
    
    Args:
        mask: Masque binaire (H, W)
        pixel_size_m2: Taille d'un pixel en m² (si connu depuis métadonnées de l'image)
    
    Returns:
        Surface en pixels ou en m² si pixel_size_m2 est fourni
    """
    surface_pixels = np.sum(mask > 0.5)
    
    if pixel_size_m2 is not None:
        return surface_pixels * pixel_size_m2
    return surface_pixels


# =============================================================================
# VISUALISATION
# =============================================================================

def visualize_predictions(image, predictions, output_path=None, show=True):
    """Visualiser les prédictions avec masques et boîtes"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Image originale
    axes[0].imshow(image)
    axes[0].set_title("Image originale")
    axes[0].axis('off')
    
    # Image avec prédictions
    axes[1].imshow(image)
    
    masks = predictions['masks']
    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']
    
    # Créer un overlay pour les masques
    overlay = np.zeros((*np.array(image).shape[:2], 4))
    
    for i, (mask, box, label, score) in enumerate(zip(masks, boxes, labels, scores)):
        class_name = CLASSES[label]
        color = COLORS.get(class_name, (128, 128, 128))
        color_normalized = [c/255 for c in color]
        
        # Masque
        mask_binary = mask[0] > 0.5
        overlay[mask_binary] = [*color_normalized, 0.5]
        
        # Boîte
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2,
            edgecolor=color_normalized,
            facecolor='none'
        )
        axes[1].add_patch(rect)
        
        # Label avec surface
        surface = calculate_surface(mask[0])
        label_text = f"{class_name}\n{score:.2f} | {surface:,} px"
        axes[1].text(
            x1, y1-10,
            label_text,
            fontsize=8,
            color='white',
            bbox=dict(boxstyle='round', facecolor=color_normalized, alpha=0.8)
        )
    
    axes[1].imshow(overlay)
    axes[1].set_title(f"Prédictions ({len(masks)} objets détectés)")
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Résultat sauvegardé: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


def export_masks(predictions, output_dir, image_name):
    """Exporter les masques individuels en PNG"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (mask, label, score) in enumerate(zip(
        predictions['masks'], 
        predictions['labels'],
        predictions['scores']
    )):
        class_name = CLASSES[label]
        mask_binary = (mask[0] > 0.5).astype(np.uint8) * 255
        
        mask_image = Image.fromarray(mask_binary)
        mask_path = os.path.join(
            output_dir, 
            f"{image_name}_{i:02d}_{class_name}_{score:.2f}.png"
        )
        mask_image.save(mask_path)
    
    print(f"Masques exportés dans: {output_dir}")


def generate_report(predictions, image_name):
    """Générer un rapport des surfaces détectées"""
    
    report = {
        'image': image_name,
        'total_objects': len(predictions['labels']),
        'surfaces_by_class': {},
        'details': []
    }
    
    for class_name in CLASSES[1:]:  # Ignorer background
        report['surfaces_by_class'][class_name] = {
            'count': 0,
            'total_surface_px': 0
        }
    
    for i, (mask, label, score, box) in enumerate(zip(
        predictions['masks'],
        predictions['labels'],
        predictions['scores'],
        predictions['boxes']
    )):
        class_name = CLASSES[label]
        surface = calculate_surface(mask[0])
        
        report['surfaces_by_class'][class_name]['count'] += 1
        report['surfaces_by_class'][class_name]['total_surface_px'] += surface
        
        report['details'].append({
            'id': i,
            'class': class_name,
            'score': float(score),
            'surface_px': int(surface),
            'bbox': box.tolist()
        })
    
    return report


def print_report(report):
    """Afficher le rapport"""
    print("\n" + "=" * 50)
    print(f"RAPPORT DE SEGMENTATION - {report['image']}")
    print("=" * 50)
    print(f"Total objets détectés: {report['total_objects']}")
    print("\nSurfaces par classe:")
    print("-" * 50)
    
    for class_name, data in report['surfaces_by_class'].items():
        if data['count'] > 0:
            print(f"  {class_name}:")
            print(f"    - Nombre: {data['count']}")
            print(f"    - Surface totale: {data['total_surface_px']:,} pixels")
    
    print("\nDétails:")
    print("-" * 50)
    for obj in report['details']:
        print(f"  [{obj['id']}] {obj['class']} (conf: {obj['score']:.2f}) - {obj['surface_px']:,} px")
    
    print("=" * 50)


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_directory(model, input_dir, output_dir, device, score_threshold=0.5):
    """Traiter toutes les images d'un répertoire"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    image_paths = [
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in image_extensions
    ]
    
    print(f"\nTraitement de {len(image_paths)} images...")
    
    all_reports = []
    
    for img_path in image_paths:
        print(f"\nTraitement: {img_path.name}")
        
        image, predictions = predict(model, str(img_path), device, score_threshold)
        
        # Visualisation
        output_path = os.path.join(output_dir, f"{img_path.stem}_pred.png")
        visualize_predictions(image, predictions, output_path, show=False)
        
        # Rapport
        report = generate_report(predictions, img_path.name)
        all_reports.append(report)
        print_report(report)
    
    # Sauvegarder tous les rapports
    import json
    reports_path = os.path.join(output_dir, "reports.json")
    with open(reports_path, 'w') as f:
        json.dump(all_reports, f, indent=2)
    
    print(f"\nRapports sauvegardés: {reports_path}")
    
    return all_reports


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Inférence Mask R-CNN Cadastral")
    parser.add_argument("--model", type=str, required=True, help="Chemin vers le checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Image ou dossier d'images")
    parser.add_argument("--output", type=str, default="./predictions", help="Dossier de sortie")
    parser.add_argument("--threshold", type=float, default=0.5, help="Seuil de confiance")
    parser.add_argument("--export-masks", action="store_true", help="Exporter les masques individuels")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Charger le modèle
    model = load_model(args.model, device)
    
    # Traitement
    input_path = Path(args.input)
    
    if input_path.is_dir():
        # Traiter un dossier
        process_directory(model, str(input_path), args.output, device, args.threshold)
    else:
        # Traiter une seule image
        os.makedirs(args.output, exist_ok=True)
        
        image, predictions = predict(model, str(input_path), device, args.threshold)
        
        output_path = os.path.join(args.output, f"{input_path.stem}_pred.png")
        visualize_predictions(image, predictions, output_path)
        
        if args.export_masks:
            export_masks(predictions, os.path.join(args.output, "masks"), input_path.stem)
        
        report = generate_report(predictions, input_path.name)
        print_report(report)


if __name__ == "__main__":
    main()
