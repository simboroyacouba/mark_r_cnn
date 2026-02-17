# Mask R-CNN - Segmentation des Toitures Cadastrales

Projet de segmentation d'instance pour la détection et classification automatique des types de toitures à partir d'images aériennes.

## Classes détectées

| Classe | Description |
|--------|-------------|
| `toiture_tole_ondulee` | Toiture en tôle ondulée |
| `toiture_tole_bac` | Toiture en tôle bac |
| `toiture_tuile` | Toiture en tuiles |
| `toiture_dalle` | Toiture en dalle béton |

## Installation

### 1. Créer l'environnement

```bash
# Avec conda (recommandé)
conda create -n maskrcnn python=3.10
conda activate maskrcnn

# Ou avec venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 2. Installer PyTorch

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU seulement
pip install torch torchvision
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

## Préparation des données

### Export depuis CVAT

1. Dans CVAT, ouvrir votre projet/tâche
2. Menu → **Export dataset**
3. Format: **COCO 1.0**
4. Télécharger le ZIP

### Structure attendue

```
dataset/
├── images/
|   ├── default/
│   |   ├── image_001.jpg
│   |   ├── image_002.jpg
│   |   └── ...
└── annotations/
    └── instances_default.json
```

### Vérifier le dataset

```bash
python verify_dataset.py --images data/images --annotations data/annotations/instances_default.json --visualize
```

## Entraînement

### Configuration

Modifier `train.py` si nécessaire :

```python
CONFIG = {
    "images_dir": "./data/images",
    "annotations_file": "./data/annotations/instances_default.json",
    "output_dir": "./output",
    
    "num_epochs": 25,
    "batch_size": 2,        # Augmenter si GPU > 8GB
    "learning_rate": 0.005,
    ...
}
```

### Lancer l'entraînement

```bash
python train.py
```

### Sortie

```
output/
├── best_model.pth          # Meilleur modèle (val loss minimale)
├── final_model.pth         # Modèle final
├── checkpoint_epoch_*.pth  # Checkpoints intermédiaires
├── history.json            # Historique des pertes
└── loss_curves.png         # Graphique des courbes de perte
```

## Inférence

### Sur une seule image

```bash
python inference.py \
    --model output/best_model.pth \
    --input test_image.jpg \
    --output predictions/ \
    --threshold 0.5
```

### Sur un dossier d'images

```bash
python inference.py \
    --model output/best_model.pth \
    --input test_images/ \
    --output predictions/ \
    --threshold 0.5 \
    --export-masks
```

### Sortie

```
predictions/
├── image_pred.png          # Visualisation avec masques
├── masks/                  # Masques individuels (si --export-masks)
│   ├── image_00_toiture_tole_ondulee_0.95.png
│   └── ...
└── reports.json            # Rapport avec surfaces
```

## Calcul des surfaces

Le script d'inférence calcule automatiquement :
- Surface en **pixels** de chaque toiture détectée
- Répartition par type de toiture

Pour convertir en **m²**, il faut connaître la résolution de vos images (GSD - Ground Sample Distance).

Exemple avec GSD = 0.1 m/pixel :

```python
pixel_size_m2 = 0.1 * 0.1  # = 0.01 m²/pixel
surface_m2 = surface_pixels * pixel_size_m2
```

## Conseils pour améliorer les performances

### Avec 200 images

1. **Augmentation de données** : Déjà incluse (flips horizontal/vertical)

2. **Transfer learning** : Le modèle utilise des poids pré-entraînés sur COCO

3. **Ajuster les hyperparamètres** :
   - Si overfitting → réduire `num_epochs`, augmenter `weight_decay`
   - Si underfitting → augmenter `num_epochs`, réduire `learning_rate`

4. **Ajouter plus d'augmentation** :
```python
# Dans train.py, ajouter dans get_transforms():
transforms.append(RandomBrightness(0.2))
transforms.append(RandomContrast(0.2))
```

5. **Augmenter le dataset** :
   - Annoter plus d'images variées
   - Inclure différentes conditions (ombres, saisons)

## Dépannage

### CUDA out of memory
Réduire `batch_size` dans CONFIG (1 ou 2)

### Images manquantes
Vérifier que les noms de fichiers correspondent entre `instances_default.json` et le dossier images

### Mauvaises détections
- Augmenter `score_threshold` dans l'inférence
- Vérifier la qualité des annotations dans CVAT
- Entraîner plus longtemps

## Auteur

Projet de thèse - Exploitation de l'IA pour l'évaluation cadastrale automatisée
Burkina Faso - SYCAD/DGI
