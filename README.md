# Image Classification with Transfer Learning (Computer Vision)

Classifying CIFAR-10 images into 10 categories using a pretrained ResNet18.

## Problem

Given a small (32x32) color image, classify it into one of ten categories:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. This
demonstrates transfer learning — adapting a large pretrained network to a new
task with minimal training time.

## Dataset

CIFAR-10: 60,000 images (50,000 train / 10,000 test), auto-downloaded via
torchvision.

## Approach

1. **Data augmentation** — random horizontal flip and rotation on the training
   set to reduce overfitting.
2. **Transfer learning** — loaded a ResNet18 pretrained on ImageNet, froze all
   base convolutional layers, and replaced the final fully-connected layer to
   output 10 classes instead of ResNet's original 1,000.
3. **Training** — fine-tuned only the new final layer for 5 epochs using Adam
   optimizer (GPU-accelerated on Google Colab).
4. **Evaluation** — accuracy, per-class precision/recall/F1, and a confusion
   matrix.

## Results

**Final test accuracy: 79.8%** after only 5 epochs of training (frozen base
layers), running on a Colab T4 GPU.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| airplane | 0.77 | 0.87 | 0.82 |
| automobile | 0.81 | 0.92 | 0.86 |
| bird | 0.79 | 0.72 | 0.76 |
| cat | 0.76 | 0.63 | 0.69 |
| deer | 0.74 | 0.75 | 0.74 |
| dog | 0.70 | 0.81 | 0.75 |
| frog | 0.81 | 0.85 | 0.83 |
| horse | 0.86 | 0.77 | 0.81 |
| ship | 0.94 | 0.78 | 0.85 |
| truck | 0.83 | 0.88 | 0.85 |

## Key Insight

The model performed strongly on classes with distinctive shapes and
backgrounds (ship: 94% precision, automobile: 92% recall), but struggled most
with **cats (63% recall)** — the model missed more cats than any other class,
and the confusion matrix shows cat/dog as the most confused pair. This makes
intuitive sense: cats and dogs share overlapping visual features (fur texture,
body shape, similar poses) at CIFAR-10's low 32x32 resolution, which is a
known, well-documented challenge for this dataset.

## What I'd Improve With More Time

- Unfreeze and fine-tune more of ResNet18's later layers instead of only the
  final classifier layer, which often boosts accuracy further.
- Train for more epochs (10-15) to see if accuracy climbs meaningfully.
- Add Grad-CAM visualizations to show which parts of misclassified cat/dog
  images the model focused on.

## How to Run

```bash
pip install -r requirements.txt
python image_classification_transfer_learning.py
```

GPU strongly recommended (Google Colab free tier works well).

Outputs: `training_history.png`, `cv_confusion_matrix.png`,
`resnet18_cifar10.pth`.
