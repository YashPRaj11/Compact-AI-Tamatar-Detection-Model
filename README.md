# Lightweight Tomato Detection Model 🍅

A compact, custom object-detection model for detecting and classifying tomatoes as **raw** or **ripe**.

This project is the result of **many rounds of experimentation, modification, debugging, and trial-and-error**. The main goal was not simply to build a detector, but to reduce the model footprint while keeping a practical multi-scale detection pipeline.

## ⭐ Key Result: Compact Model

| Property | Result |
|---|---:|
| Input size | `416 × 416 × 3` |
| Classes | 2 (`raw`, `ripe`) |
| Detection scales | 3 |
| Anchors per scale | **1** |
| Trainable model parameters | **372,685** |
| Saved H5 model size | ~**4.4 MB** |
| TFLite model size | **1,508,620 bytes (~1.44 MiB / 1.51 MB)** |
| Training epochs in the recorded run | 50 |

The **372K-parameter** model is deliberately much smaller than a conventional YOLO-style detector. The final TFLite artifact is only about **1.51 MB**, making the model suitable for experiments involving lightweight or edge/mobile deployment.

> **Important:** The H5 file contains optimizer/training information, so its file size is larger than the deployable TFLite model. The recorded TFLite export is `1,508,620` bytes.

---

## Why This Model Is Compact

The compact size was achieved through several architectural choices and repeated experimentation.

### 1. One anchor per detection scale

Instead of using multiple anchors at every detection scale, the final model uses:

- **Fine:** 1 anchor
- **Medium:** 1 anchor
- **Coarse:** 1 anchor

This gives only **3 anchor boxes in total**.

The implementation explicitly validates that every detection scale uses the same number of anchors and stores one anchor for each of the three scales.

### 2. Lightweight backbone

The backbone uses:

- Standard convolution blocks only where useful
- **Depthwise separable-style blocks**
- Batch normalization
- Swish activation
- Progressive downsampling

The depthwise block separates spatial filtering from channel mixing, substantially reducing computation compared with a conventional convolution using the same number of channels.

### 3. Small feature-channel widths

The backbone intentionally uses relatively small channel sizes:

```text
Input: 416 × 416
        ↓
208 × 208
        ↓
104 × 104
        ↓
52 × 52   → Fine
        ↓
26 × 26   → Medium
        ↓
13 × 13   → Coarse
```

The detector therefore preserves three spatial scales without requiring a large backbone.

### 4. Lightweight feature pyramid

A small top-down FPN is used to combine information from different resolutions.

The feature pyramid produces:

```text
P3 → 52 × 52
P4 → 26 × 26
P5 → 13 × 13
```

Nearest-neighbor upsampling is used instead of a heavier learned upsampling operation.

### 5. Minimal detection heads

Each detection head predicts:

```text
x
y
w
h
objectness
class probabilities
```

With:

```text
1 anchor × (5 + 2 classes) = 7 values
```

Therefore the output tensors are:

```text
Fine:   (52, 52, 1, 7)
Medium: (26, 26, 1, 7)
Coarse: (13, 13, 1, 7)
```

---

# Model Architecture

The main model is implemented in `model_architecture.py`.

## Backbone

```text
                 Input
            416 × 416 × 3
                    │
                    ▼
              Conv + BN + Swish
                    │
              208 × 208
                    │
                    ▼
              Conv + BN + Swish
                    │
              104 × 104
                    │
                    ▼
        Depthwise Conv Block × 2
                    │
              52 × 52
                    │
                    ├──────────────► Fine
                    │
                    ▼
        Depthwise Conv Block × 2
                    │
              26 × 26
                    │
                    ├──────────────► Medium
                    │
                    ▼
        Depthwise Conv Block × 2
                    │
              13 × 13
                    │
                    └──────────────► Coarse
```

A lightweight top-down feature pyramid then refines these feature maps before the detection heads.

---

# Detection Scales

The detector uses three scales to handle objects at different apparent sizes.

| Scale | Grid | Role |
|---|---:|---|
| Fine | `52 × 52` | Smaller / detailed objects |
| Medium | `26 × 26` | Medium objects |
| Coarse | `13 × 13` | Larger objects |

The final model uses one anchor at each scale.

Default anchors:

```text
Fine:   [0.10, 0.13]
Medium: [0.28, 0.35]
Coarse: [0.58, 0.64]
```

Anchors are represented in normalized width/height coordinates.

---

# Data Pipeline

`data_loader.py` handles image loading, annotation parsing, preprocessing, target generation, and augmentation.

## Supported annotations

The loader can extract bounding boxes from several annotation formats:

- `bbox`
- `points`
- `polygon`
- `keypoints`

Class names can also be automatically detected from the annotation files.

For the recorded dataset:

```text
Images: 398
Classes:
    raw
    ripe
```

---

## Letterbox Preprocessing

Images are resized to:

```text
416 × 416
```

while preserving the original aspect ratio.

Padding is added when required, and the transformation metadata is retained so predicted boxes can later be mapped back to the original image coordinates.

---

# Anchor Estimation

The data loader contains an anchor-estimation procedure based on **IoU-based k-means-style clustering** over normalized ground-truth widths and heights.

The process:

1. Collect normalized bounding-box widths and heights.
2. Select cluster centers.
3. Calculate width/height IoU.
4. Assign boxes to the best matching anchor.
5. Update cluster centers using the median box dimensions.
6. Repeat until convergence or the iteration limit.
7. Sort anchors by area.
8. Assign them to fine, medium, and coarse scales.

The final configuration uses:

```python
num_anchors_per_scale = 1
```

which produces three anchors in total.

---

# Target Generation

Ground-truth boxes are converted into scale-specific tensors.

For each object, the loader:

1. Finds the best matching anchor using width/height IoU.
2. Selects the corresponding detection scale.
3. Determines the grid cell containing the object's center.
4. Encodes the center offset.
5. Encodes width and height relative to the selected anchor.
6. Sets objectness.
7. Sets the corresponding class target.

The target tensor shape for the recorded dataset is:

```text
Fine:   (398, 52, 52, 1, 7)
Medium: (398, 26, 26, 1, 7)
Coarse: (398, 13, 13, 1, 7)
```

---

# Data Augmentation

The `DataGenerator` provides label-consistent augmentation.

Implemented transformations include:

- Horizontal flipping
- Random brightness adjustment
- Random contrast adjustment

For horizontal flipping, the bounding-box grid assignment and encoded x-offset are updated together so that the labels remain geometrically consistent with the transformed image.

---

# Loss Function

The model uses a custom scale-specific loss.

The loss combines:

### Bounding-box regression

Uses **Complete IoU (CIoU)** to improve localization.

### Objectness

Uses sigmoid cross-entropy with an IoU-aware positive target.

### Classification

For multiple classes, softmax cross-entropy is used.

The final loss combines:

```text
Total Loss =
    10.0 × Box Loss
  + 5.0 × Positive Objectness Loss
  + 0.40 × Negative Objectness Loss
  + 1.0 × Classification Loss
```

Gradient clipping is also used with:

```text
clipnorm = 5.0
```

and the recorded model is trained with Adam.

---

# Training

The recorded training pipeline used:

```text
Dataset:       398 images
Input:         416 × 416
Classes:       2
Batch size:    16
Epochs:        50
Detection:     3 scales
Anchors:       1 per scale
```

The training run reached epoch 50, with the final recorded total training loss around `2.9148` and validation loss around `6.5305`.

These loss values are reported from the recorded training run; they should not be interpreted as mAP, precision, recall, or classification accuracy.

---

# Inference

The model performs the following inference pipeline:

```text
Input Image
     │
     ▼
RGB conversion
     │
     ▼
Letterbox → 416 × 416
     │
     ▼
Normalize /255
     │
     ▼
Lightweight CNN
     │
     ├── Fine
     ├── Medium
     └── Coarse
          │
          ▼
     Decode predictions
          │
          ▼
     Confidence filtering
          │
          ▼
     Class-aware NMS
          │
          ▼
     Original-image boxes
```

The inference API exposes:

```python
predict_objects(
    model,
    image_path,
    confidence_threshold=0.25,
    iou_threshold=0.5,
    max_detections=100
)
```

Confidence and IoU thresholds can therefore be adjusted without changing the trained network.

---

# TensorFlow Lite Export

The trained model can be exported to TensorFlow Lite through SavedModel.

The recorded workflow was:

```text
Keras/H5
   ↓
Load custom Lightweight model
   ↓
Build model with a dummy input
   ↓
Export SavedModel
   ↓
TFLite Converter
   ↓
model.tflite
```

The recorded output size was:

```text
1,508,620 bytes
```

or approximately:

```text
1.51 MB
```

This is the key deployment artifact when model footprint matters.

---

# Project Files

```text
.
├── model_architecture.py
├── data_loader.py
├── preprocess_3.ipynb
├── model.h5
└── model.tflite
```

### `model_architecture.py`

Contains:

- Lightweight model
- Backbone
- Feature pyramid
- Detection heads
- Prediction decoder
- CIoU calculation
- Custom `ScaleLoss`
- NMS
- Inference function
- Model creation

### `data_loader.py`

Contains:

- Image loading
- JSON annotation parsing
- Letterbox preprocessing
- Bounding-box conversion
- Anchor estimation
- Target tensor generation
- Data augmentation
- Batch generation
- Visualization utilities

### `preprocess_3.ipynb`

Contains the experimental/training workflow, dataset loading, model training, visualization, inference experiments, model saving, and TFLite conversion.

### `model.h5`

Saved trained model in HDF5 format.

### `model.tflite`

Compact TensorFlow Lite deployment artifact.

---

# What Took the Most Iteration

The final compact model was not produced in a single attempt.

The development process involved repeated experimentation around:

- Number of detection anchors
- Detection scales
- Backbone channel sizes
- Lightweight convolution design
- Feature-pyramid structure
- Upsampling
- Anchor assignment
- Bounding-box encoding
- CIoU localization
- Objectness loss
- Classification loss
- Confidence threshold
- IoU/NMS threshold
- Model serialization
- TensorFlow Lite conversion

The final architecture reflects these iterations: **the objective was to remove unnecessary complexity while retaining the three-scale detection mechanism.**

---

# Compactness Summary

The important numbers are:

```text
Input resolution       : 416 × 416
Detection scales       : 3
Anchors per scale      : 1
Total anchors          : 3
Classes                : 2
Trainable parameters   : 372,685
TFLite size            : 1,508,620 bytes
                         ≈ 1.51 MB
```

For a custom three-scale object detector, the final result is intentionally focused on a **small model footprint and straightforward edge deployment** rather than a large backbone.

---

# Current Scope

The current implementation is designed for tomato detection/classification with:

```text
raw tomato
ripe tomato
```

It is a research/prototype implementation and should be evaluated with a dedicated held-out test set and standard object-detection metrics such as:

- Precision
- Recall
- IoU
- mAP@0.5
- mAP@0.5:0.95
- Inference latency
- Memory usage

before making claims about real-world deployment performance.

---

## Acknowledgement of the Development Process

This model represents an **iterative engineering process rather than a single architecture choice**.

The compact footprint was achieved through repeated trials, failures, modifications, and validation of different design decisions. The final **372K-parameter architecture and ~1.51 MB TFLite model** are the result of progressively simplifying the detector while preserving multi-scale prediction, bounding-box regression, objectness estimation, and two-class classification.
