# Plant Disease Recognition

A deep learning model that identifies plant species and diagnoses leaf diseases from a single photo, covering 39 classes across 14 plant types. Built with TensorFlow/Keras using transfer learning on EfficientNetB4, reaching **97.8% test accuracy**.

## Overview

Given a photo of a leaf, the model predicts both the plant species and its health condition for example `Tomato___Late_blight`, `Apple___healthy`, or `Grape___Black_rot`. Rather than training a convolutional network from scratch, this project fine-tunes **EfficientNetB4**, a model pre-trained on ImageNet, adapting its learned visual features to the specific task of leaf disease recognition. This transfer learning approach dramatically reduces the data and compute needed to reach high accuracy compared to training from random initialization.

## Results

| Metric | Score |
|---|---|
| Test Accuracy | **97.79%** |
| Test Loss | 0.0697 |
| Validation Accuracy (final) | 97.43% |
| Training Accuracy (final) | 96.30% |

The close gap between training and validation accuracy indicates the model generalizes well rather than overfitting to the training set.

## Dataset

- **Source**: [Plant Leaf Diseases Dataset with Augmentation](https://data.mendeley.com/public-files/datasets/tywbtsjrjv) (Mendeley Data)
- **Classes**: 39 total healthy and diseased variants across Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato, plus a background/no-leaf class
- **Split**: 80% train / 10% validation / 10% test, using `split-folders` for a stratified, reproducible split
- **Volume**: ~61,500 images total → 49,179 training / 6,139 validation / 6,168 test images

## Model Architecture

```
Input (160 × 160 × 3)
        │
        ▼
EfficientNetB4 preprocessing
        │
        ▼
EfficientNetB4 base (ImageNet weights, top removed)
        │  → outputs (5, 5, 1792) feature maps
        ▼
GlobalAveragePooling2D → (1792,)
        │
        ▼
Dropout (0.2)
        │
        ▼
Dense (39 units) → class prediction
```

**Total parameters**: ~17.7M of which only ~69K (the classification head) are trained in the initial phase, with the EfficientNetB4 backbone fully frozen.

## Training Strategy

The model is trained in **two phases**, a standard and effective transfer learning pattern:

### Phase 1 — Feature Extraction
The pre-trained EfficientNetB4 backbone is completely frozen (`trainable = False`), and only the new classification head (pooling + dropout + dense layer) is trained. This lets the model quickly learn to map EfficientNet's general-purpose visual features onto the 39 leaf disease classes without disturbing the pre-trained weights.

- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy
- **Result after Phase 1**: 90.4% training accuracy → 91.5% validation accuracy

### Phase 2 — Fine-Tuning
The backbone is unfrozen, but the first 100 of its 475 layers remain frozen preserving EfficientNet's low-level, general-purpose visual features (edges, textures, colors) while allowing the deeper, more task-specific layers to adapt to leaf-specific patterns. The model is recompiled with a fresh optimizer state and trained for additional epochs at a lower effective learning rate.

- **Layers fine-tuned**: 375 of 475 (from layer 100 onward)
- **Result after Phase 2**: 96.3% training accuracy → 97.4% validation accuracy → **97.8% test accuracy**

This two-phase approach freeze first, then selectively unfreeze avoids destroying the useful pre-trained features early in training, which is a common failure mode when fine-tuning starts from randomly initialized weights on top of a frozen backbone.

## Tech stack

- Python
- TensorFlow / Keras
- EfficientNetB4 (`tf.keras.applications.efficientnet`)
- NumPy, Matplotlib
- `split-folders` (dataset splitting)

## Running it

This project was developed and trained in Google Colab (GPU runtime recommended for training).

```bash
# Install dependencies
pip install tensorflow split-folders matplotlib numpy

# Download and unzip the dataset
wget -O dataset.zip "<mendeley-dataset-url>"
unzip dataset.zip

# Split into train/val/test
python -c "import splitfolders; splitfolders.ratio('Plant_leave_diseases_dataset_with_augmentation', output='dataset', seed=1337, ratio=(.8, .1, .1))"

# Run the notebook
jupyter notebook Plant_Disease_Recognition.ipynb
```

To use the trained model directly for inference:

```python
import tensorflow as tf

model = tf.keras.models.load_model("Plant_Disease_Recognition_Model.keras")
predictions = model.predict(image_batch)
predicted_class = class_names[tf.argmax(predictions[0])]
```

## Project structure

```
├── Plant_Disease_Recognition.ipynb   # Full training pipeline
├── Plant_Disease_Recognition_Model.keras   # Saved trained model
├── requirements.txt
└── README.md
```

## Key design decisions

- **Why EfficientNetB4** — offers a strong accuracy-to-compute tradeoff among the EfficientNet family, well suited to fine-grained visual classification tasks like leaf texture and lesion patterns without requiring excessive training time.
- **Why freeze-then-fine-tune instead of training end-to-end from the start** — training a randomly initialized classification head jointly with an unfrozen pre-trained backbone risks large, destructive gradient updates to the backbone early on. Training the head first stabilizes it before any backbone weights are touched.
- **Why only unfreeze from layer 100 onward** — the earliest layers of a CNN learn generic, transferable features (edges, colors, simple textures) that are broadly useful regardless of task. Keeping these frozen preserves that transferable knowledge while letting deeper, more abstract layers specialize.

## Limitations

- Trained and evaluated on a dataset composed largely of controlled, single-leaf images against plain backgrounds real-world field photos with cluttered backgrounds, poor lighting, or multiple leaves may reduce accuracy
- The 39 classes are fixed at training time; the model cannot recognize diseases or plant species outside this label set
- Class balance across the 39 categories was not explicitly analyzed here some diseases may be represented by more images than others, which can bias predictions toward better-represented classes
- The output layer uses a sigmoid activation with categorical crossentropy loss; a softmax activation is the more conventional choice for single-label multi-class classification and may be worth revisiting

## Future improvements

- Evaluate per-class precision/recall to identify which specific diseases the model struggles with most
- Test on real-world, uncontrolled field photos to assess generalization beyond the training distribution
- Experiment with data augmentation at training time (rotation, brightness, occlusion) to improve robustness
- Package the model behind a simple web or mobile interface for practical field use
