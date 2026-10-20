# Wheat Plant Disease Classification

A convolutional neural network (TensorFlow/Keras) that identifies wheat diseases from a photo. Give it an image of a wheat plant and it returns the most likely class.

**15 classes:** Aphid, Black Rust, Blast, Brown Rust, Common Root Rot, Fusarium Head Blight, Healthy, Leaf Blight, Mildew, Mite, Septoria, Smut, Stem fly, Tan spot, Yellow Rust.

## Dataset
[Wheat Plant Diseases on Kaggle](https://www.kaggle.com/datasets/kushagra3204/wheat-plant-diseases) (13,104 train / 300 validation images, plus a test split). Download it and extract so the layout is:

```
Data/
  train/<class>/*.jpg
  valid/<class>/*.jpg
  test/<class>/*.jpg
```
The dataset is not included in this repo.

## Setup
```bash
python -m venv venv
venv\Scripts\activate        # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

## Usage
```bash
python make_clean_split.py           # leak-free split into Data_clean/
python train.py --data Data_clean    # trains, saves models/wheat_cnn.keras
python evaluate.py --data Data_clean # test-set report + confusion matrix
python predict.py path/to/photo.jpg  # identify a disease from a photo
```
Example output:
```
Prediction: Brown Rust (93.2% confidence)

  Brown Rust             93.2%
  Yellow Rust             4.1%
  Septoria                1.0%
```

## Model
A small CNN: three Conv(16)+MaxPool blocks, then two Dense(128) layers and a 15-way softmax, on 256x256 images. `wheat_disease_classification.ipynb` is the original exploration; `train.py` trains the same network with early stopping and best-model checkpointing, and optional `--augment` / `--dropout` flags.

**Dataset note:** the Kaggle train/valid/test splits share many identical images, so scores on them are inflated. `make_clean_split.py` removes duplicates and re-splits the data into `Data_clean/`; use that for honest numbers (`python train.py --data Data_clean`).

## Files
| File | Purpose |
|---|---|
| `wheat_disease_classification.ipynb` | Exploration and first training run |
| `train.py` | Training script |
| `evaluate.py` | Test-set evaluation |
| `check_data.py` | Reports images shared between splits |
| `make_clean_split.py` | Builds a de-duplicated 70/15/15 split |
| `predict.py` | Command-line prediction on one image |
| `models/class_names.json` | Class order used by the model |

## License
MIT