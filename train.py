"""
train.py — Brain Tumor Detection Model Training
================================================
Uses VGG16 transfer learning to classify MRI scans as tumor/no-tumor.

Dataset structure expected:
    dataset/
        yes/   ← MRI images WITH tumor
        no/    ← MRI images WITHOUT tumor

Run: python train.py
Output: model/brain_tumor_model.h5
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ─── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE    = 224          # VGG16 expects 224×224
BATCH_SIZE  = 16
EPOCHS      = 20           # EarlyStopping will cut this short if needed
DATASET_DIR = "dataset"
MODEL_DIR   = "model"
MODEL_PATH  = os.path.join(MODEL_DIR, "brain_tumor_model.h5")

# ─── 1. Load images from dataset/yes and dataset/no ──────────────────────────
print("[INFO] Loading dataset...")

images = []
labels = []

for label, folder in enumerate(["no", "yes"]):          # 0 = no tumor, 1 = tumor
    folder_path = os.path.join(DATASET_DIR, folder)
    if not os.path.exists(folder_path):
        raise FileNotFoundError(
            f"Missing folder: {folder_path}\n"
            "Please add MRI images inside dataset/yes/ and dataset/no/"
        )
    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"  Found {len(files)} images in '{folder}/'")
    for fname in files:
        fpath = os.path.join(folder_path, fname)
        try:
            img = load_img(fpath, target_size=(IMG_SIZE, IMG_SIZE))
            arr = img_to_array(img) / 255.0   # normalize to [0, 1]
            images.append(arr)
            labels.append(label)
        except Exception as e:
            print(f"  [WARN] Skipping {fname}: {e}")

X = np.array(images)
y = np.array(labels)
print(f"[INFO] Total samples: {len(X)}  |  Tumor: {y.sum()}  |  No Tumor: {(y==0).sum()}")

# ─── 2. Train / Validation split ──────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[INFO] Train: {len(X_train)}  |  Val: {len(X_val)}")

# ─── 3. Data Augmentation (helps prevent overfitting on small datasets) ────────
train_gen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
)
train_gen.fit(X_train)

# ─── 4. Build model with VGG16 as feature extractor ───────────────────────────
print("[INFO] Building VGG16 transfer learning model...")

base_model = VGG16(
    weights="imagenet",       # pretrained on ImageNet
    include_top=False,        # remove VGG16's classifier head
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze all VGG16 layers — we only train the new head first
for layer in base_model.layers:
    layer.trainable = False

# Custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)   # binary output

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()

# ─── 5. Callbacks ─────────────────────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
]

# ─── 6. Phase 1: Train only the new head ──────────────────────────────────────
print("\n[PHASE 1] Training classification head (VGG16 frozen)...")
history1 = model.fit(
    train_gen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

# ─── 7. Phase 2: Fine-tune top VGG16 layers ───────────────────────────────────
print("\n[PHASE 2] Fine-tuning top VGG16 layers...")

# Unfreeze the last 4 convolutional layers
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Recompile with a lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

history2 = model.fit(
    train_gen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=10,
    callbacks=callbacks,
    verbose=1,
)

# ─── 8. Evaluate on validation set ────────────────────────────────────────────
print("\n[INFO] Evaluating model...")
loss, accuracy, auc = model.evaluate(X_val, y_val, verbose=0)
print(f"  Validation Accuracy : {accuracy*100:.2f}%")
print(f"  Validation AUC      : {auc:.4f}")
print(f"  Validation Loss     : {loss:.4f}")

y_pred = (model.predict(X_val) > 0.5).astype(int).flatten()
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=["No Tumor", "Tumor"]))

# ─── 9. Plot training curves ───────────────────────────────────────────────────
def merge_history(h1, h2, key):
    return h1.history.get(key, []) + h2.history.get(key, [])

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(merge_history(history1, history2, "accuracy"), label="Train Accuracy")
plt.plot(merge_history(history1, history2, "val_accuracy"), label="Val Accuracy")
plt.title("Model Accuracy"); plt.xlabel("Epoch"); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(merge_history(history1, history2, "loss"), label="Train Loss")
plt.plot(merge_history(history1, history2, "val_loss"), label="Val Loss")
plt.title("Model Loss"); plt.xlabel("Epoch"); plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"))
print(f"\n[INFO] Training curves saved to {MODEL_DIR}/training_curves.png")
print(f"[INFO] Model saved to {MODEL_PATH}")
print("\n✅ Training complete!")
