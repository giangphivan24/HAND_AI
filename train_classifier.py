# train_classifier.py
import csv
import numpy as np
import os
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

CSV_PATH = "keypoints_dataset.csv"
H5_PATH = "gesture_classifier.h5"
TFLITE_PATH = "gesture_classifier.tflite"
LABELS_PATH = "labels.txt"

def load_dataset(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader]
    labels = [r[0] for r in rows]
    X = np.array([[float(x) for x in r[1:]] for r in rows], dtype=np.float32)
    return X, labels

def build_model(input_dim, num_classes, use_advanced=True):
    if use_advanced:
        # Improved architecture with better regularization
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            # First layer with batch normalization
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            # Second layer
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            # Third layer
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            # Output layer
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ])
        # Use learning rate scheduling in optimizer
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    else:
        # Original simpler architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ])
        optimizer = "adam"
    
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", "sparse_top_k_categorical_accuracy"]
    )
    return model

def main():
    X, labels = load_dataset(CSV_PATH)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Print dataset statistics
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(X)}")
    print(f"Number of classes: {len(le.classes_)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Class distribution:")
    for cls, count in zip(le.classes_, counts):
        print(f"  {cls}: {count} samples")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Compute class weights for imbalanced datasets
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nClass weights: {class_weight_dict}")

    model = build_model(X.shape[1], len(le.classes_), use_advanced=True)
    
    # Callbacks for better training
    callbacks = [
        # Early stopping to prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # Model checkpointing
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train with more epochs and better settings
    history = model.fit(
        X_train, y_train,
        epochs=100,  # Increased from 25, but early stopping will prevent overfitting
        batch_size=32,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,  # Handle class imbalance
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best model if checkpoint was used
    if os.path.exists('best_model.h5'):
        print("\nLoading best model from checkpoint...")
        model = tf.keras.models.load_model('best_model.h5')
    
    # Evaluate final model
    val_loss, val_accuracy, val_top_k = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal Validation Results:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_accuracy:.4f}")
    print(f"  Top-K Accuracy: {val_top_k:.4f}")
    
    model.save(H5_PATH)

    # Save labels
    with open(LABELS_PATH, "w") as f:
        for c in le.classes_:
            f.write(c + "\n")

    # Export to TFLite - use SavedModel format for better compatibility
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_path = os.path.join(tmpdir, "saved_model")
        model.export(saved_model_path)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        tflite_model = converter.convert()
        with open(TFLITE_PATH, "wb") as f:
            f.write(tflite_model)

    print("Saved:", H5_PATH, TFLITE_PATH, LABELS_PATH)

if __name__ == "__main__":
    main()