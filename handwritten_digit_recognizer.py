"""Train and evaluate a CNN for handwritten digit recognition on MNIST."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train.astype("float32") / 255.0, -1)
    x_test = np.expand_dims(x_test.astype("float32") / 255.0, -1)
    return (x_train, y_train), (x_test, y_test)

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ], name="mnist_digit_cnn")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def main():
    parser = argparse.ArgumentParser(description="Recognize handwritten digits with a CNN.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive.")

    (x_train, y_train), (x_test, y_test) = load_mnist()
    model = build_model()
    history = model.fit(x_train, y_train, validation_split=0.1,
                        epochs=args.epochs, batch_size=args.batch_size, verbose=1)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    predicted = np.argmax(model.predict(x_test, verbose=0), axis=1)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4%}")
    print(classification_report(y_test, predicted, digits=4))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "mnist_cnn.keras")

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.title("MNIST CNN Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend(); plt.tight_layout()
    plt.savefig(output / "training_accuracy.png", dpi=150); plt.show()

    ConfusionMatrixDisplay(confusion_matrix(y_test, predicted), display_labels=range(10)).plot(
        cmap="Blues", values_format="d"
    )
    plt.title("MNIST Test Confusion Matrix"); plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png", dpi=150); plt.show()

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_test[i].squeeze(), cmap="gray")
        ax.set_title(f"Pred: {predicted[i]} | True: {y_test[i]}")
        ax.axis("off")
    fig.suptitle("Sample MNIST Predictions"); fig.tight_layout()
    fig.savefig(output / "sample_predictions.png", dpi=150); plt.show()
    print(f"Saved model and plots in: {output}")

if __name__ == "__main__":
    main()
