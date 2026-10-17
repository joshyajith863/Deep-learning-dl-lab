import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.utils import to_categorical



N_TRAIN = 5000      
N_TEST = 1000       
IMG_SIZE = 224      
BATCH_SIZE = 32
EPOCHS = 15         
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)



(x_train, y_train), (x_test, y_test) = mnist.load_data()

if N_TRAIN is not None:
    idx = np.random.permutation(len(x_train))[:N_TRAIN]
    x_train, y_train = x_train[idx], y_train[idx]

if N_TEST is not None:
    idx = np.random.permutation(len(x_test))[:N_TEST]
    x_test, y_test = x_test[idx], y_test[idx]


x_test_gray = x_test.copy()
y_test_labels = y_test.copy()

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"Train: {x_train.shape}  Test: {x_test.shape}")



def preprocess_image(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = preprocess_input(image)          
    return image, label


def make_dataset(x, y):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



train_dataset = make_dataset(x_train, y_train)
test_dataset = make_dataset(x_test, y_test)




base_model = VGG19(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

feature_extractor = Sequential([
    base_model,
    GlobalAveragePooling2D()
])

print("\nExtracting training features...")
train_features = feature_extractor.predict(train_dataset, verbose=1)

print("Extracting test features...")
test_features = feature_extractor.predict(test_dataset, verbose=1)

print(f"Feature shape: {train_features.shape}")   # (N, 512)

FEATURE_DIM = train_features.shape[1]



def build_head(units):
    return Sequential([
        Input(shape=(FEATURE_DIM,)),
        Dense(units, activation="relu"),
        Dropout(0.3),
        Dense(10, activation="softmax")
    ])


def train_head(units, name):
    print(f"\n{'=' * 40}\nTraining {name} (Dense {units})\n{'=' * 40}")
    model = build_head(units)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    history = model.fit(
        train_features, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,     
        shuffle=True,
        verbose=1
    )
    loss, acc = model.evaluate(test_features, y_test, verbose=0)
    return model, history, loss, acc


model1, hist1, loss1, acc1 = train_head(128, "MODEL 1")
model2, hist2, loss2, acc2 = train_head(256, "MODEL 2")



pred1 = model1.predict(test_features[:2], verbose=0)
pred2 = model2.predict(test_features[:2], verbose=0)



def show_results(name, preds, accuracy, loss):
    print(f"\n{'=' * 30}\n{name} RESULTS\n{'=' * 30}")
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Loss:", round(loss, 4))

    plt.figure(figsize=(8, 4))

    for i in range(2):
        predicted = preds[i].argmax()
        actual = y_test_labels[i]
        vector = y_test[i].astype(int)
        vector_text = "[" + ",".join(map(str, vector)) + "]"

        plt.subplot(1, 2, i + 1)
       
        plt.imshow(x_test_gray[i], cmap="gray")
        plt.title(
            f"Predicted: {predicted}\n"
            f"Actual: {actual}\n"
            f"One-Hot: {vector_text}",
            fontsize=9
        )
        plt.axis("off")

    plt.suptitle(f"{name} Predictions")
    plt.tight_layout()
    plt.show()


show_results("MODEL 1", pred1, acc1, loss1)
show_results("MODEL 2", pred2, acc2, loss2)


print(f"\n{'=' * 40}")
print("COMPARISON")
print(f"{'=' * 40}")
print(f"Model 1 (Dense 128): {acc1 * 100:.2f}%  loss {loss1:.4f}")
print(f"Model 2 (Dense 256): {acc2 * 100:.2f}%  loss {loss2:.4f}")

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(hist1.history["val_accuracy"], label="Model 1 (128)")
plt.plot(hist2.history["val_accuracy"], label="Model 2 (256)")
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(hist1.history["val_loss"], label="Model 1 (128)")
plt.plot(hist2.history["val_loss"], label="Model 2 (256)")
plt.title("Validation Loss")
plt.xlabel("Epoch")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
