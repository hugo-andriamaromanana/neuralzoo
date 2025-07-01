from typing import Any

from numpy import reshape
from tensorflow.keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD


from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


def transform(src_data: Any) -> Any:
    (X_train, y_train), (X_test, y_test) = src_data

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    X_train = reshape(X_train, (X_train.shape[0], -1))
    X_test = reshape(X_test, (X_test.shape[0], -1))
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    X_train /= 255
    X_test /= 255
    return (X_train, y_train), (X_test, y_test)


from numpy import argmax


def train(data: Any) -> None:
    (X_train, y_train), (X_test, y_test) = data
    model = Sequential()
    model.add(Dense(256, activation="relu", input_dim=3072))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(
        X_train, y_train, epochs=10, batch_size=32, validation_split=0.2
    )

    score = model.evaluate(X_test, y_test, batch_size=128)
    print(model.metrics_names)
    print(score)

    y_pred_probs = model.predict(X_test)
    y_pred = argmax(y_pred_probs, axis=1)
    y_true = argmax(y_test, axis=1)

    print(f"Confusion matrix:\n{confusion_matrix(y_true, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, average='macro'):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, average='macro'):.4f}")
