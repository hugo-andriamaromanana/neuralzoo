from typing import Any

from sklearn.base import accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def evaluate_model(model: Any, dataframe_train, dataframe_test, target_train, target_test) -> None:
    model.fit(dataframe_train, target_train)
    predictions = model.predict(dataframe_test)

    print(f"La matrice de confusion : {confusion_matrix(target_test, predictions)}")
    print(f"La moyenne : {accuracy_score(target_test, predictions)}")
    print(f"La précision : {precision_score(target_test, predictions)}")
    print(f"La sensibilité : {recall_score(target_test, predictions)}")
    print(f"La moyenne harmonique : {f1_score(target_test, predictions)}")
