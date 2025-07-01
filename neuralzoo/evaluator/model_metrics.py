from typing import Any

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


def evaluate_model(
    model: Any, dataframe_train, dataframe_test, target_train, target_test
) -> None:
    model.fit(dataframe_train, target_train)
    predictions = model.predict(dataframe_test)

    print(f"La matrice de confusion : {confusion_matrix(target_test, predictions)}")
    print(f"La moyenne : {accuracy_score(target_test, predictions)}")
    print(f"La précision : {precision_score(target_test, predictions)}")
    print(f"La sensibilité : {recall_score(target_test, predictions)}")
    print(f"La moyenne harmonique : {f1_score(target_test, predictions)}")


# get it done, using MLP, get some kind of accuracy
# display on a graph, accuracy per epochs

# What is cipher10, odds, MLP, implementation, CNN, implementation, conclusion
