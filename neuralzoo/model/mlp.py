from typing import Any
from pandas import DataFrame
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from neuralzoo.transformer.one_hot_encoder import one_encode_labels
from numpy import ndarray, stack
from neuralzoo.evaluator.model_metrics import evaluate_model


def get_clean_values_targets(dataframe: DataFrame) -> tuple[ndarray, DataFrame]:
    values, target = dataframe["image"], dataframe.drop(columns=["image"],axis=1)
    values, target = dataframe["image"], dataframe.drop(columns=["image"],axis=1)
    values_array = stack(values.values)
    values_flat = values_array.reshape(values_array.shape[0], -1)
    return values_flat, target

def transform(src_data: Any) -> tuple[DataFrame, DataFrame]:
    (dataframe_train, target_train), (dataframe_test, target_test) = src_data
    full_dataframe_train = DataFrame(
        data={
            "image": [element for element in dataframe_train],
            "label": [element[0] for element in target_train],
        }
    )
    full_dataframe_test = DataFrame(
        data={
            "image": [element for element in dataframe_test],
            "label": [element[0] for element in target_test],
        }
    )

    UNWANTED_LABELS = [0, 1, 8, 9]
    full_dataframe_train = full_dataframe_train[
        ~full_dataframe_train["label"].isin(UNWANTED_LABELS)
    ]
    full_dataframe_train.reset_index(drop=True, inplace=True)

    full_dataframe_train = one_encode_labels(full_dataframe_train, True)

    return full_dataframe_train, full_dataframe_test

def train(train_data: DataFrame, test_data: DataFrame) -> None:
    values_train, target_train = get_clean_values_targets(train_data)
    values_test, target_test = get_clean_values_targets(test_data)
    evaluate_model(MultiOutputClassifier(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)), values_train, values_test, target_train, target_test)