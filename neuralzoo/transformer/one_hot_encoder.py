from pandas import DataFrame, concat

from sklearn.preprocessing import OneHotEncoder


def one_encode_labels(dataframe: DataFrame, label_column: bool = False) -> DataFrame:
    if label_column:
        dataframe["label"] = dataframe["label"].map(
            arg={
                2: "is_frog",
                3: "is_deer",
                4: "is_bird",
                5: "is_horse",
                6: "is_cat",
                7: "is_dog",
            }
        )
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(dataframe[["label"]])

    labels_dataframe = DataFrame(encoded_data, columns=encoder.get_feature_names_out())
    dataframe = dataframe.drop(columns=["label"], axis=1)

    return concat([dataframe, labels_dataframe], axis=1)
