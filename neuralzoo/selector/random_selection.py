from random import shuffle

from pandas import DataFrame

def get_random_image_per_class(dataframe: DataFrame) -> list:
    values = []
    potential_values = dataframe['label'].unique()  
    for value in potential_values:
        images_in_class = [dataframe.loc[index, 'image'] for index in range(len(dataframe)) if dataframe.loc[index, 'label'] == value]
        if images_in_class:
            shuffle(images_in_class)
            values.append(images_in_class[0])  
    
    return values