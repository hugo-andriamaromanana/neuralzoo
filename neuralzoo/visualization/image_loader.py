from collections import Counter
from numpy import sqrt
from matplotlib.pyplot import bar, figure, grid, subplot, imshow, show, axis, tight_layout, title, xlabel, xticks, ylabel
from pandas import DataFrame

def load_image(images: list, size: int = 10) -> None:
    images = images[:size] 
    side = int(sqrt(len(images))) + 1

    figure(figsize=(side * 2, side * 2))
    for index, image in enumerate(images):
        subplot(side, side, index + 1)
        print(f"Image {index} type: {type(image)}, shape: {getattr(image, 'shape', 'no shape')}")

        imshow(image, interpolation='nearest')
        axis('off')
    show()

def dataframe_per_labels(dataframe: DataFrame) -> None:
    counter = Counter(dataframe['label'])
    sorted_labels = sorted(counter.keys())
    counts = [counter[label] for label in sorted_labels]

    figure(figsize=(8, 5))
    bar(sorted_labels, counts, color='skyblue')
    title('Répartion par label')
    xlabel("Label")
    ylabel("Nombre d'observations")
    xticks(sorted_labels)
    grid(axis='y', linestyle='--', alpha=0.7)
    tight_layout()
    show()