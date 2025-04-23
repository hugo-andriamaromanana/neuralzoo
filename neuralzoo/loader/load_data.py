from os import path
from pickle import load as load_pickle

from numpy import array, concatenate
from numpy._typing import NDArray
from pandas import DataFrame

def unpickle(file: str) -> dict:
    with open(file, 'rb') as fo:
        dict = load_pickle(fo, encoding='bytes')
    return dict

def load_cifar_single_batch(file: str) -> tuple[NDArray, NDArray]:
    datadict = unpickle(file)
    dataframe = datadict[b'data']
    target = datadict[b'labels']
    dataframe = dataframe.reshape(10000,3072)
    target = array(target)
    return dataframe, target

def load_all_cifar() -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """ load all of cifar """
    all_dataframes = []
    all_target = []
    for b in range(1,6):
        f = path.join('../data/', 'data_batch_%d' % (b, ))
        dataframe, target = load_cifar_single_batch(f)
        all_dataframes.append(dataframe)
        all_target.append(target)
    dataframes = concatenate(all_dataframes)
    targets = concatenate(all_target)
    dataframe_test, target_test = load_cifar_single_batch(path.join('../data/', 'test_batch'))
    return dataframes, targets, dataframe_test, target_test

def dataframes_from_cifar(num_training=49000, num_validation=1000, num_test=10000) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    dataframe_train, target_train, dataframe_test, target_test = load_all_cifar()

    mask = range(num_training, num_training + num_validation)
    mask = range(num_training)
    dataframe_train = dataframe_train[mask]
    target_train = target_train[mask]
    mask = range(num_test)
    dataframe_test = dataframe_test[mask]
    target_test = target_test[mask]

    dataframe_train = dataframe_train.astype('float32')
    dataframe_test = dataframe_test.astype('float32')

    dataframe_train /= 255
    dataframe_test /= 255

    dataframe_test = DataFrame(dataframe_test)
    dataframe_train = DataFrame(dataframe_train)
    target_test = DataFrame(target_test)
    target_train = DataFrame(target_train)

    return dataframe_train, target_train, dataframe_test, target_test