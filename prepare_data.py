import pandas as pd
import numpy as np
from sklearn import preprocessing


# считываем данные
def prepare_data():
    dataframe = pd.read_csv('./data/data.csv')
    type_set = set(dataframe.Class)
    class_mapping = {name: index for index, name in enumerate(sorted(list(type_set)))}
    num_classes = len(type_set)
    dataframe = dataframe.replace({'Class': class_mapping})
    np_dataset = np.array(dataframe)
    normalized_features = preprocessing.normalize(np_dataset[:, :-1])
    labels = np_dataset[:, -1]
    return np_dataset, normalized_features, labels, num_classes
