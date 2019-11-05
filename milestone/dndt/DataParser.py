from utils import logi
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


class DataParser(object):
    '''
    DataParser objects combine the signal and background CsvParser objects, add a label and shuffle the data
    '''
    # Attributes
    NAME          = ''
    DESCRIPTION   = ''
    DATA_DF       = None
    DATA_NP       = None # numpy array of shape (nx, m)
    DATA_ROWS     = 0
    DATA_COLUMNS  = 0
    DATA_NX       = 0
    DATA_M        = 0
    DATA_HEADER   = ''
    X             = []
    Y             = []

    def __init__(self, signal, background, name='', descr='') -> None:
        '''
        Merge signal and background dataframe and add a label column
        :param signal    : CsvParser Object
        :param background: CsvParser Object
        '''
        df_signal              = signal.DATA_DF
        df_background          = background.DATA_DF
        pd.options.mode.chained_assignment = None  # default='warn'
        df_signal    ['label'] = 1
        df_background['label'] = 0
        pd.options.mode.chained_assignment = 'warn'  # default='warn'
        df_merge               = pd.concat([df_background, df_signal])
        df_shuffle             = df_merge.sample(frac=1, random_state=1)
        self.NAME = name
        self.DESCRIPTION = descr
        self.updateData(df_shuffle)

    def updateData(self, df):
        self.DATA_DF                      = df
        self.DATA_NP                      = df.to_numpy().T
        self.DATA_ROWS, self.DATA_COLUMNS = df.shape
        self.DATA_M   , self.DATA_NX      = df.shape
        self.X, self.Y = self.getXY(df)
        self.DATA_HEADER                  = self.getHeader(df)

    def getXY(self, df):
        X = df.drop(columns=['label']).to_numpy().T
        Y = np.reshape(df['label'].to_numpy().T, (1, -1))
        return X, Y

    def load_dataset(self, test_size=0.33):
        '''
        This should be a twin method of the load_dataset() method in the coursera class: Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization
Week 2 - TensorFlow notebook
        :param test_size    : test size % value is between (0, 1)
        :return: X, Y train and test sets (split is specified by the input variable test_size)
        '''
        X_train, X_test, Y_train, Y_test = train_test_split(self.X.T, self.Y.T, test_size = test_size, random_state = 1)
        classes = self.DATA_DF.label.unique()
        return X_train.T, Y_train.T, X_test.T, Y_test.T, classes

    def assertShapes(self):
        assert(self.X.shape[1]  == self.Y.shape[1])
        
    def getHeader(self, df):
        return list(df.columns)





# TODO
# def load_dataset():
#     train_dataset = h5py.File('datasets/train_signs.h5', "r")
#     train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
#     train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels
#
#     test_dataset = h5py.File('datasets/test_signs.h5', "r")
#     test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
#     test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels
#
#     classes = np.array(test_dataset["list_classes"][:])  # the list of classes
#
#     train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
#     test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
#
#     return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def main():
    logi('Executing main')
    cwd = os.getcwd()
    # raw = DataParser(os.path.join(cwd, 'data', 'pg.csv'))
    background = DataParser(os.path.join(cwd, 'datasets', 'tritrig-wab-beam_100MeV_L1L1_loose.csv'))
    signal     = DataParser(os.path.join(cwd, 'datasets', 'ap_100MeV_L1L1_loose.csv'))

    print(background.getDF())

    background.to_numpy()
    signal.to_numpy()


    # print(raw.to_numpy())

    # fig, ax = plt.figure(figsize=(8,6))
    # ax.plot(raw.to_numpy())
    print(background.CSV_HEADER)

    _ = plt.hist(background.RAW_DF['vz'], bins=50)
    _ = plt.hist(signal.RAW_DF['vz'], bins=50)

    plt.show()




if __name__== "__main__":
    main()

