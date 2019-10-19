from utils import logi
import pandas as pd
import os
import matplotlib.pyplot as plt


class DataParser(object):

    # Attributes
    CSV_FILE      = ''
    CSV_HEADER    = ''
    NAME          = ''
    DESCRIPTION   = ''
    DATA_DF       = None
    DATA_NP       = None # numpy array of shape (nx, m)
    DATA_ROWS     = 0
    DATA_COLUMNS  = 0
    DATA_NX       = 0
    DATA_M        = 0

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
        self.updateData(df_shuffle)

    def updateData(self, df):
        self.DATA_DF                      = df
        self.DATA_NP                      = df.to_numpy().T
        self.DATA_ROWS, self.DATA_COLUMNS = df.shape
        self.DATA_M   , self.DATA_NX      = df.shape




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

