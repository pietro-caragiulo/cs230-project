from utils import logi
import pandas as pd
import os
import matplotlib.pyplot as plt


class DataParser(object):

    # Attributes
    CSV_FILE = ''
    CSV_HEADER = ''
    NX = 0
    M = 0
    RAW_DF   = None
    RAW_NP   = None

    def __init__(self, csvFile) -> None:
        '''

        :param csvFile: CSV File location
        '''
        df      = pd.read_csv(csvFile)
        rows    = df.shape[0]
        columns = df.shape[1]
        logi('Reading CSV file %s found %d rows and %d columns' % (csvFile, rows, columns))
        self.NX = columns
        self.M  = rows
        logi('Found %d features and %d samples' % (self.NX, self.M))
        self.CSV_FILE   = csvFile
        self.RAW_DF = df
        self.CSV_HEADER = self.getHeader(df)

    def getHeader(self, df):
        return list(df.columns)


    def getDF(self):
        '''
        Get the dataframe from csvFile
        :return:
        '''
        return self.RAW_DF

    def to_numpy(self):
        '''
        Convert the Dataframe into a numpy array
        :return: numpy array of shape (nx, m)
        '''
        self.RAW_NP = self.RAW_DF.to_numpy()
        return self.RAW_NP

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
    background = DataParser(os.path.join(cwd, 'data', 'tritrig-wab-beam_100MeV_L1L1_loose.csv'))
    signal     = DataParser(os.path.join(cwd, 'data', 'ap_100MeV_L1L1_loose.csv'))

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

