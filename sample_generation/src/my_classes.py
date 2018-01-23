#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
import numpy as np
import pandas as pd

gene_cols = [ u'ABL1', u'AKT1',
       u'ALK', u'APC', u'ATM', u'BRAF', u'CDH1', u'CDKN2A', u'CSF1R',
       u'CTNNB1', u'EGFR', u'ERBB2', u'ERBB4', u'FBXW7', u'FGFR1', u'FGFR2',
       u'FGFR3', u'FLT3', u'GNA11', u'GNAQ', u'GNAS', u'HNF1A', u'HRAS',
       u'IDH1', u'JAK2', u'JAK3', u'KDR', u'KIT', u'KRAS', u'MET', u'MLH1',
       u'MPL', u'NOTCH1', u'NPM1', u'NRAS', u'PDGFRA', u'PIK3CA', u'PTEN',
       u'PTPN11', u'RB1', u'RET', u'SMAD4', u'SMARCB1', u'SMO', u'SRC',
       u'STK11', u'TP53', u'VHL']

class DataGenerator(object):
    'Generates data for Keras'
    def __init__(self, connection = None, dim_x = 9, dim_y = 10, dim_z = 48, batch_size = 32, shuffle = True):
        'Initialization'
        self.connection = connection
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.shuffle = shuffle

    def generate(self, list_IDs):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_IDs)
            # Generate batches
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                # Generate data
                X, y = self.__data_generation(list_IDs_temp)
                yield X, y
    def __get_exploration_order(self, list_IDs):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)
        return indexes

    def __query_db(self, list_IDs_temp):
        'query database for one sample, normalize'

        sql = "SELECT * FROM test_001.samples WHERE index IN" + str(tuple(list_IDs_temp))
        result = self.connection.execute(sql)
        #connection.close()
        # sample = pd.read_sql_query(sql, self.engine)
        # gene_arrays = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z))
        # for i, col in enumerate(gene_cols):
        #     gene_arrays[:,:,:,i] = sample[col].apply(np.asarray).values
        return result

    def __data_generation(self, list_IDs_temp):
        'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z))
        y = np.empty((self.batch_size), dtype = int)

        type_map = {'normal': 0, 'nonresponder': 1, 'responder': 1}
        data = self.__query_db(list_IDs_temp)
        # Generate data
        for i, sample in enumerate(data):
            # Store volume
            X[i, :, :, :] = np.dstack([sample[gene] for gene in gene_cols])/40.

            # Store class
            y[i] = type_map[sample['sample_type']]

        return X, sparsify(y)

def sparsify(y):
    'Returns labels in binary NumPy array'
    n_classes = 3# Enter number of classes
    return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                   for i in range(y.shape[0])])
