import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.optimizers import SGD
from my_classes import DataGenerator
#import simplejson
from sqlalchemy import create_engine
import psycopg2
import keras_script
import tiling_helper
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
#from keras.optimizers import SGD
from collections import defaultdict
from keras.models import Model
from keras.layers.merge import concatenate

all_gene_cols = [ u'ABL1', u'AKT1', u'ALK', u'APC', u'ATM', u'BRAF', u'CDH1', u'CDKN2A', u'CSF1R',
       u'CTNNB1', u'EGFR', u'ERBB2', u'ERBB4', u'FBXW7', u'FGFR1', u'FGFR2',
       u'FGFR3', u'FLT3', u'GNA11', u'GNAQ', u'GNAS', u'HNF1A', u'HRAS',
       u'IDH1', u'JAK2', u'JAK3', u'KDR', u'KIT', u'KRAS', u'MET', u'MLH1',
       u'MPL', u'NOTCH1', u'NPM1', u'NRAS', u'PDGFRA', u'PIK3CA', u'PTEN',
       u'PTPN11', u'RB1', u'RET', u'SMAD4', u'SMARCB1', u'SMO', u'SRC',
       u'STK11', u'TP53', u'VHL']

meta_cols = ['index', u'sample_type', u'signal_purity', u'touching_tumor_size',
   u'tumor_percent', u'tumor_size', u'tumor_type']

type_map = {'normal': 0, 'nonresponder': 1, 'responder': 2}

def get_interest_genes():
    responder_t_changes = {'CDKN2A', 'NOTCH1', 'EGFR', 'IDH1', 'KRAS', 'SMARCB1', 'APC', 'GNA11', 'PIK3CA'}
    responder_touch_t_changes = {'HRAS', 'SMO'}
    responder_rand_changes = {'ABL1', 'JAK3'}
    nonresponder_t_changes = {'CDKN2A', 'EGFR', 'IDH1', 'KRAS', 'SMARCB1', 'APC', 'PIK3CA'}
    keep = responder_t_changes
    keep.update(responder_touch_t_changes)
    keep.update(responder_rand_changes)
    keep.update(nonresponder_t_changes)
    return list(keep)

def format_col_list(gene_list):
    gene_list = gene_list + meta_cols
    return ", ".join(['"{}"'.format(g) for g in gene_list])

def get_sample_set(gene_string, n_samples=10000, custom_query='', aws = True):
    engine = keras_script.init_connection(aws = aws)
    connection = engine.connect()
    sql = """SELECT {} FROM test_001.samples {} LIMIT {};""".format(gene_string, custom_query, n_samples)
    query_result = connection.execute(sql)
    connection.close()
    return query_result

def format_sample_set(query_result, gene_list, n_samples):
    X = np.empty((n_samples, 9, 10, len(gene_list)))
    y = np.empty((n_samples))
    df = pd.DataFrame(index = range(0, n_samples), columns = gene_list + meta_cols)
    for i, sample in enumerate(query_result):
        if i%1000 ==0:
            print("{}/{}".format(i, n_samples))
        g = [np.array(sample[gene]) for gene in gene_list]
        m = {col:sample[col] for col in meta_cols}
        m.update(dict(zip(gene_list, [np.mean(a) for a in g])))
        df.loc[i] = m
        X[i] = np.dstack(g[0:len(gene_list)])[:, :, :]
        y[i] = type_map[sample['sample_type']]
    return df, X, y

def get_subset(X, y, exclude):
    if exclude is None:
        X_sub = X
        y_sub = y
    else:
        X_sub = X[np.where(y!=exclude)]
        y_sub = y[np.where(y!=exclude)]
    if exclude == 0:
        y_sub = y_sub-1
    elif exclude == 1:
        y_sub = y_sub/2
    n_classes = len(np.unique(y_sub))
    return X_sub, y_sub, n_classes

def get_train_test(X, y, exclude = None):
    X_sub, y_sub, n_classes = get_subset(X, y, exclude)
    X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.25, random_state=42)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 40
    X_test /= 40
    Y_train = np_utils.to_categorical(y_train, n_classes)
    Y_test = np_utils.to_categorical(y_test, n_classes)
    return X_train, X_test, Y_train, Y_test

def get_train_test_import(X, y, multichannel = False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train /= 40
    X_test /= 40
    n_classes = len(np.unique(y_train))
    if multichannel == True:
        X_train= [X_train[:,:,:,i].reshape(len(X_train),9,10,1) for i in range(0, X_train.shape[-1])]
        X_test= [X_test[:,:,:,i].reshape(len(X_test),9,10,1) for i in range(0, X_test.shape[-1])]
    Y_train = np_utils.to_categorical(y_train, n_classes)
    Y_test = np_utils.to_categorical(y_test, n_classes)
    return X_train, X_test, Y_train, Y_test

def check_binary(X, y, n_genes=48):
    subsets = {'nr_r': type_map['normal'],
               'nr_n': type_map['responder'],
               'r_n': type_map['nonresponder']}
    models = defaultdict(dict)
    for k, exclude in subsets.items():
        X_train, X_test, y_train, y_test = get_train_test(X, y, exclude = exclude)
        model = keras_script.test_model(input_shape = (9,10,n_genes))
        model.fit(X_train, y_train, batch_size=32, nb_epoch=5, verbose=1)
        loss, accuracy = model.evaluate(X_test,y_test)
        models[k]['model'] = model
        models[k]['X_train'] = X_train
        models[k]['y_train'] = y_train
        models[k]['X_test'] = X_test
        models[k]['y_test'] = y_test
        models[k]['loss'] = loss
        models[k]['accuracy'] = accuracy
    return models

def check_3_classes(X, y, n_genes=48):
    X_train, X_test, y_train, y_test = get_train_test(X, y)#, exclude = None)
    model = keras_script.test_model(n_classes = 3, input_shape = (9,10,n_genes))
    model.fit(X_train, y_train, batch_size=32, nb_epoch=10, verbose=1)
    model_dict = defaultdict(dict)
    loss, accuracy = model.evaluate(X_test,y_test)
    model_dict['model'] = model
    #model_dict['X_train'] = X_train
    #model_dict['y_train'] = y_train
    #model_dict['X_test'] = X_test
    #model_dict['y_test'] = y_test
    model_dict['loss'] = loss
    model_dict['accuracy'] = accuracy
    return model_dict

def one_gene_input():
    input_shape = (9,10,1)
    batch_shape=(None, 9,10,1)
    inputs = Input(batch_shape=batch_shape)
    zeropad = ZeroPadding2D(padding = (1,1), data_format = 'channels_last',input_shape=input_shape)(inputs)
    conv = Convolution2D(32, (3,3), activation = 'relu',data_format = 'channels_last',input_shape=input_shape )(zeropad)
    conv2 = Convolution2D(32, (3,3), activation = 'relu',data_format = 'channels_last',input_shape=input_shape )(conv)
    pool = MaxPooling2D(pool_size=(2,2))(conv2)
    drop = Dropout(0.25)(pool)
    flat = Flatten()(pool)
    return inputs, flat

def define_model(n_genes, n_classes = 3):
    flat_layers = []
    input_layers = []
    for i in range(0, n_genes):
        inputs, flat = one_gene_input()
        flat_layers.append(flat)
        input_layers.append(inputs)
    merged = concatenate(flat_layers)
	# interpretation
    dense = Dense(128, activation='relu')(merged)
    outputs = Dense(n_classes, activation='sigmoid',input_shape = (9,10,1))(dense)
    model = Model(inputs=input_layers, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# X = np.load("X.npy")
# y = np.load("y.npy")
# X_train, X_test, Y_train, Y_test = get_train_test_import(X, y, multichannel = True)
# sig = get_interest_genes()
# nosig_ix = [ix for ix, v in enumerate(all_gene_cols) if v not in sig]
# sig_ix = [ix for ix, v in enumerate(all_gene_cols) if v in sig]
# n_genes = len(X_train)
# n_classes = 3
# #X_train= [X[:,:,:,i].reshape(20000,9,10,1) for i in sig_ix[:n_genes]]
# #Y_train = np_utils.to_categorical(y, n_classes)
# model = define_model(n_genes, n_classes)
# model.fit(X_train, Y_train, epochs=5, batch_size=16)
# model.save('multichannelmodel.h5')

# gene_list = all_gene_cols
# custom_query = "WHERE index > 20000 AND signal_purity = 1"
# n_samples = 20000
# query_result = get_sample_set(format_col_list(gene_list), n_samples=n_samples, custom_query=custom_query)
# df, X, y = format_sample_set(query_result, gene_list, n_samples)
# b_48 = check_binary(X, y, n_genes=len(gene_list))
# model_48 = check_3_classes(X, y, n_genes=len(gene_list))
#
#
# gene_list = get_interest_genes()
# custom_query = "WHERE index > 20000 AND signal_purity = 1"
# n_samples = 20000
# query_result = get_sample_set(format_col_list(gene_list), n_samples=n_samples, custom_query=custom_query)
# df, X, y = format_sample_set(query_result, gene_list, n_samples)
# b_13 = check_binary(X, y, n_genes=len(gene_list))
# model_13 = check_3_classes(X, y, n_genes=len(gene_list))

if __name__=='__main__':
    print("querying")
    gene_list = all_gene_cols
    custom_query = "WHERE index > 20000"
    n_samples = 100000
    query_result = get_sample_set(format_col_list(gene_list), n_samples=n_samples, custom_query=custom_query, aws = True)
    df, X, y = format_sample_set(query_result, gene_list, n_samples)
    #df.to_pickle('data.p')
    #np.save("X_5", X)
    #np.save("y_5", y)
    if False:
        print("trying gene sets")
        sig = get_interest_genes()
        nosig_ix = [ix for ix, v in enumerate(all_gene_cols) if v not in sig]
        np.random.shuffle(nosig_ix)
        models = {}
        gene_list = sig
        for i in range(0, len(all_gene_cols)-len(sig)):
            print(i)
            gene_ix = [ix for ix, v in enumerate(all_gene_cols) if v in gene_list]
            X_new = X[:,:,:,gene_ix]
            model_ = check_3_classes(X_new, y, n_genes=len(gene_list))
            models[i] = model_
            gene_list = gene_list + [str(all_gene_cols[nosig_ix[i]])]
        print("saving results")
        results = pd.DataFrame()
        for k, v in models.items():
            temp = pd.Series({'n_extra': k,
                'accuracy': v['accuracy'],
                'loss': v['loss'],
                'params': v['model'].count_params()})
            results = results.append(temp, ignore_index = True)

        results.to_csv('results.csv')
    else:
        X_train, X_test, Y_train, Y_test = get_train_test_import(X, y, multichannel = True)
        print("trying gene sets multichannel ")
        sig = get_interest_genes()
        nosig_ix = [ix for ix, v in enumerate(all_gene_cols) if v not in sig]
        np.random.shuffle(nosig_ix)
        models = {}
        gene_list = sig
        for i in range(0, len(all_gene_cols)-len(sig)):
            print(i)
            gene_ix = [ix for ix, v in enumerate(all_gene_cols) if v in gene_list]
            new_X_train = [x for ix,  x in enumerate(X_train) if ix in gene_ix]
            new_X_test = [x for ix,  x in enumerate(X_test) if ix in gene_ix]
            n_genes = len(new_X_train)
            n_classes = 3
            #X_train= [X[:,:,:,i].reshape(20000,9,10,1) for i in sig_ix[:n_genes]]
            #Y_train = np_utils.to_categorical(y, n_classes)
            model = define_model(n_genes, n_classes)
            model.fit(new_X_train, Y_train, epochs=5, batch_size=32)
            model_dict = defaultdict(dict)
            loss, accuracy = model.evaluate(new_X_test,Y_test)
            model_dict['model'] = model
            model_dict['loss'] = loss
            model_dict['accuracy'] = accuracy
            model_dict['params'] = model.count_params()
            models[i] = model_dict
        print("saving results")
        results = pd.DataFrame()
        for k, v in models.items():
            temp = pd.Series({'n_extra': k,
                'accuracy': v['accuracy'],
                'loss': v['loss'],
                'params': v['model'].count_params()})
            results = results.append(temp, ignore_index = True)

        results.to_csv('results_mc.csv')
    # #
    # #
    # for k, v in models.items():
    #     v['model'].evaluate(v['X_test'], v['y_test'])
