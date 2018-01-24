import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.optimizers import SGD
from my_classes import DataGenerator
import simplejson
from sqlalchemy import create_engine
import psycopg2
import keras_script
import tiling_helper
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from collections import defaultdict

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

def format_gene_list(gene_list):
    return ", ".join(['"{}"'.format(g) for g in gene_list])
def get_sample_set(gene_list):
    engine = keras_script.init_connection(aws = False)
    connection = engine.connect()
    sql = """SELECT sample_type, "CDKN2A", "SMARCB1", "PIK3CA", "IDH1", "EGFR", "NOTCH1", "SMO", "JAK3", "GNA11", "ABL1", "APC", "HRAS", "KRAS" FROM test_001.samples
            WHERE index > 20000 AND
                signal_purity < 1
            LIMIT 20000;
             """
    query_result = connection.execute(sql)
    n_samples= 20000
    X = np.empty((n_samples, 9, 10, 13))
    y = np.empty((n_samples))
    type_map = {'normal': 0, 'nonresponder': 1, 'responder': 2}
    for i, sample in enumerate(query_result):
        if i%1000 ==0:
            print i
        g = [np.array(sample[gene]) for gene in keep]
        X[i] = np.dstack(g[0:13])[:, :, :]
        y[i] = type_map[sample['sample_type']]
    return X, y

def get_train_test(X, y, exclude = None):
    if exclude is None:
        X_sub = X
        y_sub = y
    else:
        X_sub = X[np.where(y!=exclude)]
        y_sub = y[np.where(y!=exclude)]
    n_classes = len(np.unique(y_sub))
    if exclude == 0:
        y_sub = y_sub-1
    elif exclude == 1:
        y_sub = y_sub/2
    X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.25, random_state=42)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 40
    X_test /= 40
    Y_train = np_utils.to_categorical(y_train, n_classes)
    Y_test = np_utils.to_categorical(y_test, n_classes)
    return X_train, X_test, Y_train, Y_test

def check_binary(X, y):
    subsets = {'nr_r': 0,
               'nr_n': 2,
               'r_n': 1}
    models = defaultdict(dict)
    for k, exclude in subsets.items():
        X_train, X_test, y_train, y_test = get_train_test(X, y, exclude = exclude)
        model = keras_script.test_model(input_shape = (9,10,13))
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

def check_3_classes(X, y):
    X_train, X_test, y_train, y_test = get_train_test(X, y)#, exclude = None)
    model = keras_script.test_model(n_classes = 3, input_shape = (9,10,13))
    model.fit(X_train, y_train, batch_size=32, nb_epoch=5, verbose=1)
    model = defaultdict(dict)
    loss, accuracy = model.evaluate(X_test,y_test)
    model['model'] = model
    model['X_train'] = X_train
    model['y_train'] = y_train
    model['X_test'] = X_test
    model['y_test'] = y_test
    model['loss'] = loss
    model['accuracy'] = accuracy
    return model

X, y = get_sample_set()



for k, v in models.items():
    v['model'].evaluate(v['X_test'], v['y_test'])
