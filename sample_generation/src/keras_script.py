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

def init_connection(aws = True):
    if aws == False:
        with open('config/config.json') as f:
            conf = simplejson.load(f)
    else:
        conf = {}
        conf['user'] = input('Enter user: ')
        conf['passw'] = input('Enter passw: ')
        conf['host'] = input('Enter host: ')
        conf['database'] = input('Enter database: ')
    conn_str = "postgresql://{}:{}@{}/{}".format(conf['user'],conf['passw'],conf['host'], conf['database'])
    engine = create_engine(conn_str)
    return engine

def init_psycopg2(aws = True):
    if aws == False:
        with open('config/config.json') as f:
            conf = json.load(f)
    else:
        conf = {}
        conf['user'] = input('Enter user: ')
        conf['passw'] = input('Enter passw: ')
        conf['host'] = input('Enter host: ')
        conf['database'] = input('Enter database: ')
        conf['port'] = input('Enter port: ')
    conn_str = "host ='{}' dbname='{}' user='{}' password='{}' port='{}'".format(conf['host'], conf['database'],conf['user'],conf['passw'],conf['port'])
	# print the connection string we will use to connect
	# get a connection, if a connect cannot be made an exception will be raised here
    conn = psycopg2.connect(conn_str)
    cursor = conn.cursor()
    return conn, cursor

def get_partitions(engine, train_size = 16000, test_size = 4000, subset = ['nonresponder', 'responder', 'normal'], min_purity = None):
    df = pd.read_sql_table("samples", engine, schema = "test_001", columns= ['signal_purity','sample_type', 'index'])
    df = df[df['index']>20000]
    df = df[df['sample_type'].apply(lambda x: True if x in  subset else False)]
    type_map = {sample_type:ix for ix, sample_type in enumerate(subset)}
    df['sample_type'] = df['sample_type'].map(type_map)
    labels = dict(df[['index', 'sample_type']].values)
    train = set(np.random.choice(df['index'].values, size=train_size, replace = False))
    test = set(df['index'].values)-set(train)
    test = np.random.choice(list(test), test_size)
    partition = {'train': list(train), 'test': list(test)}
    return partition, labels

def test_model(n_classes = 2, input_shape = (9,10,48)):
    model = Sequential()
    model.add(ZeroPadding2D(padding = (1,1), data_format = 'channels_last',input_shape=input_shape))
    model.add(Convolution2D(256, (3,3), activation='relu', data_format='channels_last', input_shape=input_shape))
    model.add(Convolution2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model

#
# #Datasets
# engine = init_connection(aws = False)
# connection = engine.connect()
# #Parameters
# params = {'connection': connection,
#           'dim_x': 9,
#           'dim_y': 10,
#           'dim_z': 48,
#           'batch_size': 1000,
#           'shuffle': True}
# subsets = {'nr_r': ['nonresponder', 'responder'],
#            'nr_n':['nonresponder', 'normal'],
#            'r_n': ['responder', 'normal']}
# models = {}
# for k, subset in subsets.items():
#     print k
#     partition, labels = get_partitions(engine, subset = subset)
#     # Generators
#     training_generator = DataGenerator(**params).generate(labels, partition['train'])
#     validation_generator = DataGenerator(**params).generate(labels, partition['test'])
#     model = test_model(n_classes = 2)
#     # Train model on dataset
#     model.fit_generator(generator = training_generator,
#                         steps_per_epoch = len(partition['train'])//params['batch_size'],
#                         validation_data = validation_generator,
#                         validation_steps = len(partition['test'])//params['batch_size'])
#     models[k] = model
