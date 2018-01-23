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

def get_partitions(engine):
    df = pd.read_sql_table("samples", engine, schema = "test_001", columns= ['signal_purity','sample_type', 'index'])
    df = df[df['signal_purity']==1]
    type_map = {'normal': 0, 'nonresponder': 1, 'responder': 2}
    labels = df['sample_type'].map(type_map).tolist()
    train_size = 8000#int(len(df)*.75)
    train = set(np.random.choice(df['index'].values, size=train_size, replace = False))
    test = set(df['index'].values)-set(train)
    test = np.random.choice(list(test), 2000)
    partition = {'train': list(train), 'test': list(test)}
    return partition#, labels


#Datasets
engine = init_connection(aws = False)
partition = get_partitions(engine)
connection = engine.connect()
#Parameters
params = {'connection': connection,
          'dim_x': 9,
          'dim_y': 10,
          'dim_z': 48,
          'batch_size': 1000,
          'shuffle': True}


# Generators
training_generator = DataGenerator(**params).generate(partition['train'])
validation_generator = DataGenerator(**params).generate(partition['test'])

n_classes = 3
# Design model
model = Sequential()
model.add(ZeroPadding2D(padding = (4,4), data_format = 'channels_last',input_shape=(9,10,48)))
model.add(Convolution2D(64, (2,2), activation='relu', data_format='channels_last', input_shape=(9,10,48)))
model.add(Convolution2D(32, (3,3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


# Train model on dataset
model.fit_generator(generator = training_generator,
                    steps_per_epoch = len(partition['train'])//params['batch_size'],
                    validation_data = validation_generator,
                    validation_steps = len(partition['test'])//params['batch_size'])


df = pd.read_sql_table("samples", engine, schema = "test_001", columns= [u'index', u'sample_type', u'signal_purity', u'touching_tumor_size',u'tumor_percent', u'tumor_size', u'tumor_type'])
df['tumor_size'] = df['tumor_size'].astype(float)
df['touching_tumor_size'] = df['touching_tumor_size'].astype(float)
df['tumor_percent']= df['tumor_percent'].astype(float)
df[df['signal_purity']==1]


normal = pd.read_sql("SELECT * FROM test_001.samples WHERE sample_type = 'normal' limit 1", engine)
responder = pd.read_sql("SELECT * FROM test_001.samples WHERE sample_type = 'responder' limit 1", engine)
nonresponder = pd.read_sql("SELECT * FROM test_001.samples WHERE sample_type = 'nonresponder' limit 1", engine)

for gene in ranges['normal']['gene'].unique():
    fig, ax = plt.subplots()
    ax.imshow(np.array(normal[gene].iloc[0]))
    ax.set_title('normal '+gene)
    fig.show()

changes = ranges['changes']
nr_changes = changes[changes['sample_type']=='nonresponder']
for gene in nr_changes['gene'].unique():
    fig, ax = plt.subplots()
    ax.imshow(np.array(nonresponder[gene].iloc[0]))
    ax.set_title('nonresponder '+gene)
    fig.show()

r_changes = changes[changes['sample_type']=='responder']
for gene in r_changes['gene'].unique():
    fig, ax = plt.subplots()
    ax.imshow(np.array(responder[gene].iloc[0]))
    ax.set_title("responder "  +gene)
    fig.show()
