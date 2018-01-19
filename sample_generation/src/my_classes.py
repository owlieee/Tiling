import pandas as pd
from tile_generator import TileSample, load_ranges
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, ForeignKey
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateSchema
import cStringIO
import json
import time
from sklearn.model_selection import train_test_split

def init_connection():
    with open('config/config.json') as f:
        conf = json.load(f)

    conn_str = "postgresql://{}:{}@{}/{}".format(conf['user'],conf['passw'],conf['host'], conf['database'])
    engine = create_engine(conn_str)
    return engine

engine = init_connection()

df = pd.read_sql_table("samples", engine, schema = "test_001", columns= ['sample_type'])
ids = df.index.values
train_size = int(len(ids)*.75)
train = set(np.random.choice(ids, size=train_size, replace = False))
test = set(ids)-set(train)
partitions = {'train': list(train), 'test': list(test)}


def __init__(self, dim_x = 9, dim_y = 10, dim_z = 48, batch_size = 32, shuffle = True):
    'Initialization'
    self.dim_x = dim_x
    self.dim_y = dim_y
    self.dim_z = dim_z
    self.batch_size = batch_size
    self.shuffle = shuffle


def __get_exploration_order(self, list_IDs):
    'Generates order of exploration'
    # Find exploration order
    indexes = np.arange(len(list_IDs))
    if self.shuffle == True:
      np.random.shuffle(indexes)
    return indexes


def __data_generation(self, labels, list_IDs_temp):
  'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
  # Initialization
  X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z, 1))
  y = np.empty((self.batch_size), dtype = int)

  # Generate data
  for i, ID in enumerate(list_IDs_temp):
      # Store volume
      X[i, :, :, :, 0] = np.load(ID + '.npy')

      # Store class
      y[i] = labels[ID]

  return X, sparsify(y)
