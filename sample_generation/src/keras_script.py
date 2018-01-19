import pandas as pd
import numpy as np
from keras.models import Sequential
from my_classes import DataGenerator

def init_connection():
    with open('config/config.json') as f:
        conf = json.load(f)

    conn_str = "postgresql://{}:{}@{}/{}".format(conf['user'],conf['passw'],conf['host'], conf['database'])
    engine = create_engine(conn_str)
    return engine

def get_partitions(engine):
    df = pd.read_sql_table("samples", engine, schema = "test_001", columns= ['sample_type', 'index'])
    type_map = {'normal': 0, 'nonresponder': 1, 'responder': 2}
    labels = df['sample_type'].map(type_map)
    train_size = int(len(df)*.75)
    train = set(np.random.choice(df['index'].values, size=train_size, replace = False))
    test = set(df['index'].values)-set(train)
    partitions = {'train': list(train), 'test': list(test)}

engine = init_connection()
