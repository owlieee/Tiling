import pandas as pd
from tile_generator import TileSample, load_ranges
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, ForeignKey
from sqlalchemy.orm import sessionmaker
import json

with open('config/config.json') as f:
    conf = json.load(f)

conn_str = "postgresql://{}:{}@{}/{}".format(conf['user'],conf['passw'],conf['host'], conf['database'])
engine = create_engine(conn_str)

Session = sessionmaker(bind=engine)
session = Session()
metadata = MetaData()
sample_info = Table('sample_info', metadata,
        Column('sample_id', Integer, primary_key=True),
        Column('sample_type', String(50)))
metadata.create_all(engine)


ranges = load_ranges()
sample = TileSample()
sample.generate_sample(ranges, sample_type = 'responder')
sample_id = make_sample_table(sample, engine))

def make_numeric(value):
    if value==[]:
        return 0
    else:
        return value

def make_str(value):
    if value==[]:
        return ''
    else:
        x_dist =value['x_dist'].__name__
        y_dist = value['y_dist'].__name__
        return 'X_'+x_dist + '_Y_' + y_dist

def make_sample_table(sample, engine, if_exists = 'append'):
    df = pd.DataFrame({'sample_type': [sample.sample_type]})
    value = sample.sample_type
    result = session.execute('INSERT INTO sample_info (sample_type) VALUES (value)')
    entry = sample_info.to_sql('sample_info',
                     engine,
                     if_exists=if_exists,
                     index = False)
    max_id = pd.read_sql("SELECT MAX(sample_id) from sample_info", engine)

def make_sample_info_table(sample, engine, if_exists = 'append'):
    sample_info = pd.DataFrame({'sample_type': sample.sample_type,
        'signal_purity': make_numeric(sample.sample_info['signal_purity']),
        'tumor_percent': make_numeric(sample.sample_info['tumor_percent']),
        'tumor_size': make_numeric(sample.sample_info['tumor_size']),
        'tumor_type': make_str(sample.sample_info['tumor_type']),
        'touching_tumor_size': make_numeric(sample.sample_info['touching_tumor_size'])},
        index = [0])
    =sample_info.to_sql('sample_info',
                     engine,
                     if_exists=if_exists,
                     index = True,
                     index_label='sample_id')
