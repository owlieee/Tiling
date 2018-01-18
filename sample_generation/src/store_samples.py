import pandas as pd
from tile_generator import TileSample, load_ranges
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, ForeignKey
from sqlalchemy.orm import sessionmaker
import json
from sqlalchemy.dialects import postgresql


def make_numeric(value):
    if value==[]:
        return np.nan
    else:
        return value

def make_str(value):
    if value==[]:
        return ''
    else:
        x_dist =value['x_dist'].__name__
        y_dist = value['y_dist'].__name__
        return 'X_'+x_dist + '_Y_' + y_dist

def make_sample_info_table(sample, sample_id, schema = None, if_exists = 'append'):
    sample_df = pd.DataFrame({'sample_type': sample.sample_type,
        'signal_purity': make_numeric(sample.sample_info['signal_purity']),
        'tumor_percent': make_numeric(sample.sample_info['tumor_percent']),
        'tumor_size': make_numeric(sample.sample_info['tumor_size']),
        'tumor_type': make_str(sample.sample_info['tumor_type']),
        'touching_tumor_size': make_numeric(sample.sample_info['touching_tumor_size'])},
        index = [sample_id])
    sample_df.to_sql('sample_info',
                     engine,
                     if_exists=if_exists,
                     schema = schema,
                     index = True,
                     index_label='sample_id')

def postgres_array(numpy_array):
    return [list(r) for r in numpy_array]

def make_gene_table(sample, sample_id, schema = None, if_exists = 'append'):
    dtypes = {gene: postgresql.ARRAY(postgresql.DOUBLE_PRECISION for gene in sample.gene_arrays.columns}
    sample.gene_arrays['sample_id'] = sample_id
    sample.gene_arrays.to_sql('gene_table',
                    engine,
                    if_exists = if_exists,
                    schema = schema,
                    index = False,
                    dtype = dtypes)

def make_tumor_table(sample, sample_id, schema = None, if_exists = 'append'):
    if sample.tumor_region is not None:
        tumor_data = pd.Series({'sample_id': int(sample_id),
                                'tumor_array': sample.tumor_region})
        tumor_df = pd.DataFrame().append(tumor_data, ignore_index =  True)
        tumor_df.to_sql('tumor_region',
                        engine,
                        if_exists = if_exists,
                        schema = schema,
                        index = False,
                        dtype = {'gene_array': postgresql.ARRAY(postgresql.DOUBLE_PRECISION)})

def make_all_tables(sample, sample_id, schema = None, if_exists = 'append'):
    make_sample_info_table(sample, sample_id, schema = schema, if_exists = if_exists)
    make_gene_tables(sample, sample_id, schema = schema, if_exists = if_exists)
    make_tumor_table(sample, sample_id, schema = schema, if_exists = if_exists)

def init_connection():
    with open('config/config.json') as f:
        conf = json.load(f)

    conn_str = "postgresql://{}:{}@{}/{}".format(conf['user'],conf['passw'],conf['host'], conf['database'])
    engine = create_engine(conn_str)
    if engine.has_table('sample_info'):
        start_id = int(1 + pd.read_sql("SELECT sample_id FROM sample_info ORDER BY sample_id DESC limit 1", engine).iloc[0][0])
    else:
        start_id = int(0)
    return engine, start_id

engine, start_id = init_connection()
ranges = load_ranges()
schema = "test"
engine.execute("CREATE SCHEMA test")

num_samples = 10
for sample_id in range(start_id, num_samples):
    print sample_id
    sample = TileSample()
    sample.generate_sample(ranges)
    sample.convert_to_list()
    make_all_tables(sample, sample_id, schema = schema, if_exists = 'append')
