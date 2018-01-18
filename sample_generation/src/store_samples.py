import pandas as pd
from tile_generator import TileSample, load_ranges
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, ForeignKey
from sqlalchemy.orm import sessionmaker
import json
from sqlalchemy.dialects import postgresql
import time

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

def make_sample_metadata(sample):
    sample_metadata = pd.Series({'sample_type': sample.sample_type,
        'signal_purity': make_numeric(sample.sample_info['signal_purity']),
        'tumor_percent': make_numeric(sample.sample_info['tumor_percent']),
        'tumor_size': make_numeric(sample.sample_info['tumor_size']),
        'tumor_type': make_str(sample.sample_info['tumor_type']),
        'touching_tumor_size': make_numeric(sample.sample_info['touching_tumor_size'])})
    return sample_metadata

def store_samples(df,dtypes, schema = None, if_exists = 'append'):
    df.to_sql('samples',
            engine,
            if_exists = if_exists,
            schema = schema,
            index = False,
            dtype = dtypes)

def get_formatted_sample_data():
    t = time.time()
    sample = TileSample()
    t1 = time.time()
    #print 1, t1-t
    sample.generate_sample(ranges)
    t2 = time.time()
    #print 2, t2-t1
    sample._convert_to_list()
    t3 = time.time()
    #print 3, t3-t2
    sample_metadata = make_sample_metadata(sample)
    t4 = time.time()
    #print 4, t4-t3
    sample_data = sample_metadata.append(sample.gene_arrays)
    t5 = time.time()
    #print 5, t5-t4
    sample_data = sample_data.append(pd.Series({'tumor_region': sample.tumor_region}))
    t6 = time.time()
    #print 6, t6-t5
    return sample_data

def init_data():
    ranges = load_ranges()
    dtypes = {gene: postgresql.ARRAY(postgresql.DOUBLE_PRECISION) for gene in ranges['normal']['gene'].values}
    dtypes['tumor_region'] = postgresql.ARRAY(postgresql.DOUBLE_PRECISION)
    return ranges, dtypes

def init_connection(schema=None):
    with open('config/config.json') as f:
        conf = json.load(f)

    conn_str = "postgresql://{}:{}@{}/{}".format(conf['user'],conf['passw'],conf['host'], conf['database'])
    engine = create_engine(conn_str)
    if schema is not None:
        try:
            engine.execute("CREATE SCHEMA "+schema)
        except:
            pass
    return engine

def get_samples(num_samples, t):
    for i in range(0, num_samples):
        if i%1000==0:
            print str(i)+"/"+str(num_samples)
            print str(time.time()-t) + "s elapsed"
        if i == 0:
            s = get_formatted_sample_data()
            df = pd.DataFrame(index = np.arange(0,num_samples), columns = s.keys())
            df.iloc[i] = s
        else:
            df.iloc[i] = get_formatted_sample_data()
    return df

def reset(engine):
    engine.execute("drop owned by owlieee")

if __name__ == '__main__':
    schema = 'test_1.0'
    num_samples = 1000
    print "storing " + str(num_samples) + " samples to " + schema + ".samples"

    print "initializing connection..."
    engine = init_connection(schema = schema)
    ranges, dtypes = init_data()

    t = time.time()
    print "generating samples..."
    df = get_samples(num_samples, t)

    print "storing samples..."
    store_samples(df, dtypes, schema = schema, if_exists = 'append')

    engine.dispose()
    print "Done! connection closed"
