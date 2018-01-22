import pandas as pd
from tile_generator import TileSample, load_ranges
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, ForeignKey
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateSchema
from io import StringIO
import json
import time

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
            index = True,
            index_label = 'index',
            dtype = dtypes)

def get_formatted_sample_data():
    sample = TileSample()
    sample.generate_sample(ranges)
    sample._convert_to_list()
    sample_metadata = make_sample_metadata(sample)
    sample_data = sample_metadata.append(sample.gene_arrays)
    sample_data = sample_data.append(pd.Series({'tumor_region': sample.tumor_region}))
    return sample_data

def init_data():
    ranges = load_ranges()
    dtypes = {gene: postgresql.ARRAY(postgresql.REAL) for gene in ranges['normal']['gene'].values}
    dtypes['tumor_region'] = postgresql.ARRAY(postgresql.REAL)
    return ranges, dtypes

def init_connection(schema=None, aws = True):
    if aws == False:
        with open('config/config.json') as f:
            conf = json.load(f)
    else:
        conf = {}
        conf['user'] = input('Enter user: ')
        conf['passw'] = input('Enter passw: ')
        conf['host'] = input('Enter host: ')
        conf['database'] = input('Enter database: ')
    conn_str = "postgresql://{}:{}@{}/{}".format(conf['user'],conf['passw'],conf['host'], conf['database'])
    engine = create_engine(conn_str)
    if schema is not None:
        try:
            engine.execute(CreateSchema(schema))
        except:
            pass
    return engine

def get_samples(num_samples, t, min_ind):
    for i in range(min_ind, num_samples+min_ind):
        # if i%1000==0:
        #     print str(i)+"/"+str(num_samples)
            # print str(time.time()-t) + "s elapsed"
        if i == min_ind:
            s = get_formatted_sample_data()
            df = pd.DataFrame(index = np.arange(min_ind,num_samples+min_ind), columns = s.keys())
            df.loc[i] = s
        else:
            df.loc[i] = get_formatted_sample_data()
    return df

def reset(engine):
    engine.execute("drop owned by owlieee")
    engine.dispose()

if __name__ == '__main__':

    schema = 'test_001'
    total_samples = 100000
    print("storing " + str(total_samples) + " samples to " + schema + ".samples")

    print("initializing connection...")
    engine = init_connection(schema = schema)#, aws = False)
    ranges, dtypes = init_data()
    completed_samples = 0
    if engine.has_table('samples', schema = schema):
        df = pd.read_sql_table("samples", engine, schema = schema, columns=['sample_type', 'index'])
        min_ind = df['index'].max() + 1
    else:
        min_ind = 0
    print("min_ind = " + str(min_ind))
    while completed_samples < total_samples:
        batch = 1000

        print("generating samples...")
        t = time.time()
        df = get_samples(batch, t, min_ind)
        print("generated " + str(batch) + ' t = ' + str(time.time() - t))

        print("storing samples...")
        t = time.time()
        store_samples(df,dtypes, schema = schema)
        print("stored " + str(batch) + ' t = ' + str(time.time() - t))

        completed_samples += batch
        min_ind += batch
        print(str(total_samples - completed_samples) + ' remain')
    engine.dispose()
    print("Done! connection closed")


# def to_sql(engine, df, dtypes, schema, table = 'samples', if_exists='fail', sep='\t', encoding='utf8'):
#     # Create table
#     table_name = schema + '.' + table
#     if engine.has_table(table, schema)==False:
#         df[:1].to_sql(table,
#                     engine,
#                     schema = schema,
#                     if_exists=if_exists,
#                     index = True,
#                     dtype = dtypes)
#
#     # Prepare data
#     output = StringIO()
#     df.to_csv(output, sep=sep, header=False)
#     output.seek(0)
#
#
#     # Insert data
#     connection = engine.raw_connection()
#     cursor = connection.cursor()
#     cursor.copy_from(output,"test_001.samples", sep=sep)
#     connection.commit()
#     cursor.close()
#     del cursor
#     connection.close()
