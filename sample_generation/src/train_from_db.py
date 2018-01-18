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

def init_connection():
    with open('config/config.json') as f:
        conf = json.load(f)

    conn_str = "postgresql://{}:{}@{}/{}".format(conf['user'],conf['passw'],conf['host'], conf['database'])
    engine = create_engine(conn_str)
    return engine

engine = init_connection()
