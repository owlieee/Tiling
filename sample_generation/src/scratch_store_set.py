import pandas as pd
from tile_generator import TileSample, load_ranges
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

responder_t_changes = {'CDKN2A', 'NOTCH1', 'EGFR', 'IDH1', 'KRAS', 'SMARCB1', 'APC', 'GNA11', 'PIK3CA'}
responder_touch_t_changes = {'HRAS', 'SMO'}
responder_rand_changes = {'ABL1', 'JAK3'}
nonresponder_t_changes = {'CDKN2A', 'EGFR', 'IDH1', 'KRAS', 'SMARCB1', 'APC', 'PIK3CA'}
keep = responder_t_changes
keep.update(responder_touch_t_changes)
keep.update(responder_rand_changes)
keep.update(nonresponder_t_changes)

matplotlib.style.use('ggplot')
ranges = load_ranges()
n_samples = 10000
type_map = {'normal': 0, 'nonresponder': 1, 'responder': 2}
X = np.empty((n_samples, 9, 10, 13))
y = np.empty((n_samples))

for i in range(0,n_samples):
    if i%1000 ==0:
        print i
    sample = TileSample()
    sample.generate_sample(ranges)
    g = {k:v for k, v in sample.gene_arrays.items() if k in keep}
    X[i] = np.dstack(g.values()[0:13])[:, :, :]
    y[i] = type_map[sample.sample_type]

import cPickle as pickle
test_data = {'X': X, 'y': y}
myfile = open('../data/test_data.pickle', 'wb')
pickle.dump(test_data, myfile)
myfile.close()
