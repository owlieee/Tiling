import pandas as pd
import numpy as np
import collections
from collections import defaultdict
from get_tiling_ranges import store_ranges
import os
import pdb
import nose.tools as n
import tile_generator as tg

ranges = tg.load_ranges()


def test_generate_tumor():
    sample = tg.TileSample()
    for sample_type in ['normal', 'nonresponder', 'responder']:
        sample.generate_sample(ranges, sample_type=sample_type)
        generated = sample.tumor_region is None
        if sample_type == 'normal':
            generated_actual = True
        else:
            generated_actual = False
            set_size = sample.sample_info['tumor_size']
            actual_size =  np.sum(sample.tumor_region>0)
            message_size = sample_type + ' sample, tumor size should be %d: Got %d' \
                      % (set_size, actual_size)
            n.ok_((n.assert_equal(set_size, actual_size), message_size))
        message1 = sample_type + ' sample, generate tumor should be ' + str(generated_actual)
        n.assert_equal(generated, generated_actual, message1)


def test_touching_tumor():
    sample = tg.TileSample()
    sample.generate_sample(ranges, sample_type='responder')
    sample.tumor_region = np.array( [[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                               [ 0.,  0.,  0.,  1.,  2.,  3.,  0.,  0.,  0.,  0.],
                               [ 0.,  0.,  0.,  1.,  1.,  5.,  1.,  0.,  0.,  0.],
                               [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
                               [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.],
                               [ 0.,  0.,  0.,  0.,  2.,  4.,  2.,  0.,  0.,  0.],
                               [ 0.,  0.,  0.,  0.,  2.,  2.,  2.,  0.,  0.,  0.],
                               [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                               [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
    actual = sample._get_touching_tumor()
    correct = np.array(  [[ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.],
                       [ 0.,  0.,  1.,  0.,  0.,  0.,  1.,  1.,  0.,  0.],
                       [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
                       [ 0.,  0.,  1.,  1.,  1.,  0.,  1.,  1.,  0.,  0.],
                       [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.],
                       [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.],
                       [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.],
                       [ 0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
                       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
    n.assert_true((actual ==correct).all())
