import pandas as pd
import numpy as np
import collections

class TileSample:
    """A tile sample generator:
        - Vectorize the raw text into features.
        - Fit a naive bayes model to the resulting features.
    """
    def __init__(self):
        #sample_type: ['normal', 'nonresponder', 'responder']
        #tumor_region: array where tumor is (empty for normal)
        #tumor_info: metadata about tumor region, size, tumor_type (normal, random_xy, random_x, random_y)
        #gene_arrays: dictionary with distribution array for each gene
        self.sample_type = None
        self.tumor_region = None
        self.tumor_info = None
        self.gene_arrays = None

    def generate_sample(self, sample_type=None):
        if sample_type == None:
            self.sample_type = np.random.choice( ['normal', 'nonresponder', 'responder'])
        else:
            self.sample_type = sample_type
        if sample_type != 'normal':
            self.generate_tumor()

    def generate_tumor(self, p_norm = 0.5):
        tumor_info = {}
        n_samples = max(0, int(np.random.normal(loc = 180, scale = 90)))
        generator_map = {True: np.random.randn, False: np.random.rand}
        tumor_type_map = {True: {True: 'normal', False: 'random_y'}, False: {True: 'random_x', False: 'random_xy'}}
        x_type = np.random.random() < np.sqrt(p_norm)
        y_type = np.random.random() < np.sqrt(p_norm)
        x = generator_map[x_type](n_samples)
        y = generator_map[y_type](n_samples)
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=(9,10))
        tumor_info['tumor_type'] = tumor_type_map[x_type][y_type]
        tumor_info['thresh'] = np.random.normal(loc = heatmap.mean(), scale = heatmap.std()/2.)
        tumor_info['n_samples'] = n_samples
        self.tumor_region = (heatmap>tumor_info['thresh']).astype(int)
        tumor_info['size'] = self.tumor_region.sum()
        self.tumor_info = tumor_info
