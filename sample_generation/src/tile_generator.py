import pandas as pd
import numpy as np
import collections
from get_tiling_ranges import store_ranges
import os


class TileSample:
    """
    Generates a single tile sample
    """
    def __init__(self):
        """
        ATTRIBUTES:
        sample_type: list, ['normal', 'nonresponder', 'responder']
        tumor_region: array, randomly generated tumor (empty for normal)
        tumor_info: dictionary, metadata about tumor region, size, tumor_type (normal, random_xy, random_x, random_y)
        gene_arrays: dictionary, store 2d array for each gene
        ranges: dictionary, expected ranges for sample_types, genes in/out of tumor_region
        """
        self.sample_type = None
        self.tumor_region = None
        self.tumor_info = None
        self.gene_arrays = {}
        self.ranges = self.load_ranges()

    def generate_sample(self, sample_type=None):
        """
        generates a single random sample according to the rules for sample_type
        INPUTS:
        sample_type: string, Default = None
        OUTPUTS: None
        """
        self.generate_normal()
        if sample_type == None:
            self.sample_type = np.random.choice( ['normal', 'nonresponder', 'responder'])
        else:
            self.sample_type = sample_type
        if sample_type != 'normal':
            self.generate_tumor()

    def generate_tumor(self, p_norm = 0.5):
        """
        generates random profile of a tumor_region
        stores metadata
        INPUTS:
        p_norm: float, probability of clustered tumor vs. scattered
        OUTPUTS: None
        """
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

    def get_touching_tumor(self):
        """
        returns binary array where 1 = touching tumor cell
        INPUTS: None
        OUTPUTS: 2 dimensional array
        """
        tumor_region = self.tumor_region
        t = np.where(tumor_region==1)
        t_coord = set(zip(t[0], t[1]))
        rowBound, colBound = (tumor_region.shape[0] - 1, tumor_region.shape[1] - 1)
        rowShifts = [t[0], np.clip(t[0]+1, 0, rowBound), np.clip(t[0]-1, 0, rowBound)]
        colShifts = [t[1], np.clip(t[1]+1, 0, colBound), np.clip(t[1]-1, 0, colBound)]
        t_edge = set()
        for row in rowShifts:
            for col in colShifts:
                t_edge.update(zip(row, col))
        t_edge = zip(*list(t_edge-t_coord))
        touch_array = np.zeros(tumor_region.shape)
        touch_array[t_edge] = 1
        return touch_array

    def generate_normal(self):
        """
        INPUT: None
        OUTPUT: None
        """
        normal_ranges = self.ranges['normal']
        for ix, row in normal_ranges.iterrows():
            self.gene_arrays.update({row['gene']: np.random.normal(loc = row['mean'], scale = row['std'], size = (9, 10))})

    def modify_genes(self):
        update_map = {'tumor': update_tumor, 'random', update_random, 'touching': update_touching}
        for data_type, ranges in self.ranges:
            if '_' + self.sample_type in data_type:
                update_function = update_map[data_type.split('_')[-1]]
                d = data_type
                self.modify(data_type)

    def update_tumor(self):
        pass

    def update_random(self):
        pass

    def update_touching(self):
        pass

    def load_ranges(self):
        files = ['../data/ranges/'+f for f in os.listdir('../data/ranges/') if f.split('.')[-1]=='csv']
        ranges = {f.split('.csv')[0].split('/')[-1]: pd.read_csv(f) for f in files}
        return ranges

if __name__=='__main__':
    sample = TileSample()
