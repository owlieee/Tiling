import pandas as pd
import numpy as np
import collections
from get_tiling_ranges import store_ranges
import os
import pdb

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
        self.ranges = self._load_ranges()

    def generate_sample(self, sample_type=None):
        """
        generates a single random sample according to the rules for sample_type
        INPUTS:
        sample_type: string, Default = None
        OUTPUTS: None
        """
        self._generate_normal()
        if sample_type == None:
            self.sample_type = np.random.choice( ['normal', 'nonresponder', 'responder'])
        else:
            self.sample_type = sample_type
        if sample_type != 'normal':
            self._generate_tumor()
            self._modify_genes()

    def _generate_tumor(self, p_norm = 0.25):
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

    def _get_touching_tumor(self):
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
        t_edge = [edge for edge in zip(*list(t_edge-t_coord))]
        touch_array = np.zeros(tumor_region.shape)
        touch_array[t_edge] = 1
        return touch_array

    def _generate_normal(self):
        """
        Generate gene arrays with normal gene concentrations
        INPUT: None
        OUTPUT: None
        """
        normal_ranges = self.ranges['_normal']
        for ix, row in normal_ranges.iterrows():
            self.gene_arrays.update({row['gene']: np.random.normal(loc = row['mean'], scale = row['std'], size = (9, 10))})

    def _modify_genes(self):
        """
        modify genes according to rules for sample_type
        INPUT: None
        OUTPUT: None
        """
        if self.sample_type == 'nonresponder':
            self._update_region(ranges = self.ranges['_nonresponder_tumor'])
        else:
            self._update_region(ranges = self.ranges['_responder_tumor'])
            self._update_region(ranges = self.ranges['_responder_random'], region =  'random')
            self._update_region(ranges = self.ranges['_responder_touching'], region = 'touching')

    def _update_region(self, ranges = None, region = 'tumor', p_change = .11):
        """
        update genes in ranges dataframe to new ranges given region
        INPUT:
        ranges: dataframe
        region: string
        p_chage: float
        OUTPUT: None
        """
        if region == 'tumor':
            update = self.tumor_region
        elif region == 'touching':
            update = self._get_touching_tumor()
        else:
            update = np.random.choice(2, size = (9, 10), p = [1-p_change, p_change])
        update_ind = np.where(update)
        for ix, row in ranges.iterrows():
            gene_array = self.gene_arrays[row['gene']]
            mod_array = np.random.normal(loc = row['mean'], scale = row['std'], size = (9, 10))
            gene_array[update_ind] = mod_array[update_ind]
            self.gene_arrays.update({row['gene']: gene_array})

    def _load_ranges(self):
        """
        load range data from csvs
        INPUT: None
        OUTPUT: dictionary
        """
        files = ['../data/ranges/'+f for f in os.listdir('../data/ranges/') if f.split('.')[-1]=='csv']
        ranges = {f.split('.csv')[0].split('/')[-1]: pd.read_csv(f) for f in files}
        return ranges

if __name__=='__main__':
    sample = TileSample()
    sample.generate_sample(sample_type = 'responder')

#look up tgen
