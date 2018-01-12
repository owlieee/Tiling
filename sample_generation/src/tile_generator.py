import pandas as pd
import numpy as np
import collections
from collections import defaultdict
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
        sample_info: dictionary, metadata about sample
        tumor_region: array, randomly generated tumor (empty for normal)
        gene_arrays: dictionary, store 2d array for each gene
        ranges: dictionary, expected ranges for sample_types, genes in/out of tumor_region
        """
        self.sample_type = None
        self.sample_info = defaultdict(list)
        self.tumor_region = None
        self.gene_arrays = {}
        self.ranges = None

    def generate_sample(self, ranges, sample_type=None):
        """
        generates a single random sample according to the rules for sample_type
        INPUTS:
        sample_type: string, Default = None
        OUTPUTS: None
        """
        self.ranges = ranges
        self._generate_normal()
        if not sample_type:
            self.sample_type = np.random.choice( ['normal', 'nonresponder', 'responder'])
        else:
            self.sample_type = sample_type

        if sample_type != 'normal':
            self._generate_tumor()
            self._modify_genes()

    def _generate_tumor(self, p_norm = 0.5):
        """
        generates random profile of a tumor_region
        stores metadata
        INPUTS:
        p_norm: float, probability of clustered tumor vs. scattered
        OUTPUTS: None
        """

        n_samples = max(10, int(np.random.normal(loc = 120, scale = 20)))
        generator_map = {True: np.random.randn, False: np.random.rand}
        tumor_type_map = {True: {True: 'normal', False: 'random_y'}, False: {True: 'random_x', False: 'random_xy'}}

        x_type = np.random.random() < np.sqrt(p_norm)
        y_type = np.random.random() < np.sqrt(p_norm)

        x = generator_map[x_type](n_samples)
        y = generator_map[y_type](n_samples)
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=(9,10))
        self.sample_info['tumor_type'] = tumor_type_map[x_type][y_type]
        self.sample_info['thresh'] = heatmap.mean()#np.random.normal(loc = heatmap.mean(), scale = heatmap.std()/2.)
        self.sample_info['n_samples'] = n_samples
        self.tumor_region = (heatmap>self.sample_info['thresh']).astype(int)
        self.sample_info['tumor_size'] = self.tumor_region.sum()

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
        self.sample_info['touching_tumor_size'] = np.sum(touch_array)
        return touch_array

    def _get_random_ind(self, p_change = 0.11):
        """
        randomly set indices to 1 or 0 in 9x10 array, p(1) = p_change
        INPUT:
        p_change: float, optional
        OUTPUT:
        array
        """
        return np.random.choice(2, size = (9, 10), p = [1-p_change, p_change])

    def _generate_normal(self):
        """
        Generate gene arrays with normal gene concentrations
        INPUT: None
        OUTPUT: None
        """
        for ix, row in self.ranges['_normal'].iterrows():
            gene_array = np.random.normal(loc = row['mean'], scale = row['std'], size = (9, 10))
            #self._store_summary_stats(row['gene'], 'normal', gene_array )
            self.gene_arrays.update({row['gene']: gene_array})

    def _modify_genes(self):
        """
        modify genes according to rules for sample_type
        INPUT: None
        OUTPUT: None
        """
        if self.sample_type == 'nonresponder':
            self._update_region(ranges=self.ranges['_nonresponder_tumor'])
        else:
            self._update_region(ranges = self.ranges['_responder_tumor'])
            self._update_region(ranges = self.ranges['_responder_random'], region =  'random')
            self._update_region(ranges = self.ranges['_responder_touching'], region = 'touching')

    def _store_summary_stats(self, gene, region, gene_array):
        try:
            if len(self.sample_info['gene_ranges'])==0:
                self.sample_info['gene_ranges'] = pd.DataFrame()
            if len(gene_array)==0:
                self.sample_info['gene_ranges'] = self.sample_info['gene_ranges'].append(
                                                    pd.Series({'gene': gene,
                                                    'region': region,
                                                    'num_changed': len(gene_array.flatten())})
                                                        , ignore_index = True)
            else:
                self.sample_info['gene_ranges'] = self.sample_info['gene_ranges'].append(
                                                    pd.Series({'gene': gene,
                                                    'region': region,
                                                    'num_changed': len(gene_array.flatten()),
                                                    'array': gene_array.flatten(),
                                                    'mean': np.mean(gene_array),
                                                    'std': np.std(gene_array),
                                                    'min': np.min(gene_array),
                                                    'max': np.max(gene_array)}), ignore_index = True)
        except:
            pdb.set_trace()
    def _get_update_ind(self, region):
        """
        find indices to update based on region
        INPUT:
        region: string
        OUTPUT:
        update_ind: 9x10 array of ones and zeros
        """
        if region == 'tumor':
            update_ind = np.where(self.tumor_region)
        elif region == 'touching':
            update_ind = np.where(self._get_touching_tumor())
        elif region == 'random':
            update_ind = np.where(self._get_random_ind())
        return update_ind

    def _update_region(self, ranges=None, region='tumor'):
        """
        update genes in ranges dataframe to new ranges given region
        INPUT:
        ranges: dataframe
        region: string
        p_chage: float
        OUTPUT: None
        """
        update_ind = self._get_update_ind(region)
        for ix, gene_range in ranges.iterrows():
            gene_array = self.gene_arrays[gene_range['gene']]
            mod_array = np.random.normal(loc = gene_range['mean'], scale = gene_range['std'], size = (9, 10))
            gene_array[update_ind] = mod_array[update_ind]
            self.gene_arrays.update({gene_range['gene']: gene_array})
            #self._store_summary_stats(gene_range['gene'], region, mod_array[update_ind])
            if region == 'random': #reset random region
                update_ind = self._get_update_ind(region)



def load_ranges():
    """
    load range data from csvs
    INPUT: None
    OUTPUT: dictionary
    """
    files = ['../data/ranges/'+f for f in os.listdir('../data/ranges/') if f.split('.')[-1]=='csv']
    ranges = {f.split('.csv')[0].split('/')[-1]: pd.read_csv(f) for f in files}
    return ranges


if __name__=='__main__':
    ranges = load_ranges()
    sample = TileSample()
    sample.generate_sample(ranges)

#look up tgen

#next step-- generate 100 of each, implement histograms for data quality check.
