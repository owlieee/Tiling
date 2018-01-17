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
        self._array_size = (9,10)

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

        if self.sample_type != 'normal':
            self._generate_tumor()
            self._modify_genes()


    def _generate_tumor(self, size = None):
        """
        generates random profile of a tumor_region.
        Optional input size to set number of tumor cells
        stores metadata
        INPUTS: size: int
        OUTPUTS: None
        """

        #generate tumor heatmap from randomly selected attributes
        self._set_random_tumor_attributes(size = size)
        self._set_tumor_heatmap()


    def _set_random_tumor_attributes(self, size = None):
        """
        randomly choose attributes for tumor generation with optional input "size"
        INPUTS: size: int or None
        OUTPUTS: None
        """
        array_area = np.product(self._array_size)

        if size is None:
            self.sample_info['tumor_size'] = float(np.floor(np.random.uniform(1,array_area)))
        else:
            self.sample_info['tumor_size'] = float(size)

        self.sample_info['tumor_percent'] = 100*self.sample_info['tumor_size']/float(array_area)

        #randomly choose normal or random distribution for x and y
        tumor_type_map = {True: np.random.randn, False: np.random.rand}
        self.sample_info['tumor_type'] = {'x_dist': tumor_type_map[np.random.random() < 0.5],
                                         'y_dist' : tumor_type_map[np.random.random() < 0.5]}

    def _set_tumor_heatmap(self):
        """
        use tumor attributes to generate tumor heatmap that covers set percentage
        of array area. store as tumor_region
        INPUTS: None
        OUTPUTS: None
        """
        #generate random x and y arrays from preselected distributions
        n_samples = 100
        x = self.sample_info['tumor_type']['x_dist'](n_samples)
        y = self.sample_info['tumor_type']['y_dist'](n_samples)
        #generate 2d histogram from x and y arrays
        self.tumor_region, _, _ = np.histogram2d(x, y, bins= self._array_size)
        #randomly increase other indices to achieve sample size
        self.tumor_region = self.tumor_region + np.random.random(self._array_size)
        thresh = np.percentile(self.tumor_region, 100-self.sample_info['tumor_percent'])
        self.tumor_region[self.tumor_region<=thresh]=0

    def _get_touching_tumor(self):
        """
        returs binary array where 1 = touching tumor cell
        INPUTS: None
        OUTPUTS: touching_array: 2 dimensional array
        """
        touching_ind = self._get_touching_ind()
        touching_array = np.zeros(self.tumor_region.shape)
        touching_array[touching_ind] = 1
        #store metadata
        self.sample_info['touching_tumor_size'] = np.sum(touching_array)
        return touching_array

    def _get_touching_ind(self):
        """
        get list of coordinates for all cells touching tumor, including diagonals,
        not including tumor coordinates
        INPUTS: None
        OUTPUTS: touching_ind: list, contains x and y indices
        """
        tumor_row_ind, tumor_col_ind = np.where(self.tumor_region>0)

        #get coordinates for all cells up/down/left/right/diagonal of tumor.
        #may overlap with tumor
        shifted_coord = self._get_shifted_coord(tumor_row_ind, tumor_col_ind)
        tumor_coord = set(zip(tumor_row_ind, tumor_col_ind))
        #remove coordinates that are also tumor cells
        touching_coord = list(shifted_coord -tumor_coord)
        #format indices properly for array indexing
        touching_ind = [edge for edge in zip(*touching_coord)]
        return touching_ind

    def _get_shifted_coord(self, tumor_row_ind, tumor_col_ind):
        """
        get list of tumor row and column indices, normal, shifted up
        and shifted down, clipped to not exceed valid indices for array size
        INPUT: tumor_row_ind: 1d array
               tumor_col_ind: 1d array
        OUTPUT: coord_shifts: set
        """
        row_bound, col_bound = (self._array_size[0] - 1, self._array_size[1] - 1)
        #list of all row indices and adjacent to row indices (right and left)
        row_shifts = [tumor_row_ind,
                    np.clip(tumor_row_ind+1, 0, row_bound),
                    np.clip(tumor_row_ind-1, 0, row_bound)]
        #list of all col indices and adjacent to col indices (above and below)
        col_shifts = [tumor_col_ind,
                    np.clip(tumor_col_ind+1, 0, col_bound),
                    np.clip(tumor_col_ind-1, 0, col_bound)]
        #nested for loop finds all indices left, right, up, down and diagonal
        #to original tumor_row_ind and tumor_col_ind, clipped to array_size
        shifted_coord = set()
        for row in row_shifts:
            for col in col_shifts:
                shifted_coord.update(zip(row, col))
        return shifted_coord


    def _get_random_ind(self, p_change = 0.11):
        """
        randomly set indices to 1 or 0 in 9x10 array, p(1) = p_change
        INPUT:
        p_change: float, optional
        OUTPUT:
        array
        """
        return np.random.choice(2, size = self._array_size, p = [1-p_change, p_change])

    def _generate_normal(self):
        """
        Generate gene arrays with normal gene concentrations
        INPUT: None
        OUTPUT: None
        """
        for ix, row in self.ranges['_normal'].iterrows():
            gene_array = np.random.normal(loc = row['mean'], scale = row['std'], size = (9, 10))
            self._store_summary_stats(row['gene'], 'normal', gene_array )
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
                                                'mean': np.mean(gene_array),
                                                'std': np.std(gene_array),
                                                'min': np.min(gene_array),
                                                'max': np.max(gene_array)}), ignore_index = True)

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
            self._store_summary_stats(gene_range['gene'], region, mod_array[update_ind])
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
