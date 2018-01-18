import pandas as pd
import numpy as np
import collections
from collections import defaultdict
from get_tiling_ranges import store_ranges
import os
import pdb
import time

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
        self.gene_arrays = pd.Series()
        self.ranges = None
        self._array_size = (9,10)
        self._gene_range = (5,40)

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

        self._calculate_signal_purity()


    def _convert_to_list(self):
        #self.gene_arrays = pd.Series(self.gene_arrays)
        self.gene_arrays = self.gene_arrays.apply(lambda x: [list(v) for v in x])
        if self.tumor_region is not None:
            self.tumor_region = [list(v) for v in self.tumor_region]

    def _calculate_signal_purity(self):
        """
        calculate signal purity based on how many samples were changed as expected
        INPUTS: None
        OUTPUTS: None
        """
        normal_pct = self._calculate_normal_pct()
        if self.sample_type =='normal':
            self.sample_info['signal_purity'] = normal_pct
        elif self.sample_type == 'nonresponder':
            num_normal = np.product(self._array_size) - self.sample_info['tumor_size']
            normal_weight = num_normal * normal_pct
            tumor_weight = self._calculate_tumor_weight()
            self.sample_info['signal_purity'] = (normal_weight + tumor_weight)/np.product(self._array_size)
        else:
            num_normal = np.product(self._array_size) - self.sample_info['tumor_size'] - self.sample_info['touching_tumor_size']
            normal_weight = num_normal * normal_pct
            tumor_weight = self._calculate_tumor_weight()
            touching_tumor_weight = self._calculate_touching_tumor_weight()
            self.sample_info['signal_purity'] = (normal_weight + \
                        tumor_weight + \
                        touching_tumor_weight)/np.product(self._array_size)

    def _calculate_touching_tumor_weight(self):
        """
        helper function for calculating percent of signal in tumor region
        """
        touching_tumor_changes = self.sample_info['gene_ranges'].loc[self.sample_info['gene_ranges']['region']=='touching_tumor'][['gene', 'num_changed']]
        if len(touching_tumor_changes)>0:
            touching_tumor_pct = np.mean(touching_tumor_changes['num_changed']/self.sample_info['touching_tumor_size'])
        return touching_tumor_pct*self.sample_info['touching_tumor_size']

    def _calculate_tumor_weight(self):
        tumor_changes = self.sample_info['gene_ranges'].loc[self.sample_info['gene_ranges']['region']=='tumor'][['gene', 'num_changed']]
        if len(tumor_changes)>0:
            tumor_pct = np.mean(tumor_changes['num_changed']/self.sample_info['tumor_size'])
        return tumor_pct*self.sample_info['tumor_size']

    def _calculate_normal_pct(self):
        if len(self.sample_info['gene_ranges'])>0:
            normal_changes = self.sample_info['gene_ranges'].loc[self.sample_info['gene_ranges']['region']=='normal_outlier'][['gene', 'num_changed']]
            if len(normal_changes)>0:
                n = normal_changes.merge(self.ranges['normal'][['gene', 'prob_outlier']], how = 'outer').fillna(0)
                normal_pct = np.mean((90-n['num_changed'])/90.)
        else:
            normal_pct = 1
        return normal_pct

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


    def _get_random_ind(self, p_change = None):
        """
        randomly set indices to 1 or 0 in 9x10 array, p(1) = p_change
        INPUT:
        p_change: float, optional
        OUTPUT:
        array
        """
        if not p_change:
            p_change = np.clip(np.random.exponential(scale = 0.1), 1./np.product(self._array_size), 1)
        return np.random.choice(2, size = self._array_size, p = [1-p_change, p_change])

    def _generate_normal(self):
        """
        Generate gene arrays with normal gene concentrations
        Randomly create outliers with probability set in gene info table
        INPUT: None
        OUTPUT: None
        """

        for ix, gene_info in self.ranges['normal'].iterrows():
            mean = np.random.normal(loc = gene_info['mean'], scale = gene_info['std'])
            gene_array = np.random.normal(loc = mean, scale = gene_info['std'], size = self._array_size)
            #self.gene_arrays.update({gene_info['gene']: gene_array})
            self.gene_arrays[gene_info['gene']] = gene_array
            if np.random.random()<gene_info['prob_outlier']:
                self._set_normal_outliers(gene_info)

    def _set_normal_outliers(self, gene_info):
        """
        Randomly generate outliers in normal gene_array with broader uniform range
        INPUT: None
        OUTPUT: None
        """
        alter_ind = np.where(self._get_random_ind()==1)
        if len(alter_ind)>0:
            #randomly set min, max for uniform distribution to 2-5 standard deviations from mean
            uniform_min = max(gene_info['mean']-gene_info['std']*(2+2*np.random.random()), self._gene_range[0])
            uniform_max = min(gene_info['mean']+gene_info['std']*(2+2*np.random.random()), self._gene_range[1])
            self.gene_arrays[gene_info['gene']][alter_ind] = np.random.uniform(uniform_min,
                                                                               uniform_max,
                                                                               size = self._array_size)[alter_ind]
            self._store_summary_stats(gene_info['gene'], 'normal_outlier', self.gene_arrays[gene_info['gene']][alter_ind])

    def _modify_genes(self):
        """
        modify genes for all regions specified by sample_type
        INPUT: None
        OUTPUT: None
        """
        change_ranges = self.ranges['changes'].loc[self.ranges['changes']['sample_type']==self.sample_type]
        for region in change_ranges['region'].unique():
            self._update_region(ranges = change_ranges.loc[change_ranges['region']==region], region = region)

    def _store_summary_stats(self, gene, region, gene_array):
        if len(self.sample_info['gene_ranges'])==0:
            self.sample_info['gene_ranges'] = pd.DataFrame()
        self.sample_info['gene_ranges'] = self.sample_info['gene_ranges'].append(
                                            pd.Series({'gene': gene,
                                            'region': region,
                                            'num_changed': gene_array.size}), ignore_index = True)

    def _get_update_ind(self, region):
        """
        find indices to update based on region
        INPUT:
        region: string
        OUTPUT:
        update_ind: 9x10 array of ones and zeros
        """
        if region == 'tumor':
            update_ind = np.where(self.tumor_region>0)
        elif region == 'touching_tumor':
            update_ind = np.where(self._get_touching_tumor())
        elif region == 'random':
            update_ind = np.where(self._get_random_ind())
        return update_ind

    def _randomly_sparse_ind(self, expected_update_ind, prob_change):
        """
        helper function to randomly set indices to zero
        INPUT:
        ind: 2d array
        output: 2d array
        """
        if np.random.random()<prob_change:
            update_ind = expected_update_ind
        else:
            a = np.zeros(self._array_size)
            a[expected_update_ind] = 1
            update_ind = np.where(a*self._get_random_ind())
        return update_ind

    def _update_region(self, ranges=None, region='tumor'):
        """
        update genes in ranges dataframe to new ranges given region
        randomly sparsify the region based on prob_change
        INPUT:
        ranges: dataframe
        region: string
        p_change: float
        OUTPUT: None
        """
        #outside for loop to avoid repeat calls for tumor and touching tumor regions
        expected_update_ind = self._get_update_ind(region)

        for ix, gene_info in ranges.iterrows():
            #randomly sparsify with probability gene_info['prob_changes']
            update_ind = self._randomly_sparse_ind(expected_update_ind, gene_info['prob_change'])
            mod_array = np.random.normal(loc = gene_info['mean'], scale = gene_info['std'], size = (9, 10))
            self.gene_arrays[gene_info['gene']][update_ind] = mod_array[update_ind]
            self._store_summary_stats(gene_info['gene'], region, mod_array[update_ind])
            if region == 'random':
                #always reset for random region
                expected_update_ind = self._get_update_ind(region)




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
