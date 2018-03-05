import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from tile_generator import load_ranges, TileSample
import matplotlib
import h5py
import numpy as np
style.use('ggplot')

def make_numeric(value):
    """
    convert empty list to 0
    """
    if value==[]:
        return 0
    else:
        return value

def make_str(value):
    """
    convert empty list to string
    create tag for distribution type
    """
    if value==[]:
        return ''
    else:
        x_dist =value['x_dist'].__name__
        y_dist = value['y_dist'].__name__
        return 'X_'+x_dist + '_Y_' + y_dist

def make_sample_metadata(sample):
    """
    convert sample object into series of metadata
    """
    sample_metadata = pd.Series({'sample_type': sample.sample_type,
        'signal_purity': make_numeric(sample.sample_info['signal_purity']),
        'tumor_percent': make_numeric(sample.sample_info['tumor_percent']),
        'tumor_size': make_numeric(sample.sample_info['tumor_size']),
        'tumor_type': make_str(sample.sample_info['tumor_type']),
        'touching_tumor_size': make_numeric(sample.sample_info['touching_tumor_size'])})
    return sample_metadata

def get_groups():
    """
    order for plotting
    """
    group1 = ['KRAS', 'EGFR', 'CDKN2A', 'IDH1', 'APC', 'PIK3CA', 'SMARCB1']
    group2 = ['GNA11','NOTCH1', 'SMO', 'HRAS', 'ABL1', 'JAK3', '']
    group3 = ['AKT1','ALK','ATM','BRAF','CDH1','CSF1R','CTNNB1']
    group4 = ['ERBB2','ERBB4','FBXW7','FGFR1','FGFR2','FGFR3','FLT3']
    group5 = ['GNAQ','GNAS','HNF1A','JAK2','KDR','KIT','MET']
    group6 = ['MLH1','MPL','NPM1','NRAS','PDGFRA','PTEN','PTPN11']
    group7 = ['RB1','RET','SMAD4','SRC','STK11','TP53','VHL']
    groups = group1+group2+group3+group4+group5+group6+group7
    return groups

def standardize_gene_array(gene_array, gene):
    """
    standardize gene_array with normal mean and std for gene
    """
    gene_ranges =  ranges['normal'][ranges['normal']['gene']==gene]
    mean = gene_ranges['mean'].iloc[0]
    std = gene_ranges['std'].iloc[0]
    return ((gene_array-mean)/std)

def patch(x, y, hatch, color):
    """
    annotate cells with textured rectangle
    """
    return matplotlib.patches.Rectangle((x-0.5, y-0.5), 1, 1, hatch = hatch, fill = False, color = color)

def plot_gene_heatmaps(sample, standardize = False):
    """
    make gene heatmap plot with annotated tumor region
    """
    groups = get_groups()
    fig, ax = plt.subplots(7, 7, frameon = False, figsize = (8,8))
    gene_arrays = []
    for ix, ax in enumerate(ax.flatten()):
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        gene = groups[ix]
        if gene == '':
            continue
        ax.set_title(gene)
        gene_array = sample.gene_arrays[gene]
        vmin = 5
        vmax = 40
        if standardize==True:
            gene_array = standardize_gene_array(gene_array, gene)
            vmin = -5
            vmax = 5
        gene_arrays.append(gene_array)
        #colormap_r = matplotlib.colors.ListedColormap('RdYlGn')#.reversed()
        im =ax.imshow( gene_array, cmap = 'RdYlGn', interpolation = None, vmin = vmin, vmax = vmax)
        for y,x in zip(np.where(sample.tumor_region>0)[0], np.where(sample.tumor_region>0)[1]):
            ax.add_patch(patch(x, y, '///', 'white'))
    fig.suptitle(sample.sample_type +
            " signal purity = " + str(np.round(sample.sample_info['signal_purity']*100)) +\
            "% Tumor size = " + str(sample.sample_info['tumor_size']))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    return gene_arrays, fig

def series_to_sample(series):
    """
    Convert pandas series to tile sample object
    """
    sample = TileSample()
    genes = get_groups()
    for gene in genes:
        if gene == '':
            continue
        sample.gene_arrays[gene]=np.array(series[gene])
    sample.sample_type =  series['sample_type']
    sample.tumor_region = np.array(series['tumor_region'])
    for col in [u'index', u'signal_purity', u'touching_tumor_size',
        u'tumor_percent', u'tumor_size', u'tumor_type']:
        sample.sample_info[col]=series[col]
    return sample

def sample_to_hdf5(sample, fname):
    """
    store tile sample object as hdf5 object
    """
    hf = h5py.File(fname + '.h5', 'w')
    g1 = hf.create_group('gene_arrays')
    for gene, gene_array in sample.gene_arrays.items():
        g1.create_dataset(gene, data=gene_array)
    g1.attrs.create('sample_type', data = sample.sample_type)
    g2 = hf.create_group('sample_info')
    g2.create_dataset('tumor_region', data =sample.tumor_region)
    sample_metadata = make_sample_metadata(sample)
    for k, v in sample_metadata.items():
        g2.attrs.create(k, data = v)
    hf.close()

def hdf5_to_sample(fname):
    """
    load tile sample object from hdf5
    """
    sample = TileSample()
    hf = h5py.File(fname, 'r')
    for i,v in hf['gene_arrays'].items():
        sample.gene_arrays[i]=np.array(v)
    sample.sample_type =  hf['gene_arrays'].attrs['sample_type']
    sample.tumor_region =np.array(hf['sample_info']['tumor_region'])
    for i,v in hf['sample_info'].attrs.items():
        sample.sample_info[i]=v
    hf.close()
    return sample

if __name__ == '__main__':
    #load normal and modified ranges from google docs
    ranges = load_ranges()
    #initialize sample
    sample = TileSample()
    #choose any filename
    filename = 'responder_sample'
    #generate sample:
    #  sample_type options: 'responder, 'nonresponder', 'normal'
    #  tumor_size: 0-90, ignored for normal samples
    sample.generate_sample(ranges, sample_type = 'responder', tumor_size =20)
    #save sample as hdf5 file
    sample_to_hdf5(sample, filename)
    #plot sample (heatmaps of each gene)
    gene_arrays, fig = plot_gene_heatmaps(sample, standardize = False)
    #save figure
    plt.savefig(filename+'.png')
    #show figure
    fig.show()
    #--------------
    #To load previously saved file:
    #sample = hdf5_to_sample(filename+'.h5')
