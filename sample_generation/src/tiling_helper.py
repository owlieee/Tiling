import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from tile_generator import load_ranges, TileSample
import matplotlib
import h5py
style.use('ggplot')

def make_numeric(value):
    if value==[]:
        return 0
    else:
        return value

def make_str(value):
    if value==[]:
        return ''
    else:
        x_dist =value['x_dist'].__name__
        y_dist = value['y_dist'].__name__
        return 'X_'+x_dist + '_Y_' + y_dist

def make_sample_metadata(sample):
    sample_metadata = pd.Series({'sample_type': sample.sample_type,
        'signal_purity': make_numeric(sample.sample_info['signal_purity']),
        'tumor_percent': make_numeric(sample.sample_info['tumor_percent']),
        'tumor_size': make_numeric(sample.sample_info['tumor_size']),
        'tumor_type': make_str(sample.sample_info['tumor_type']),
        'touching_tumor_size': make_numeric(sample.sample_info['touching_tumor_size'])})
    return sample_metadata

def get_groups():
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
    gene_ranges =  ranges['normal'][ranges['normal']['gene']==gene]
    mean = gene_ranges['mean'].iloc[0]
    std = gene_ranges['std'].iloc[0]
    #pdb.set_trace()
    return ((gene_array-mean)/std)

def patch(x, y, hatch, color):
    return matplotlib.patches.Rectangle((x-0.5, y-0.5), 1, 1, hatch = hatch, fill = False, color = color)

def plot_gene_heatmaps(sample, standardize = False):
    groups = get_groups()
    fig, ax = plt.subplots(7, 7, frameon = False, figsize = (8,8))
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
        im =ax.imshow( gene_array, cmap = 'coolwarm', interpolation = None, vmin = vmin, vmax = vmax)
        for y,x in zip(np.where(sample.tumor_region>0)[0], np.where(sample.tumor_region>0)[1]):
            ax.add_patch(patch(x, y, '///', 'black'))
    fig.suptitle(sample.sample_type +
            " signal purity = " + str(np.round(sample.sample_info['signal_purity']*100)) +\
            "% Tumor size = " + str(sample.sample_info['tumor_size']))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    return fig

def sample_to_hdf5(sample, fname):
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

# ranges = load_ranges()
# sample = TileSample()
# sample.generate_sample(ranges)
#
# sample = hdf5_to_sample('data.h5')
# fig = plot_gene_heatmaps(sample, standardize = False)
# #plt.savefig('gene_sample_responder_standardized')
# fig.show()
