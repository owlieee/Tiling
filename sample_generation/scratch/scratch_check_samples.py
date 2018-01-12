import pandas as pd
from tile_generator import TileSample, load_ranges
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
ranges = load_ranges()
sample_types = ['normal', 'nonresponder', 'responder']

results = pd.DataFrame()
for sample_type in sample_types:
    for i in range(0,100):
        sample = TileSample()
        sample.generate_sample(ranges, sample_type  = sample_type)
        temp =  sample.sample_info['gene_ranges']
        temp['sample_id'] = i
        if sample_type != 'normal':
            temp['tumor_size'] = sample.sample_info['tumor_size']
            temp['tumor_type'] = sample.sample_info['tumor_type']
        if sample_type == 'responder':
            temp['touching_tumor_size'] = sample.sample_info['touching_tumor_size']
        temp['sample_type'] = sample_type
        results = results.append(temp, ignore_index  = True)


nonresponder_tumor_ranges = ranges['_nonresponder_tumor']
nonresponder_tumor_ranges['sample_type'] = 'nonresponder'
responder_tumor_ranges = ranges['_responder_tumor']
responder_tumor_ranges['sample_type'] = 'responder'
tumor_ranges = responder_tumor_ranges.append(nonresponder_tumor_ranges, ignore_index = True)
normal_ranges = ranges['_normal']
for gene in normal_ranges['gene'].unique():
    responder_v = []
    nonresponder_v = []
    normal_row = normal_ranges[normal_ranges['gene']==gene]
    nonresp_row = tumor_ranges[(tumor_ranges['gene']==gene)& (tumor_ranges['sample_type']=='nonresponder')]
    resp_row = tumor_ranges[(tumor_ranges['gene']==gene) & (tumor_ranges['sample_type']=='responder')]
    normal = results[(results['sample_type']=='normal')
                        & (results['gene']==gene)]
    normal_v = [y for x in normal['array'].values.tolist() for y in x]
    if len(nonresp_row)>0:
        nonresponder = results[(results['sample_type']=='nonresponder')
                        & (results['region']=='tumor')
                        & (results['gene']==gene)]
        nonresponder_v = [y for x in nonresponder['array'].values.tolist() for y in x]
    if len(resp_row)>0:
        responder = results[(results['sample_type']=='responder')
                            & (results['region']=='tumor')
                            & (results['gene']==gene)]
        responder_v = [y for x in responder['array'].values.tolist() for y in x]
    _, bins = np.histogram(normal_v + nonresponder_v + responder_v, bins = 'auto')
    fig, ax = plt.subplots()
    ax.hist(normal_v, bins = bins,  label = 'normal', color = 'grey', alpha = 0.5 )
    ax.axvline(x = normal_row['min'].values[0], color = 'grey', alpha = 0.5, linestyle = '--')
    ax.axvline(x = normal_row['max'].values[0], color = 'grey',alpha = 0.5, linestyle = '--')
    if len(responder_v)>0:
        ax.hist(responder_v, bins =bins, label = 'responder in tumor region', color = 'blue', alpha = 0.5)
        ax.axvline(x = resp_row['min'].values[0], color = 'blue',alpha = 0.5, linestyle = '--')
        ax.axvline(x = resp_row['max'].values[0], color = 'blue', alpha = 0.5,linestyle = '--')
    if len(nonresponder_v)>0:
        ax.hist(nonresponder_v, bins =bins, label = 'nonresponder in tumor region', color = 'red', alpha = 0.5 )
        ax.axvline(x = nonresp_row['min'].values[0], color = 'red', alpha = 0.5, linestyle = '--')
        ax.axvline(x = nonresp_row['max'].values[0], color = 'red', alpha = 0.5,linestyle = '--')
    ax.legend()
    ax.set_title(gene + " Concentrations")
    plt.savefig('../figures/gene_concentrations/'+gene + '.png')
    #fig.show()
    plt.close('all')


touching_ranges = ranges['_responder_touching']
for gene in touching_ranges['gene'].unique():
    responder_v = []
    resp_row = touching_ranges[(touching_ranges['gene']==gene)]
    normal_row = normal_ranges[normal_ranges['gene']==gene]
    normal = results[(results['sample_type']=='normal')
                        & (results['gene']==gene)]
    normal_v = [y for x in normal['array'].values.tolist() for y in x]
    if len(resp_row)>0:
        responder = results[(results['sample_type']=='responder')
                            & (results['region']=='touching')
                            & (results['gene']==gene)]
        responder_v = [y for x in responder['array'].dropna().values.tolist() for y in x]
    _, bins = np.histogram(normal_v + responder_v, bins = 'auto')
    fig, ax = plt.subplots()
    ax.hist(normal_v, bins = bins,  label = 'normal', color = 'grey', alpha = 0.5 )
    ax.axvline(x = normal_row['min'].values[0], color = 'grey', alpha = 0.5, linestyle = '--')
    ax.axvline(x = normal_row['max'].values[0], color = 'grey',alpha = 0.5, linestyle = '--')
    if len(responder_v)>0:
        ax.hist(responder_v, bins =bins, label = 'responder touching tumor region', color = 'blue', alpha = 0.5)
        ax.axvline(x = resp_row['min'].values[0], color = 'blue',alpha = 0.5, linestyle = '--')
        ax.axvline(x = resp_row['max'].values[0], color = 'blue', alpha = 0.5,linestyle = '--')
    ax.legend()
    ax.set_title(gene + " Concentrations")
    fig.show()
    plt.savefig('../figures/gene_concentrations/'+gene + '.png')
    #fig.show()
    plt.close('all')


random_ranges = ranges['_responder_random']
for gene in random_ranges['gene'].unique():
    responder_v = []
    resp_row = random_ranges[(random_ranges['gene']==gene)]
    normal_row = normal_ranges[normal_ranges['gene']==gene]
    normal = results[(results['sample_type']=='normal')
                        & (results['gene']==gene)]
    normal_v = [y for x in normal['array'].values.tolist() for y in x]
    if len(resp_row)>0:
        responder = results[(results['sample_type']=='responder')
                            & (results['region']=='random')
                            & (results['gene']==gene)]
        responder_v = [y for x in responder['array'].dropna().values.tolist() for y in x]
    _, bins = np.histogram(normal_v + responder_v, bins = 'auto')
    fig, ax = plt.subplots()
    ax.hist(normal_v, bins = bins,  label = 'normal', color = 'grey', alpha = 0.5 )
    ax.axvline(x = normal_row['min'].values[0], color = 'grey', alpha = 0.5, linestyle = '--')
    ax.axvline(x = normal_row['max'].values[0], color = 'grey',alpha = 0.5, linestyle = '--')
    if len(responder_v)>0:
        ax.hist(responder_v, bins =bins, label = 'responder randomly changed region', color = 'blue', alpha = 0.5)
        ax.axvline(x = resp_row['min'].values[0], color = 'blue',alpha = 0.5, linestyle = '--')
        ax.axvline(x = resp_row['max'].values[0], color = 'blue', alpha = 0.5,linestyle = '--')
    ax.legend()
    ax.set_title(gene + " Concentrations")
    fig.show()
    plt.savefig('../figures/gene_concentrations/'+gene + '.png')
    #fig.show()
    plt.close('all')


fig, ax = plt.subplots()
results.groupby(['sample_type', 'sample_id'])['tumor_size'].mean().reset_index()['tumor_size'].hist(ax = ax)
fig.show()


fig, ax = plt.subplots()
responder['num_changed'].hist(ax = ax)
fig.show()

for i in range(0,100):
    sample = TileSample()
    sample.generate_sample(ranges, sample_type  = 'responder')
    fig, ax = plt.subplots()
    ax.imshow(sample.tumor_region)
    ax.set_xticks(np.arange(0.5,10.5))
    ax.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
    ax.set_yticks(np.arange(0.5,9.5))
    ax.set_yticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9'])
    ax.set_title('Sample Tumor, yellow = Tumor')
    plt.savefig('../figures/sample_tumors/' + str(i))
    plt.close('all')
