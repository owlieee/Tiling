import csv
import os
import pandas as pd
files = ['Tiling Training Set/'+f for f in os.listdir('Tiling Training Set/') if f.split('.')[-1]=='csv']


df = pd.DataFrame(columns = ['gene', 'sample_type', 't_array', 'gene_array'])

responder_t = np.array([[1,1,1,0,0,0,0,0,0,0],
                        [1,1,1,0,0,0,0,0,0,0],
                        [1,1,1,0,0,0,0,0,0,0],
                        [1,1,1,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0]])
responder_touch_t = np.array([[0,0,0,1,0,0,0,0,0,0],
                        [0,0,0,1,0,0,0,0,0,0],
                        [0,0,0,1,0,0,0,0,0,0],
                        [0,0,0,1,0,0,0,0,0,0],
                        [1,1,1,1,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0]])
nonresponder_t = np.array([[0,0,0,1,1,1,0,0,0,0],
                        [0,0,0,1,1,1,0,0,0,0],
                        [0,0,0,1,1,1,0,0,0,0],
                        [0,0,0,1,1,1,0,0,0,0],
                        [0,0,0,1,1,1,0,0,0,0],
                        [0,0,0,1,1,1,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0]])
normal_t = np.zeros((9,10))

genes = ['ABL1', 'AKT1', 'ALK', 'APC', 'ATM', 'BRAF', 'CDH1', 'CDKN2A',
       'CSF1R', 'CTNNB1', 'EGFR', 'ERBB2', 'ERBB4', 'FBXW7', 'FGFR1',
       'FGFR2', 'FGFR3', 'FLT3', 'GNA11', 'GNAQ', 'GNAS', 'HNF1A', 'HRAS',
       'IDH1', 'JAK2', 'JAK3', 'KDR', 'KIT', 'KRAS', 'MET', 'MLH1', 'MPL',
       'NOTCH1', 'NPM1', 'NRAS', 'PDGFRA', 'PIK3CA', 'PTEN', 'PTPN11',
       'RB1', 'RET', 'SMAD4', 'SMARCB1', 'SMO', 'SRC', 'STK11', 'TP53',
       'VHL']

responder_t_changes = {'CDKN2A', 'NOTCH', 'EGFR', 'IDH1', 'KRAS', 'SMARCB1', 'APC', 'GNA11', 'PIK3CA'}
responder_touch_t_changes = {'HRAS', 'SMO'}
responder_rand_changes = {'ABL1', 'JAK3'}
nonresponder_t_changes = {'CDKN2A', 'EGFR', 'IDH1', 'KRAS', 'SMARCB1', 'APC', 'PIK3CA'}

t_dict = {'normal': {g:normal_t for g in genes},
        'nonresponder': {g:normal_t for g in genes},
        'responder': {g:normal_t for g in genes}}
t_dict['nonresponder'].update({g:nonresponder_t for g in genes if g in nonresponder_t_changes})
t_dict['responder'].update({g:responder_t for g in genes if g in responder_t_changes})
t_dict['responder'].update({g:responder_touch_t for g in genes if g in responder_touch_t_changes})

for f in files:
    sample_type = f.split('.')[0].split('-')[-1]
    with open(f, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        for row in reader:
            if row[0]=='':
                continue
            if (str.isalpha(row[0][0])) & (row[1]==''):
                gene = row[0]
                gene_array = np.zeros((9,10))
                t_array = t_dict[sample_type][gene]
                i = 0
            elif str.isdigit(row[0][0]):
                concentrations = np.array([float(x) for x in row])
                gene_array[i, :] = concentrations
                i +=1
                if i==gene_array.shape[0]:
                    vals = {}
                    vals['sample_type'] = sample_type
                    vals['gene'] = gene
                    vals['t_array'] = t_array
                    vals['gene_array'] = gene_array
                    df = df.append(pd.Series(vals), ignore_index = True)
            else:
                continue

df['gene_t_array']=(df['gene_array'] * df['t_array']).apply(lambda x: x[x>0])
df['gene_nt_array']=(df['gene_array'] * df['t_array'].apply(lambda x: (x==0).astype(int))).apply(lambda x: x[x>0])

df['nt_min'] = df['gene_nt_array'].apply(lambda x: x.min())
df['nt_max'] = df['gene_nt_array'].apply(lambda x: x.max())
df['t_min'] = df['gene_t_array'].apply(lambda x: x.min() if len(x) > 0 else np.nan)
df['t_max'] = df['gene_t_array'].apply(lambda x: x.max() if len(x) > 0 else np.nan)


ranges = df.groupby(['sample_type', 'gene']).mean().reset_index()
normal = ranges[ranges['sample_type']=='normal'][['gene', 'nt_min', 'nt_max']]
normal = normal.rename(columns = {'nt_min': 'normal_min', 'nt_max': 'normal_max'})
responder = ranges[ranges['sample_type']=='responder'][['gene', 'nt_min', 'nt_max', 't_min', 't_max']]
responder = responder.rename(columns = {'nt_min': 'resp_min', 'nt_max': 'resp_max',
                            't_min': 'resp_tumorregion_min', 't_max': 'resp_tumorregion_max'})
nonresponder = ranges[ranges['sample_type']=='nonresponder'][['gene', 'nt_min', 'nt_max', 't_min', 't_max']]
nonresponder = nonresponder.rename(columns = {'nt_min': 'nonresp_min', 'nt_max': 'nonresp_max',
                            't_min': 'nonresp_tumorregion_min', 't_max': 'nonresp_tumorregion_max'})
ranges = normal.merge(responder).merge(nonresponder)
ranges.to_csv('gene_ranges.csv', index = False)

df['nt_mean'] = df['gene_nt_array'].apply(lambda x: x.mean())
df['nt_std'] = df['gene_nt_array'].apply(lambda x: x.std())
df['t_mean'] = df['gene_t_array'].apply(lambda x: x.mean())
df['t_std'] = df['gene_t_array'].apply(lambda x: x.std())

for ix, row in df[df['sample_type']=='normal'].iterrows():
    fig, ax = plt.subplots()
    ax.hist(row['gene_nt_array'])
    ax.set_title(row['gene'])
    fig.show()
ranges = df.groupby(['sample_type', 'gene']).mean().reset_index()
normal = ranges[ranges['sample_type']=='normal'][['gene', 'nt_min', 'nt_max']]
normal = normal.rename(columns = {'nt_min': 'normal_min', 'nt_max': 'normal_max'})
responder = ranges[ranges['sample_type']=='responder'][['gene', 'nt_min', 'nt_max', 't_min', 't_max']]
responder = responder.rename(columns = {'nt_min': 'resp_min', 'nt_max': 'resp_max',
                            't_min': 'resp_tumorregion_min', 't_max': 'resp_tumorregion_max'})
nonresponder = ranges[ranges['sample_type']=='nonresponder'][['gene', 'nt_min', 'nt_max', 't_min', 't_max']]
nonresponder = nonresponder.rename(columns = {'nt_min': 'nonresp_min', 'nt_max': 'nonresp_max',
                            't_min': 'nonresp_tumorregion_min', 't_max': 'nonresp_tumorregion_max'})
ranges = normal.merge(responder).merge(nonresponder)
