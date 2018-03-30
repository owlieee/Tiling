import pandas as pd
import numpy as np


genes = ['ABL1', 'AKT1', 'ALK', 'APC', 'ATM', 'BRAF', 'CDH1', 'CDKN2A',
       'CSF1R', 'CTNNB1', 'EGFR', 'ERBB2', 'ERBB4', 'FBXW7', 'FGFR1',
       'FGFR2', 'FGFR3', 'FLT3', 'GNA11', 'GNAQ', 'GNAS', 'HNF1A', 'HRAS',
       'IDH1', 'JAK2', 'JAK3', 'KDR', 'KIT', 'KRAS', 'MET', 'MLH1', 'MPL',
       'NOTCH1', 'NPM1', 'NRAS', 'PDGFRA', 'PIK3CA', 'PTEN', 'PTPN11',
       'RB1', 'RET', 'SMAD4', 'SMARCB1', 'SMO', 'SRC', 'STK11', 'TP53',
       'VHL']
types = ['normal', 'nonresponder', 'responder']
responder_t_changes = {'CDKN2A', 'NOTCH', 'EGFR', 'IDH1', 'KRAS', 'SMARCB1', 'APC', 'GNA11', 'PIK3CA'}
responder_touch_t_changes = {'HRAS', 'SMO'}
responder_rand_changes = {'ABL1', 'JAK3'}
nonresponder_t_changes = {'CDKN2A', 'EGFR', 'IDH1', 'KRAS', 'SMARCB1', 'APC', 'PIK3CA'}
normal_ranges = pd.read_csv('data/normal_ranges.csv')

def generate_normal(low, high):
    normal = np.random.uniform(low = low, high = high, size = (9, 10))
    return normal

def generate_tumor(p_norm = 0.5):
    tumordata = {}
    choice_x = np.random.random()
    if choice_x > 1-np.sqrt(p_norm):
        x = np.random.randn(90)
        tumordata['x'] = 'normal'
    else:
        x = np.random.rand(90)
        tumordata['x'] = 'uniform'
    choice_y = np.random.random()
    if choice_y > 1-np.sqrt(p_norm):
        y = np.random.randn(90)
        tumordata['y'] = 'normal'
    else:
        y = np.random.rand(90)
        tumordata['y'] = 'uniform'
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=(9,10))
    tumordata['thresh'] = np.mean(heatmap)
    tumordata['tumor'] = (heatmap>np.mean(heatmap)).astype(int)
    tumordata['size'] = tumordata['tumor'].sum()
    tumordata['edges'] =get_touching_tumor(tumordata['tumor'])
    return tumordata

def get_touching_tumor(g):
    t = np.where(g==1)
    t_coord = set(zip(t[0], t[1]))
    rowBound, colBound = (g.shape[0] - 1, g.shape[1] - 1)
    rowShifts = [t[0], np.clip(t[0]+1, 0, rowBound), np.clip(t[0]-1, 0, rowBound)]
    colShifts = [t[1], np.clip(t[1]+1, 0, colBound), np.clip(t[1]-1, 0, colBound)]
    t_edge = set()
    for row in rowShifts:
        for col in colShifts:
            t_edge.update(zip(row, col))
    t_edge = zip(*list(t_edge-t_coord))
    touch_array = np.zeros(g.shape)
    touch_array[t_edge] = 1
    return touch_array

for ix, row in normal_ranges.iterrows():
    normal = generate_normal(row['normal_min'], row['normal_max'])
