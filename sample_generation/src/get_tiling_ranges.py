import pandas as pd


urls = {'nonresp_tumor': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRZhXPauZjX2eeTFy-u0T0fDAuj0NKVfIRHpmCExvEIf9q8CRrun-owAdiwgaUuIEoXIU8ePuz3u558/pub?gid=495047546&single=true&output=csv',
'resp_tumor': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRZhXPauZjX2eeTFy-u0T0fDAuj0NKVfIRHpmCExvEIf9q8CRrun-owAdiwgaUuIEoXIU8ePuz3u558/pub?gid=1172511086&single=true&output=csv',
'resp_touchingtumor': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRZhXPauZjX2eeTFy-u0T0fDAuj0NKVfIRHpmCExvEIf9q8CRrun-owAdiwgaUuIEoXIU8ePuz3u558/pub?gid=2059489507&single=true&output=csv',
'resp_rand': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRZhXPauZjX2eeTFy-u0T0fDAuj0NKVfIRHpmCExvEIf9q8CRrun-owAdiwgaUuIEoXIU8ePuz3u558/pub?gid=2050640054&single=true&output=csv',
'normal':'https://docs.google.com/spreadsheets/d/e/2PACX-1vRZhXPauZjX2eeTFy-u0T0fDAuj0NKVfIRHpmCExvEIf9q8CRrun-owAdiwgaUuIEoXIU8ePuz3u558/pub?gid=1144547897&single=true&output=csv'}

def store_ranges():
    ranges = {}
    for data_type, url in urls.items():
        df = pd.read_csv(url, index_col = 0, encoding = 'utf8')
        df.to_csv('../data/ranges/'+data_type + '.csv')
