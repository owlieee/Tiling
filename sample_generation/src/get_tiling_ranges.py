import pandas as pd

urls = {'changes': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQP0fedcuCqCXteKZ-tbpYUdrUxKIACz7VZB-f6N7vIdG6vD-6BkHRMLCd5OX5MPcuR2kwjQOdDwtVi/pub?gid=0&single=true&output=csv',
        'normal': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQP0fedcuCqCXteKZ-tbpYUdrUxKIACz7VZB-f6N7vIdG6vD-6BkHRMLCd5OX5MPcuR2kwjQOdDwtVi/pub?gid=207218135&single=true&output=csv'}

def store_ranges():
    ranges = {}
    for data_type, url in urls.items():
        df = pd.read_csv(url, index_col = 0, encoding = 'utf8')
        df.to_csv('../data/ranges/'+data_type + '.csv')

if __name__ == '__main__':
    store_ranges()
