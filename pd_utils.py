import pandas as pd
import numpy  as np

def csv_to_dataframe(inputname):
    df=pd.read_csv(inputname)
    
    def fix_csv_array(old):
        tmp = old.replace('[','').replace(']','').replace('\n','').split(' ')
        new = [float(x) for x in tmp if x is not '']
        return np.array(new,dtype=np.float64)
    
    for c in df:
        if df[c].dtype=='object':
            df[c] = d.df[c].apply(fix_csv_array)
            
    return df

    
