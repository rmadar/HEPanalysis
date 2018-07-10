import pandas as pd
import numpy  as np

def flatten(column):
    try:
        return np.array([v for e in column for v in e])
    except (TypeError, ValueError):
        return column
            
def match_shape(arr, ref):
    shape = [len(a) for a in ref]
    if len(arr) != np.sum(shape):
        raise ValueError('Incompatible shapes: len(arr) = {}, total elements in ref: {}'.format(len(arr), np.sum(shape)))
    return [arr[ptr:(ptr + nobj)].tolist() for (ptr, nobj) in zip(np.cumsum([0] + shape[:-1]), shape)]    
        
def flat_variable(df, varname, Nelements=10):
    for i in range(0,Nelements):
        bname=varname+str(i)
        df[bname] = df[varname].apply(lambda c: get_value(c,i))
    return

def get_value(x,i):
    try:
        return x[i]
    except IndexError:
        return 0.

def sel_to_var(sel):
    var_array = sel.replace(' ','').split('||')
    return var_array

def csv_to_dataframe(inputname):
    df=pd.read_csv(inputname)
    
    def fix_csv_array(old):
        tmp = old.replace('[','').replace(']','').replace('\n','').split(' ')
        new = [float(x) for x in tmp if x is not '']
        return np.array(new,dtype=np.float64)
    
    for c in df:
        if df[c].dtype=='object':
            df[c] = df[c].apply(fix_csv_array)
            
    return df

    
