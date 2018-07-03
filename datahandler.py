# Common tools
import glob
import os

# ROOT tools
import ROOT
from   root_numpy import root2array

# Python tools
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import matplotlib        as mpl

# Machine learning tools
from sklearn.metrics import roc_curve


def root2panda(files_path, tree_name, mask = False, **kwargs):
    files = glob.glob(files_path)
    if (tree_name == ''): ss = stack_arrays([root2array(fpath, **kwargs).view(np.recarray) for fpath in files])
    else: ss = stack_arrays([root2array(fpath, tree_name, **kwargs).view(np.recarray) for fpath in files])
    if (mask): return pd.DataFrame(ss.data)
    else:
        try: return pd.DataFrame(ss)
        except Exception, e: return pd.DataFrame(ss.data)

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

def get_dataframe(listOfDISD, listOfBranch=['jet_pt','jet_mv2c20','lep_pt','ht','met_met']):
    """
    Load a list of DSID, add proper weight variable to each dataframe and return it.
    """
    filename_list=[]
    if (os.path.isdir('data')):
        for ids in listOfDISD: filename_list.append( 'data/'+str(ids)+'.root' )
    else:
        print('data file not found')

    weightBranches = ['weight_mc','weight_pileup','weight_leptonSF_tightLeps','weight_bTagSF_77','weight_jvt']
    UsedBranches   = listOfBranch+weightBranches
    Usedselections =  'SSee_2015 || SSee_2016 || SSem_2015 || SSem_2016 || SSmm_2015 || SSmm_2016 ||'
    Usedselections += 'eee_2015  || eee_2016  || eem_2015  || eem_2016  || emm_2015  || emm_2016 || mmm_2015 || mmm_2016'
    UsedBranches   += Usedselections.replace(' ','').split('||')    
    data = []
    for fname in filename_list:

        # load data
        thisdata = pd.DataFrame( root2array(fname, 'nominal_Loose', branches=UsedBranches, selection=Usedselections).view(np.recarray) )
        if (thisdata.empty): continue
        
        # add xsec weights
        rootfile = ROOT.TFile(fname)
        w_xsec   = 1.0/rootfile.Get('hIntLum').GetBinContent(1)
        thisdata['weight'] = w_xsec
        for wname in weightBranches: thisdata['weight'] *= thisdata[wname]
            
        # flat arrays for non e
        for var in thisdata.columns.tolist():
            try:
                if ( type(thisdata[var][0]) is np.ndarray ):
                    flat_variable(thisdata, var)
            except IndexError: print 'IndexError, I am not sure why'
        data.append(thisdata)
        
    return pd.concat(data)

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

