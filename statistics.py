# Utilities function for statistics computation

import numpy as np

def getZ0(s,b,berr2=None,mode='complete'):
    '''
    Compute significance from using either s/sqrt(b+berr^2) or eq.20 of 
    https://www.pp.rhul.ac.uk/~cowan/stat/medsig/medsigNote.pdf
    
    Args:
     - b     [np.array]: total background
     - s     [np.array]: signal
     - berr2 [np.array]: integrated luminosity to normalize samples
     - mode  [string]  : 'simple' s/sqrt(b), 
                         'simple_MCstat' (s/sqrt(b+berr^2) 
                         'complete' (eq.20 of the paper)
    Return z0,nfail: 
     - z0 [float] var-shaped signifiance
     - nfail [int] number of bins where sig2 is negative or nan
    '''
    if (mode=='simple'):
        err2 = b
        sig2 = s**2/(err2)

    elif (mode=='simple_MCstat'):
        err2 = b+berr2
        sig2 = s**2/(err2)

    elif (mode=='complete'):
        term1 = (s+b)*np.log((s+b)*(b+berr2)/(b**2+(s+b)*berr2))
        term2 =  b**2/berr2*np.log(1+berr2*s/b/(b+berr2))
        sig2  = 2*(term1-term2)

    else:
        raise NameError('computeZ0()::Wrong mode')
    
    nfail=np.sum(np.isnan(sig2)) + np.sum(sig2<0) + np.sum(np.isinf(sig2))
    sig2[np.isnan(sig2)]=0.0
    sig2[np.isinf(sig2)]=0.0
    sig2[sig2<0]=0.0
    return np.sqrt(np.sum(sig2)),nfail
    

def computeZ0(bkg,sig,lumi,binning,var='HT',mode='complete'):
    '''
    Compute significance as defined in getZ0() from any variable 'var'
    contained in the dataframes 'bkg' and 'sig', for an arbitrary binning.
    and integrated luminosity.
    
    Args:
     - bkg [dataframe]: total background
     - sig [dataframe]: signal
     - lumi [float]: integrated luminosity to normalize samples
     - binning [array]: final variable binning
     - var [string]: name of the variable to extract the significance
     - mode [string]: 'simple' (s/sqrt(b+berr^2) or 'complete' (eq.20 of the paper)
    
    Return getZ0(): 
     - z0 [float] var-shaped signifiance
     - nfail [int] number of bins where sig2 is negative or nan
    '''

    bkg_v,sig_v=bkg[var].values,sig[var].values
    bkg_w,sig_w=bkg['weight_raw'].values,sig['weight_raw'].values
    
    b,_     = np.histogram(bkg_v, weights=bkg_w*lumi     , bins=binning)
    s,_     = np.histogram(sig_v, weights=sig_w*lumi     , bins=binning)
    berr2,_ = np.histogram(bkg_v, weights=(bkg_w*lumi)**2, bins=binning)
    
    return getZ0(s,b,berr2=berr2,mode=mode)
                                                            
