# Utilities function for statistics computation

import numpy as np

def computeZ0(bkg,sig,lumi,binning,var='ht',mode='complete'):
    '''
    Compute significance using either s/sqrt(b+berr^2) or eq.20 of 
    https://www.pp.rhul.ac.uk/~cowan/stat/medsig/medsigNote.pdf
    
    Args:
     - bkg [dataframe]: total background
     - sig [dataframe]: signal
     - lumi [float]: integrated luminosity to normalize samples
     - binning [array]: final variable binning
     - var [string]: name of the variable to extract the significance
     - mode [string]: 'simple' (s/sqrt(b+berr^2) or 'complete' (eq.20 of the paper)
    
    Return: 
     - var-shaped signifiance [float]
    '''
    
    b,_     = np.histogram(bkg[var], weights=bkg['weight_raw']*lumi     , bins=binning)
    s,_     = np.histogram(sig[var], weights=sig['weight_raw']*lumi     , bins=binning)
    berr2,_ = np.histogram(bkg[var], weights=(bkg['weight_raw']*lumi)**2, bins=binning)
    
    if (mode=='simple'):
        err2 = b+berr2
        sig2 = s**2/(err2)
        sig2[err2<=0]=0.0 # setting the significance to 0 for bins with no background or negative weights
        
    elif (mode=='complete'):
        term1 = (s+b)*np.log((s+b)*(b+berr2)/(b**2+(s+b)*berr2))
        term2 =  b**2/berr2*np.log(1+berr2*s/b/(b+berr2))
        sig2  = 2*(term1-term2)
        
    sig2[np.isnan(sig2)]=0.0
    return np.sqrt(np.sum(sig2))
                                                        
