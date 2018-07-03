# Some utils function for HEP studies

import pandas            as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def compare_distributions(data_proc_array, variables, selection='', myfigsize=(20,20)):
    """
    Plot normalized distributions for two processes, for a given selection (9 variables max)
    variables: array of variable name (9 max)
    selection: string of event selection
    myfigsize: figure size in case of less than 9 variables
    """
    if (selection is not ''):
        for data,_ in data_proc_array:
            data = data.query(selection)

    plt.figure(figsize=myfigsize)
    for i,var in enumerate(variables):
        if (i>9):
            break
        plt.subplot(3,3,i)

        for data,label in data_proc_array:
            plt.hist(data[var], bins=50, histtype='step', density=True, linewidth=2.5, label=label, weights=data['weight'])
        plt.legend()
        plt.xlabel(var)

    plt.tight_layout()    
    return


def plot_roc_curves(Xsig, Xbkg, variables, regressors, selections=''):
    """
    Xsig      : [dataframe, 'sig eff label'] object containing signal events
    Xbkg      : [dataframe, 'bkg eff label'] object containing background events
    variables : array of variable name that will be use to plot ROC curve
    regressors: array of [reg_method,name] where reg_method is regression and name is its legend name
    """
    
    # Prepare the full dataset
    sig_labelled = Xsig[0]
    sig_labelled['isSig'] = True
    bkg_labelled = Xbkg[0]
    bkg_labelled['isSig'] = False
    X = pd.concat( [sig_labelled,bkg_labelled] )
    if selections: X = X.query(selections)
    
    # Produce the plots
    plt.figure(figsize=(10,8))
    for var in variables:
        fake,eff,_= roc_curve(X['isSig'],X[var])
        plt.plot(fake,eff,label=var)
    
    Xeval=X.drop('isSig',axis=1)
    Xeval.head()
    for reg,name in regressors:
        fake,eff,_= roc_curve(X['isSig'],reg.predict(Xeval))
        plt.plot(fake,eff,label=name)
    
    plt.xlabel(Xbkg[1])
    plt.ylabel(Xsig[1])
    plt.legend()
    return
