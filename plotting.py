# Some utils function for HEP plotting
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt


def compare_distributions(data_proc_array, variables, selection='', myfigsize=(20,20)):
    '''
    Plot normalized distributions for two processes, for a given selection (9 variables max)
    variables: array of variable name (9 max)
    selection: string of event selection
    myfigsize: figure size in case of less than 9 variables
    '''
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
    '''
    Xsig      : [dataframe, 'sig eff label'] object containing signal events
    Xbkg      : [dataframe, 'bkg eff label'] object containing background events
    variables : array of variable name that will be use to plot ROC curve
    regressors: array of [reg_method,name] where reg_method is regression and name is its legend name
    '''
    
    from sklearn.metrics import roc_curve

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


def plot(sample_dict,v,sel='',mode='stack',lumi=80.0):
    '''
    DocString to be filled
    '''
    
    if (sel): selected_data = {k:s.apply_selection(sel) for k,s in sample_dict.items()}
    else    : selected_data = sample_dict

    # ordering dictionary
    from collections import OrderedDict
    ordered_data = OrderedDict()
    if ('others' in selected_data.keys()): ordered_data['others'] = selected_data['others']
    if ('ttW'    in selected_data.keys()): ordered_data['ttW']    = selected_data['ttW']
    if ('ttZ'    in selected_data.keys()): ordered_data['ttZ']    = selected_data['ttZ']
    if ('ttH'    in selected_data.keys()): ordered_data['ttH']    = selected_data['ttH']
    if ('ttbar'  in selected_data.keys()): ordered_data['ttbar']  = selected_data['ttbar']
    if ('4topSM' in selected_data.keys()): ordered_data['4topSM'] = selected_data['4topSM']
    
    colors  = [d.color                 for d in ordered_data.values()]
    labels  = [d.latexname             for d in ordered_data.values()]
    weights = [d.df['weight_raw']*lumi for d in ordered_data.values()]
    values  = [d.df[v.varname]         for d in ordered_data.values()]
    
    plt.figure()

    if (mode is 'stack'):
        _,barray,_ = plt.hist(values, color=colors, label=labels, \
                              weights=weights, stacked=True, \
                              bins=v.binning, log=v.logy)
        total  = pd.concat([bkg.df for k,bkg in ordered_data.items() if k is not '4topSM'])
        tot,_  = np.histogram(total[v.varname], weights=(total['weight_raw']*lumi)   , bins=v.binning)
        err2,_ = np.histogram(total[v.varname], weights=(total['weight_raw']*lumi)**2, bins=v.binning)
        err = np.sqrt(err2)
        em=[[n-e,n-e] for n,e in zip(tot,err)]
        ep=[[n+e,n+e] for n,e in zip(tot,err)]
        rec=[[barray[i],barray[i+1]] for i in range(0,len(barray)-1)]
        plt.errorbar([barray[i]+1*(barray[i+1]-barray[i])/2. for i in range(0,len(barray)-1)],tot,np.sqrt(tot),\
                     fmt='.',color='k', alpha=1.0, markersize=15, label='$B \pm \sqrt{B}$', zorder=3)
        plt.fill_between([r for re in rec for r in re],\
                         [r for re in ep  for r in re],\
                         [r for re in em  for r in re],\
                         color='linen',alpha=0.8,label='MC Stat. Unc.',zorder=2)
        
    elif (mode is 'normalized'):
        for bkg in ordered_data.values():
            Ntot=np.sum(bkg.df['weight_raw']*lumi)
            varray,barray,_ = plt.hist(bkg.df[v.varname], color=bkg.color, label=bkg.latexname,\
                                       weights=bkg.df['weight_raw']*lumi/Ntot, \
                                       bins=v.binning , log=v.logy,\
                                       histtype='step', linewidth=2.5)
            err2,_ = np.histogram(bkg.df[v.varname], \
                                  weights=((bkg.df['weight_raw']*lumi)**2), \
                                  bins=v.binning)
            plt.errorbar([barray[i]+(barray[i+1]-barray[i])/2. for i in range(0,len(barray)-1)],\
                         varray,np.sqrt(err2)/Ntot,fmt='.',color=bkg.color, alpha=0.7, markersize=10)

    else:
        print('Plot style \'' + mode+ '\' is not accepted')


    plt.ylabel(v.ylabel)
    plt.xlabel(v.xlabel)
    plt.legend()
    plt.draw()
    plt.title('ATLAS Simulation  $\sqrt{s}=13\,$TeV, $'+'{:.0f}'.format(lumi)+'\,$fb$^{-1}$')
    plt.savefig(v.varname+'_'+sel.replace('==','eq').replace('>','gt')+'.png',dpi=250)
    
    return

