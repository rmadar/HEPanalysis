# Usual tools
import os
import pandas as pd
import numpy  as np

# ROOT tools
import ROOT
from   root_numpy import root2array

# My tools
import pd_utils as pdu

# Disable warnings
import warnings
warnings.filterwarnings('ignore')

### TO DO ###
# 1. add method to access the tree (process both with pandas and root)
# 2. improve the string formating function


class sample:
    def __init__(self,config,dsid_array,name,latexname,color,df=pd.DataFrame()):
        '''
        config [dictionnary {property:info}] with property=[path,selection,branches,weight]
        '''
        self.config      = config
        self.dsid_array  = dsid_array
        self.filelist    = self.get_files_list()
        self.name        = name
        self.latexname   = latexname
        self.color       = color
        self.tree        = 0
        self.df          = df
        if (self.df.empty):
            self.df = self.get_dataframe()
        self.Nentries = len(self.df)
    
    def __str__(self):
        return ('{:>'+str(10)+'}: {:>'+str(10)+'.0f} entries,'+(' '*5)+'color={}').format(self.name,self.Nentries,self.color)

    def __copy__(self,s):
        return sample(s.config,s.dsid_array,s.name,s.latexname,s.color,s.df)

    def get_files_list(self):
        if (os.path.isdir(self.config['path'])): return [self.config['path']+str(ids)+'.root' for ids in self.dsid_array ]
        else: print('Sample::get_file_list():: ERROR, data directory is not found')
    
    def apply_selection(self,selection):
        copy          = self.__copy__(self)
        copy.df       = copy.df.query(selection)
        copy.Nentries = len(copy.df)
        return copy

    def add_observable(self,formula,name):
        self.df[name] = formula(self.df)
        
    def get_Ngen_dict(self):
        wghts,Ngen=['totalEvents','totalEventsWeighted'],{}
        for ids in self.dsid_array:
            d=pd.DataFrame(root2array(self.config['path']+str(ids)+'.root', 'sumWeights', branches=wghts).view(np.recarray))
            Ngen[ids]=[np.sum(d[w].values) for w in wghts]
        return Ngen

    def get_dataframe(self):
        weightBranches = self.config['wght_branches']
        varBranches    = self.config['var_branches']
        usedselections = self.config['selection']
        usedBranches   = varBranches+weightBranches+sel_to_var(usedselections)          

        data_array = []
        for fname in self.filelist:

            # load data
            thisdata = pd.DataFrame(root2array(fname, 'nominal_Loose', branches=usedBranches, selection=usedselections).view(np.recarray))
            if(thisdata.empty): continue

            # add xsec weights
            get_weight = self.config['getweight']
            get_weight(ROOT.TFile.Open(fname),thisdata,weightBranches)

            # flat arrays for variable arrays branches
            if 'var_flat' in self.config:
                for v,n in self.config['var_flat']:
                    pdu.flat_variable(thisdata,v,n)
            else:
                for v in thisdata.columns.tolist():
                    try:
                        if ( type(thisdata[var][0]) is np.ndarray ):
                            pdu.flat_variable(thisdata, var)
                    except IndexError: print 'IndexError, I am not sure why'

            # append this dataset
            data_array.append(thisdata)
                
        return pd.concat(data_array)


