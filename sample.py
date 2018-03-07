# Usual tools
import os
import pandas as pd
import numpy  as np

# ROOT tools
import ROOT
from   root_numpy import root2array


### TO DO ###
# 1. add method to access the tree (process both with pandas and root)
# 2. improve the string formating function
# 3. improve the management of weights

class sample:
    
    def __init__(self,dsid_array,name,latexname,color,weight,proctype,df=pd.DataFrame()):
        self.dsid_array  = dsid_array
        self.filelist    = self.get_files_list()
        self.name        = name
        self.latexname   = latexname
        self.color       = color
        self.weight      = weight
        self.proctype    = proctype
        self.tree        = 0
        self.df          = df
        if (self.df.empty):
            self.df = self.get_dataframe()
        self.Nentries = len(self.df)
    
    def __str__(self):
        return ('{:>'+str(10)+'}: {:>'+str(10)+'.0f} entries,'+(' '*5)+'color={}').format(self.name,self.Nentries,self.color)

    def __copy__(self,s):
        return sample(s.dsid_array,s.name,s.latexname,s.color,s.weight,s.proctype,s.df)

    def get_files_list(self):
        # To be tunable from outside
        dir_path='/home/rmadar/Documents/work/ATLAS/4topSM/general-studies-4topSM/data/'
        #---------------------------
        if (os.path.isdir(dir_path)): return [dir_path+str(ids)+'.root' for ids in self.dsid_array ]
        else: print('Sample::get_file_list():: ERROR, data directory is not found')
    
    def apply_selection(self,selection):
        copy          = self.__copy__(self)
        copy.df       = copy.df.query(selection)
        copy.Nentries = len(copy.df)
        return copy

    def add_observable(self,formula,name):
        self.df[name] = formula(self.df)


    def get_dataframe(self):

        # Function in function --> good practice?
        #---------------------------------------
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
        #---------------------------------------
        

        # this part should be tunable from outside the class - need to think how to do it properly
        #-----------------------------------------------------------
        weightBranches = ['weight_mc','weight_pileup','weight_leptonSF_tightLeps','weight_bTagSF_77','weight_jvt']
        UsedBranches   = ['jet_pt','jet_mv2c20','jet_mv2c10','lep_pt','ht','met_met']+weightBranches
        Usedselections =  'SSee_2015 || SSee_2016 || SSem_2015 || SSem_2016 || SSmm_2015 || SSmm_2016 ||'
        Usedselections += 'eee_2015  || eee_2016  || eem_2015  || eem_2016  || emm_2015  || emm_2016 || mmm_2015 || mmm_2016'
        UsedBranches   += Usedselections.replace(' ','').split('||')
        #-----------------------------------------------------------
        
        data_array = []
        for fname in self.filelist:

            # load data
            thisdata = pd.DataFrame(root2array(fname, 'nominal_Loose', branches=UsedBranches, selection=Usedselections).view(np.recarray))
            if(thisdata.empty): continue

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

            # append this dataset
            data_array.append(thisdata)
                
        return pd.concat(data_array)


