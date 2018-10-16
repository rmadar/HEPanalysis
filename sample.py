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


# =======================================
#  Viability of this class to be decided
# =======================================
# 
# + working with pandas dataframe is too time/memory consuming
# + plots are not as tuned as in ROOT (cf. hepplotting package)
# + could be more relevant to use
#   - TTrees
#   - metadata (colors - both ROOT and MPL, names, legends, groups)
#   - hepplotting for data/MC comparisons
#   - aux plotting (eg. ROC curves)
#   - pandas dataframe conversion for a subset of flat variables
#
# =======================================


class sample:
    def __init__(self,config,dsid_array,name,latexname,color,df=pd.DataFrame()):
        '''
        config [dictionnary {property:info}] with property=[path,selection,branches,weight]
        '''
        self.config      = config
        self.dsid_array  = dsid_array
        self.files_dict  = self.get_files_dict()
        self.Nraw_dict   = self.get_Ngen_dict()
        self.name        = name
        self.latexname   = latexname
        self.color       = color
        self.tree        = self.get_tree()
        self.df          = df
        if self.df.empty:
            self.df = self.get_dataframe()
        self.Nentries   = self.tree.GetEntries()

        
    def __str__(self):
        return ('{:>'+str(10)+'}: {:>'+str(10)+'.0f} entries,'+(' '*5)+'color={}').format(self.name,self.Nentries,self.color)

    def __copy__(self,s):
        return sample(s.config,s.dsid_array,s.name,s.latexname,s.color,s.df)

    
    def get_files_dict(self):
        
        def find_all_files(path,dsid):
            import os
            res=[]
            for dirpath, dirnames, files in os.walk(path):
                for name in files:
                    fname=os.path.join(dirpath,name)
                    if '.root' in fname and str(dsid) in fname:
                        res.append(fname)
            return res
        
        if (os.path.isdir(self.config['path'])):
            output_dict={}
            for dsid in self.dsid_array:
                output_dict[dsid] = find_all_files(self.config['path'],dsid)
            return output_dict
        else: 
            err='Sample::get_file_dict():: ERROR, data directory {} is not found'.format(self.config['path'])
            raise NameError(err)
    
    
    def apply_selection(self,selection):
        copy          = self.__copy__(self)
        copy.df       = copy.df.query(selection)
        copy.Nentries = len(copy.df)
        return copy


    def add_observable(self,formula,name):
        self.df[name] = formula(self.df)
        

    def get_Ngen_dict(self):
        wghts,Ngen=['totalEvents','totalEventsWeighted'],{}
        for dsid,fname_list in self.files_dict.items():
            weight_dict,totalEvents,totalEventsWeighted={},0,0
            for fname in fname_list:
                f=ROOT.TFile.Open(fname)
                if (hasattr(f,'sumWeights')):
                    d=pd.DataFrame(root2array(fname,'sumWeights',branches=wghts).view(np.recarray))
                    totalEvents+=np.sum(d['totalEvents'].values)
                    totalEventsWeighted+=np.sum(d['totalEventsWeighted'].values)
                    weight_dict['totalEvents']=totalEvents
                    weight_dict['totalEventsWeighted']=totalEventsWeighted
                else:
                    weight_dict['totalEvents']=None
                    weight_dict['totalEventsWeighted']=None
            Ngen[dsid]=weight_dict
        return Ngen


    def get_tree(self):
        chain = ROOT.TChain('nominal_Loose')
        for dsid,fname_list in self.files_dict.items():
            for fname in fname_list:
                chain.Add(fname)
        return chain    
    
    def get_dataframe(self):
        weightBranches,varBranches,usedSelections=None,None,None
        if 'wght_branches' in self.config: weightBranches = self.config['wght_branches']
        if 'var_branches'  in self.config: varBranches    = self.config['var_branches']
        if 'selection'     in self.config: usedSelections = self.config['selection']
        if varBranches==None   : varBranches=[]
        if weightBranches==None: weightBranches=[]
        if usedSelections==None: usedSelections=''
        usedBranches = varBranches+weightBranches+pdu.sel_to_var(usedSelections)
        data_array = []
        for dsid,fname_list in self.files_dict.items():
            
            for fname in fname_list:

                f=ROOT.TFile.Open(fname)
                if f.nominal_Loose.GetEntries()==0:
                    continue
        
                # load data
                if usedBranches==[''] : usedBranches=None
                if usedSelections=='': usedSelections=None
                thisdata = pd.DataFrame(root2array(fname,'nominal_Loose', 
                                                   branches=usedBranches, 
                                                   selection=usedSelections).view(np.recarray))
                
                # add xsec weights
                if 'add_weight' in self.config:
                    add_weight = self.config['add_weight']
                    if add_weight:
                        add_weight(sample=self,
                                   root_file_name=fname,
                                   dsid=dsid,
                                   dataframe=thisdata,
                                   weightslist=weightBranches,
                        )
                
                # flat arrays for variable arrays branches
                if 'var_flat' in self.config:
                    for v,n in self.config['var_flat']:
                        pdu.flat_variable(thisdata,v,n)

                data_array.append(thisdata)

        final_df=pd.DataFrame()
        if data_array:
            final_df=pd.concat(data_array)
        
        return final_df


