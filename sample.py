import os
import datahandler
import pandas as pd

### TO DO ###
# 2. add method to access thet tree (process both with pandas and root)
# 3. improve the string formating function
# 4. improve the management of weights
# 5. migrate the root file loading in this class

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
            self.df = datahandler.get_dataframe(self.dsid_array)
        self.Nentries = len(self.df)
    
    def __str__(self):
        return ('{:>'+str(10)+'}: {:>'+str(10)+'.0f} entries,'+(' '*5)+'color={}').format(self.name,self.Nentries,self.color)

    def __copy__(self,s):
        return sample(s.dsid_array,s.name,s.latexname,s.color,s.weight,s.proctype,s.df)

    def get_files_list(self):
        if (os.path.isdir('data')): return ['data/'+str(ids)+'.root' for ids in self.dsid_array ]
        else: print('Sample::get_file_list():: ERROR, data directory is not found')
    
    def apply_selection(self,selection):
        copy          = self.__copy__(self)
        copy.df       = copy.df.query(selection)
        copy.Nentries = len(copy.df)
        return copy

    def add_observable(self,formula,name):
        self.df[name] = formula(self.df)

    def get_dataframe(self):
        #ss = stack_arrays( [root2array(thisfile, tree_name, **kwargs).view(np.recarray) for thisfile in self.filelist] )
        #try:                 return pd.DataFrame(ss)
        #except Exception, e: return pd.DataFrame(ss.data)
                

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



        
        return
