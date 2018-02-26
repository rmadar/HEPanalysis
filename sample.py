import datahandler
import pandas as pd

### TO DO ###
# 1. add method to get the list of files
# 2. add method to access thet tree (process both with pandas and root)
# 3. improve the string formating function
# 4. improve the management of weights
# 5. migrate the root file loading in this class

class sample:
    
    def __init__(self,dsid_array,name,latexname,color,weight,proctype,df=pd.DataFrame()):
        self.dsid_array  = dsid_array
        self.filelist    = []
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

    
    def apply_selection(self,selection):
        copy          = self.__copy__(self)
        copy.df       = copy.df.query(selection)
        copy.Nentries = len(copy.df)
        return copy

    def add_observable(self,formula,name):
        self.df[name] = formula(self.df)

    def get_dataframe(self):
        return
