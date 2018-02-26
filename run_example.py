# Usual packages
import numpy             as np
import matplotlib        as mpl
import matplotlib.pyplot as plt

# Plotting
from rootpy.plotting.style import set_style
set_style('ATLAS', mpl=True)
mpl.rcParams['axes.titlesize']  = 20
mpl.rcParams['axes.labelsize']  = 24
mpl.rcParams['lines.linewidth'] = 2.5

# My packages
import sample   as spl
import variable as var

# Investigate timing
from timeit import default_timer
t0 = default_timer()

# Load my data
#-------------
sttW = spl.sample([410155]                            ,'ttW'   , r'$t\bar{t}W$'       ,'orange','weight','bkg')
sttZ = spl.sample([410218,410219,410220,410156,410157],'ttZ'   , r'$t\bar{t}Z$'       ,'grey'  ,'weight','bkg')
stt  = spl.sample([410000]                            ,'ttbar' , r'$t\bar{t}$'        ,'blue'  ,'weight','bkg')
s4t  = spl.sample([410080]                            ,'4topSM', r'$t\bar{t}t\bar{t}$','red'   ,'weight','sig')
data = {
    s4t.name : s4t ,
    stt.name : stt ,
    sttW.name: sttW,
    sttZ.name: sttZ,
}
t1   = default_timer()
print ('  Loading done in {:.2f}s for {} events\n'.format( t1-t0 , np.sum([d.Nentries for d in data.values()]) ) )



#-----------------------------------
# Add as many observable you need to
#-----------------------------------
def count_obj(df,obj):
    if  (obj is 'jet') : return df['jet_pt'].apply( lambda x: len(x))
    elif(obj is 'lep') : return df['jet_pt'].apply( lambda x: len(x))
    elif(obj is 'bjet'): return df['jet_mv2c20'].apply( lambda x: np.sum(x>0.645925))
    else: print('ERROR::count_obj():: Object \'{}\' is not supported'.format(obj))
    
def isChannel(df,channel):
    try:    return np.where( df[channel+'_2015']+df[channel+'_2016']>0, True, False )
    except: print('ERROR::isChannel():: Something went wrong with requesting channel \'{}\''.format(channel))

def ChannelIndex(df):
    return df['SSee']*1 + df['SSem']*2 + df['SSmm']*3 + df['eee']*4 + df['eem']*5 + df['emm']*6 + df['mmm']*7

for d in data.values():
    d.add_observable(lambda df: df['ht']/1000.      ,  'HT'  )
    d.add_observable(lambda df: df['met_met']/1000. , 'MET'  )
    d.add_observable(lambda df: count_obj(df,'jet' ), 'njet' )
    d.add_observable(lambda df: count_obj(df,'lep' ), 'nlep' )
    d.add_observable(lambda df: count_obj(df,'bjet'), 'nbjet')
    d.add_observable(lambda df: isChannel(df,'SSee'), 'SSee' )
    d.add_observable(lambda df: isChannel(df,'SSem'), 'SSem' )
    d.add_observable(lambda df: isChannel(df,'SSmm'), 'SSmm' )
    d.add_observable(lambda df: isChannel(df,'eee' ), 'eee'  )
    d.add_observable(lambda df: isChannel(df,'eem' ), 'eem'  )
    d.add_observable(lambda df: isChannel(df,'emm' ), 'emm'  )
    d.add_observable(lambda df: isChannel(df,'mmm' ), 'mmm'  )
    d.add_observable(lambda df: ChannelIndex(df)    , 'ChIn' )

t2 = default_timer()
print '   Variables added in {:.2f}s\n'.format(t2-t1)
#---------------------------------



#-----------------------------
# Plot some nice distributions
#-----------------------------
def plot(var_obj_array,sel=''):
    if (sel): selected_data = {k:s.apply_selection(sel) for k,s in data.items()}
    colors  = [d.color               for d in selected_data.values()]
    labels  = [d.latexname           for d in selected_data.values()]
    weights = [d.df['weight']*120    for d in selected_data.values()]
    for v in var_obj_array:
        values  = [d.df[v.varname] for d in selected_data.values()]
        plt.figure()
        plt.hist(values, color=colors, label=labels, weights=weights, stacked=True, bins=v.binning, log=v.logy)
        plt.ylabel(v.ylabel)
        plt.xlabel(v.xlabel)
        plt.legend()
        plt.draw()
        plt.savefig(v.varname+'_'+sel.replace('==','eq').replace('>','gt')+'.png',dpi=250)
    return


var_HT    = var.variable('HT'     ,[0,250,500,750,1000,1500,3000,10000],r'$H_T$ [GeV]'       ,'Entries',True)
var_MET   = var.variable('MET'    ,[0,250,500,750,1000,1500,3000,10000],r'$E^{miss}_T$ [GeV]','Entries',True)
var_NJET  = var.variable('njet'   ,np.linspace(0.5,20.5,21)            ,r'$N_{jets}$'        ,'Entries',True)
var_NBJET = var.variable('nbjet'  ,np.linspace(0.5,10.5,11)            ,r'$N_{b-jets}$'      ,'Entries',True)
var_SRi   = var.variable('ChIn'   ,np.linspace(0.5,7.5,8)              ,r'Channel index'     ,'Entries',True)
var_array = [var_HT,var_MET,var_NJET,var_NBJET,var_SRi]

plot(var_array,'ht>500e3')
plot(var_array,'nbjet==4')

print '   --> All happened in {:.2f}s\n'.format(tend-t0)




