import numpy as np
import itertools
import copy

def count_nonnan(a,axis=-1):
    '''
    Count the number of elements failing np.isnan() along
    a given axis.
    '''
    if   axis==-1: ntot=a.size
    else         : ntot=a.shape[axis]
    return ntot-np.count_nonzero(np.isnan(a),axis=axis)

def replace_nan(a,value=0):
    '''
    Replace all np.nan from a by value (0 by default) and return 
    a copy of the initial array.
    '''
    output=copy.copy(a)
    output[np.isnan(output)]=value
    return output

def replace_val(a,value_old,value_new=0):
    '''
    Replace all value_old from a by value_new (0 by default) and return a copy 
    of the initial array.
    '''
    output=copy.copy(a)
    output[output==value_old]=value_new
    return output  

def contains_collections(arrays):
    '''
    Return True if at least one of the dimension
    of the sub-array in arrays is at least 1.
    '''
    dims=np.array([e.ndim for e in arrays])
    return np.count_nonzero(dims>=1)>0


def get_indexed_value(a,index):
    '''
    Give the an array of indexed values, with an event-dependent index.

    This function takes an array of shape (Nevts,Nobj) and returns an 
    array of shape (Nevts,). Each element corresponds the value of the 
    ith object, where i is different for each event and are regrouped 
    in the array 'index'. For e.g., of one wants the lepton isolation
    for the lepton having the highest eta:
     - lep_iso, shape=(Nevts,Nlep)
     - index=np.argmax(np.abs(lep_eta),axis=1), shape=(Nevts,)
     - iso_max_eta=get_indexed_value(lep_iso,index), shape=(Nevts,)
    
   
    Parameters
    ----------
    a: np.array
        The shape of this array must be (Nevts,Nobj)
    index: np.array
        The shape of this array must be (Nevts,)
    
    
    Returns
    -------
    out: np.ndarray
        The shape of the array is (Nevts,)
             
    
    Examples
    --------
    >>> import numpy as np
    >>> a=np.arange(6).reshape(2,3)
    >>> a
    >>> array([[0, 1, 2],
               [3, 4, 5]])
    >>>
    >>> get_indexed_value(a,index=[0,1])
    >>> array([0, 4])
    '''
    
    # Make sure we manipulate numpy arrays
    a,index=np.array(a),np.array(index)
    
    # Sanity checks
    if a.ndim!=2 or index.ndim!=1:
        err  = 'This function requires an array \'a\' of dimension 2 (it is currently {})\n'.format(a.ndim)
        err += 'and an array \'index\' of dimension 1 (it is currently {})'.format(index.ndim)
        raise NameError(err)
    if a.shape[0] != index.shape[0]:
        err  = 'The two array must have the same number of element along the first axis.\n'
        err += 'while currently \'a\' has {} elements and \'index\' has {}.'.format(a.shape[0],index.shape[0])
        raise NameError(err)
    
    # Actuall work
    N=np.arange(a.shape[0])
    return np.array( [a[i,index[i]] for i in N] )


def get_all_but_indexed_value(a,index):
    '''
    Give the an array of all values but the indexed ones, with 
    an event-dependent index.

    This function takes an array of shape (Nevts,Nobj) and returns an 
    array of shape (Nevts,Nobj-1). Each element corresponds the value of the 
    all objects but the ith, where i is different for each event and are 
    regrouped in the array 'index'. For e.g., of one wants the lepton isolation
    for the all leptons but the one with the highest eta:
     - lep_iso, shape=(Nevts,Nlep)
     - index=np.argmax(np.abs(lep_eta),axis=1), shape=(Nevts,)
     - iso_other_eta=get_all_but_indexed_value(lep_iso,index), shape=(Nevts,Nlep-1)
    
   
    Parameters
    ----------
    a: np.array
        The shape of this array must be (Nevts,Nobj)
    index: np.array
        The shape of this array must be (Nevts,)
    
    
    Returns
    -------
    out: np.ndarray
        The shape of the array is (Nevts,Nobj-1)
             
    
    Examples
    --------
    >>> import numpy as np
    >>> a=np.arange(6).reshape(2,3)
    >>> a
    >>> array([[0, 1, 2],
               [3, 4, 5]])
    >>>
    >>> get_all_but_indexed_value(a,index=[0,1])
    >>> array([[1, 2],
              [3, 5]])
    '''
    
    # Make sure we manipulate numpy arrays
    a,index=np.array(a),np.array(index)
    
    # Sanity checks
    if a.ndim!=2 or index.ndim!=1:
        err  = 'This function requires an array \'a\' of dimension 2 (it is currently {})\n'.format(a.ndim)
        err += 'and an array \'index\' of dimension 1 (it is currently {})'.format(index.ndim)
        raise NameError(err)
    if a.shape[0] != index.shape[0]:
        err  = 'The two array must have the same number of element along the first axis.\n'
        err += 'while currently \'a\' has {} elements and \'index\' has {}.'.format(a.shape[0],index.shape[0])
        raise NameError(err)
    
    N=np.arange(a.shape[0])
    return np.array([np.concatenate([a[i,:index[i]],a[i,index[i]+1:]]) for i in N])



def square_jagged_2Darray(a,**kwargs):    
    '''
    Give the same dimension to all raws of a jagged 2D array.

    This function equalizes the the size of every raw (obj collection)
    using a default value 'val' (nan if nothing specifed) using either
    the maximum size of object collection among all column (events) or
    using a maximum size 'size'. The goal of this function is to fully
    use numpy vectorization which works only on fixed size arrays.
    
    Parameters
    ----------
    a: array of arrays with different sizes this is the jagged 2D 
    array to be squared
    
    keyword arguments
    -----------------
    dtype: string
        data type of the variable-size array. If not specified, 
        it is 'float32'. None means dt=data.dt.
    nobj: int
        max size of the array.shape[1]. if not specified (or None), 
        this size is the maximum size of all raws.
    val: float32
        default value used to fill empty elements in order to get 
        the proper size. If not specified (or None), val is np.nan.
    
    Returns
    -------
    out: np.ndarray
        with a dimension (ncol,nobj).
             
    Examples
    --------
    >>> import numpy as np
    >>> a=np.array([
        [1,2,3,4,5],
        [6,7],
        [8],
        [9,10,11,12,13]
    ])
    >>>
    >>> square_jagged_2Darray(a)
    array([[  1.,   2.,   3.,   4.,   5.],
       [  6.,   7.,  nan,  nan,  nan],
       [  8.,  nan,  nan,  nan,  nan],
       [  9.,  10.,  11.,  12.,  13.]], dtype=float32)
    >>>
    >>> square_jagged_2Darray(a,nobj=2,val=-999)
    >>> array([[   1.,    2.],
       [   6.,    7.],
       [   8., -999.],
       [   9.,   10.]], dtype=float32)
    '''
    
    # Sanity checks
    if a.ndim>=2:
        err  = 'The input array a should be a 1D array of 0D/1D arrays. This means that '
        err += 'a.shape=(N,) or (1,) while here '
        err += 'a.shape={}'.format(a.shape)
        raise NameError(err)    
    dims=np.array([e.ndim for e in a])
    Neq0,Ngt2=np.count_nonzero(dims==0),np.count_nonzero(dims>=2)
    if Neq0==len(a):
        return a
    if Neq0>0 or Ngt2>0:
        err  = 'The input array should be a 1D array of 1D arrays'
        err += ' in order to be converted into a 2D array.\n Some'
        err += ' of the sub-array have dim>=2 or dim=0 (ie not an array):\n'
        err += '  -> Number of d==0 element: {} (if ==len(a), it\'s not a jagged array!)\n'.format(Ngt2)
        err += '  -> Number of d>=2 element: {}\n'.format(Ngt2)
        raise NameError(err)
    
    # kwargs
    val,size,dtype=np.nan,None,'float32'
    if 'dtype' in kwargs: dtype=kwargs['dtype']
    if 'nobj'  in kwargs: size=kwargs['nobj']
    if 'val'   in kwargs: val=kwargs['val']
        
    # Get lengths of each row of data
    lens = np.array([len(i) for i in a])

    # Mask valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]
   
    # Setup output array and put elements from data into masked positions
    if (dtype): dt=dtype
    else      : dt=data.dtype
    out = np.zeros(mask.shape, dtype=dt)
    out.fill(val)
    out[mask] = np.concatenate(a)
    
    # Keep the number of element to size
    if size: out=out[:,:size]
    
    return out



def all_pairs_nd(a,b=None,Nmax=None,axis=1,timing=False):
    '''
    Compute all possible pairs along a given axis.

    This function performs the list of all possible pairs along a given axis
    of the between the two arrays a and b. The typical use case it the following:
    there are Nevts events with two collections of 5 vectors {r_i} and 10 vector 
    {q_j} (where each vector q,r=(px,py,pz)), and the pair (q,r) with the smallest 
    distance is wanted. In that case, one has:
      a.shape=(Nevts, 5,3)
      b.shape=(Nevts,10,3)
      all_pairs_nd(a,b).shape=(Nevts,50,2,3)
    
    NB1: If only a is given, the unordered/unrepeated combinations 
         are performed.
    NB2: all axis must have the same dimension, expect the one along which the 
         pairing is done.

    Parameters
    ----------
    a: np.ndarray
        The array contains the objects collection for each event. If Nobja
        is the number of objects a and k the number of variable of each object a
        (e.g. [px,py,pz,E,btagg,iso]): a.shape=(Nevt,Nobj_a,k)
    b: np.ndarray
        The array contains the objects collection for each event. If Nobj
        is the number of objects and l the number of variable of each object b
        (e.g. [px,py,pz,E,btagg,iso]): l must be equal to k and b.shape=(Nevt,Nobj_b,k).
        If not specified, combinations of a elements are returned.
    Nmax: int
        Maximal number of elements considered to compute all combinations
    axis: int
        The dimension along which the pairing is done (axis=1 if not specified since
        the most common HEP array is (Nevt,Nobj,k)).
    timing: boolean
        Print the time of each of the four main steps and the total one (useful
        to degub).
    
    Returns
    -------
    pairs: nd.ndarray
        For each event (element along axis=0), the output array has Npairs of 2 objects,
        meaning that output.shape=(Nevt, Npairs, 2, k).
                 
    Examples
    --------
    >>> import numy as np
    >>> a=np.array([ # Nevt=1, Nobj=3, k=2
        [[0, 1],[2, 3],[4, 5]]
    ])
    >>>
    >>> b=np.array([ # Nevt=1, Nobj=2, k=2
        [[6, 7],[8, 9]]
    ])
    >>> 
    >>> all_pairs_nd(a,b)
    >>> array([
        [
         [[0, 1],[6, 7]],

         [[0, 1],[8, 9]],

         [[2, 3],[6, 7]],

         [[2, 3],[8, 9]],

         [[4, 5],[6, 7]],

         [[4, 5],[8, 9]]
         ]
       ])
    >>>
    >>> all_pairs_nd(a)
    >>> array([
        [
         [[0, 1],[2, 3]],

         [[0, 1],[4, 5]],

         [[2, 3],[4, 5]]
        ]
      ])
    >>>
    >>> npu.all_pairs_nd(a,Nmax=2)
    >>> array([
      [
       [[0, 1],[2, 3]]
      ]
    ])
    '''
    
    from timeit import default_timer
    t0 = default_timer()
    
    # Is it the same collection
    same_arrays = b is None
        
    # Sanity check
    if not same_arrays:
        good_shape=np.array_equal(np.delete(a.shape,axis),np.delete(b.shape,axis))
        if not good_shape:
            err  = 'The shape along all dimensions but the one of axis={}'.format(axis)
            err += ' should be equal, while here:\n'
            err += '  -> shape of a is {} \n'.format(a.shape)
            err += '  -> shape of b is {} \n'.format(b.shape)
            raise NameError(err)
    
    # Reduce the number of objects to Nmax
    if Nmax:
        sl=[slice(None)]*a.ndim
        sl[axis]=slice(0,Nmax)
        if same_arrays: a,b=a[sl],None
        else          : a,b=a[sl],b[sl]
    
    t1 = default_timer()
    if timing: print('   * Sanity checks done in {:.3f}s'.format(t1-t0))
          
    # Individual indices
    if same_arrays:
        ia,jb=np.arange(a.shape[axis]),[]
    else:
        ia,jb=np.arange(a.shape[axis]),np.arange(b.shape[axis])
    
    t2 = default_timer()
    if timing: print('   * Individual indices done in {:.3f}'.format(t2-t1))
          
    # Pairs of indicies
    dt=np.dtype([('', np.intp)]*2)
    if same_arrays: ij=np.fromiter(itertools.combinations(ia,2),dtype=dt)     
    else          : ij=np.fromiter(itertools.product(ia,jb),dtype=dt)
    ij=ij.view(np.intp).reshape(-1, 2)
    
    t3 = default_timer()
    if timing: print('   * Pairs of indices done in {:.3f}s'.format(t3-t2))
          
    # Array of all pairs
    if same_arrays: out=np.take(a,ij,axis=axis)
    else          : out=np.stack([a.take(ij[:,0],axis=axis),b.take(ij[:,1],axis=axis)],axis=axis+1)
          
    t4 = default_timer()
    if timing: print('   * Take and stack arrays done in {:.3f}s'.format(t4-t3))
    if timing: print(' ==> total time: {:.3f}'.format(t4-t0))
    
    return out



def df2array(df,variables,**kwargs):
    '''
    Convert a list of Ncols pandas dataframe columns into a regular
    (Nevt,Nobj,Ncol)-dim numpy array.
    
    In practice, the exact size of the final array is Nevt (the number
    of events), Nobj (number of objects) and Ncol which is the number of 
    float for each event and object. 
    
    It is possible to give default values in order to later form collections
    with the same number of variables:
        jets     =df2array(df,['jet_eta', 'jet_phi', 'jet_bw',     '999'])
        electrons=df2array(df,[ 'el_pt' ,  'el_phi ,    'nan', 'trk_iso'])
        pairs    =all_pairs_nd(jets,electrons)
    This allows to get the electron isolation and the b-tagging weight
    for the electron-jet pair being the closest to each other.
    
    Parameters
    ----------
    df: pandas.DataFrame
    variables: list of column names to extract
    
    keyword arguments
    -----------------
    The same as for square_jagged_2Darray(a,**kwargs) function
    
    Returns
    -------
    output: np.array
        3D array given with output.shape=(df[v].shape[0],df[v].shape[1],len(variables))
        
    Examples
    --------
    >>>
    >>> data=pd.DataFrame(data={
        'jet_eta':np.array([np.array([1,2,3]),np.array([4,5])]),
        'jet_phi':np.array([np.array([6,7,8]),np.array([9,10])]),
        })
    >>> print(data)
    >>>      jet_eta    jet_phi
        0  [1, 2, 3]  [6, 7, 8]
        1     [4, 5]    [9, 10]
    >>>
    >>> jets_direction=npu.df2array(data,['jet_eta','jet_phi'])
    >>> jets_direction
    >>> array([[[  1.,   6.],
        [  2.,   7.],
        [  3.,   8.]],

       [[  4.,   9.],
        [  5.,  10.],
        [ nan,  nan]]], dtype=float32)
    '''
        
    # Get the default array with the proper shape
    if variables[0] not in df.columns:
        err  = 'The first variable must be a valid column and not a default value.\n'
        err += 'The variable \'{}\' is not in the list of dataframe columns'.format(variables[0])
        raise NameError(err)
    Nevt,Nobj=len(df),np.max([len(i) for i in df[variables[0]].values])
    if 'nobj'  in kwargs: Nobj=kwargs['nobj']
    
    def default_array(str_val):
        try:               
            val=float(str_val)
        except ValueError: 
            if str_val in ('nan','NaN','Nan','NAN'):
                val=np.nan
            else:
                err  = 'The default value \'{}\' is not supported. Please only use '.format(str_val)
                err += 'a number in a string (e.g. \'999\') or \'nan\'.'
                raise NameError(err)
        return np.full_like(np.zeros((Nevt,Nobj)),val)
    
    # Get the list of all arrays, each of shape: (Nevts,Nobj)
    list_arrays = [square_jagged_2Darray(df[v].values,**kwargs) if v in df.columns else default_array(v) for v in variables]
    
    # Check that there are a collection (and not only value, like MET)
    isCollection=contains_collections(list_arrays)
    
    # Adding a dimension for further concatenation in case of 
    # (Nevts,Nobj) shape; new shape is (Nevts,Nobj,1)
    list_arrays = [a[...,None] if a.ndim==2 else a for a in list_arrays]
        
    # Check that the number of object is the same for all column
    if isCollection:
        axis=2
        if np.std([a.shape[1] for a in list_arrays])!=0:
            err ='The shape along the dimensions of axis=1 (number of objects) '
            err+='must be the same for all variables. This function cannot merge different '
            err+='object collections (eg "jet_pT" and "ele_pT").\n'
            err+='If you need to do so, check stack_collection() functions.'
            raise NameError(err)
    else:
        axis=1
    
    # Performe the concatenation and output shape is (Nevts,Nobj,Nvariables)
    return np.concatenate(list_arrays,axis=axis)




def stack_collections(arrays):
    '''
    Stack list of arrays of shape (Nevts,Nobj_i,Nval) along axis=1.
    
    The typical use case of the function is to build a single collection
    of objects from different collections. Let's take the example where
    one wants to make a 'lepton' collection out of 'electron' and 'muon'
    collection: each collection has Nval variables so that each array will
    be of shape el.shape=(Nevts,Nel,Nval) and mu.shape=(Nevts,Nmu,Nval).
    lep=stack_collections([el,mu]) will have lep.shape(Nevt,Nel+Nmu,Nval).
   
   
    Parameters:
    ----------
    arrays: list of ndarray
        arrays which needs to be stacked
   
   
    Return:
    -------
    output: ndarray
        array of shape (Nevt,Ntot,Nval) where Ntot
        is the sum of all objects (e.g. Nlep+Njet)
   
   
    Examples:
    --------
    >>> a=np.arange(30).reshape(2,5,3)
    >>> a # 2 events, 5 objects, 3 variables
    >>> array([
        [
         [ 0,  1,  2],
         [ 3,  4,  5],
         [ 6,  7,  8],
         [ 9, 10, 11],
         [12, 13, 14]
        ],

       [
         [15, 16, 17],
         [18, 19, 20],
         [21, 22, 23],
         [24, 25, 26],
         [27, 28, 29]
        ]
      ])
    >>>
    >>> b=np.arange(12).reshape(2,2,3)
    >>> b # 2 events, 2 objects, 3 variables
    >>> array([
         [
          [ 0,  1,  2],
          [ 3,  4,  5]
         ],

         [
          [ 6,  7,  8],
          [ 9, 10, 11]
         ]
        ])
    >>> 
    >>> npu.stack_collections([a,b])
    >>> array([
         [
          [ 0,  1,  2],
          [ 3,  4,  5],
          [ 6,  7,  8],
          [ 9, 10, 11],
          [12, 13, 14],
          [ 0,  1,  2],
          [ 3,  4,  5]
         ], 

         [
          [15, 16, 17],
          [18, 19, 20],
          [21, 22, 23],
          [24, 25, 26],
          [27, 28, 29],
          [ 6,  7,  8],
          [ 9, 10, 11]
         ]
        ])
    '''
    
    # Check there are collection
    if not contains_collections(arrays):
        err ='One of the array is not a collection, while this function needs ' 
        err+='collections of objects (ie at least 2D arrays - 1D for events '
        err+='and 1D for the collection'
        raise NameError(err)
           
    # Check that the number of variables per object is the same for all column
    has_one_var= np.count_nonzero([a.ndim==1 for a in arrays])==len(arrays)
    if not has_one_var: is_ok=True
    elif len(np.unique([a.shape[2] for a in arrays]))!=0: is_ok=False
    else: is_ok=True
    if not is_ok:
        err ='The shape along the dimensions of axis=2 (number of variables per object) '
        err+='must be the same for all objects. This function cannot merge '
        err+='collections with different number of variables (eg [jet_pT,jet_eta] and [ele_pT]).\n'
        raise NameError(err)
    
    out = np.concatenate(arrays,axis=1)
    return out
