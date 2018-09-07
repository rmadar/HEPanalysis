import numpy as np
import itertools

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
    size: int
        size of array. if not specified (or None), this size is 
        the maximum size of all raws.
    val: float32
        default value used to fill empty elements in order to get 
        the proper size. If not specified (or None), val is np.nan.
    
    Returns
    -------
    out: np.ndarray
        with a dimension (ncol,size).
             
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
    >>> square_jagged_2Darray(a,size=2,val=-999)
    >>> array([[   1.,    2.],
       [   6.,    7.],
       [   8., -999.],
       [   9.,   10.]], dtype=float32)
    '''
    
    # Sanity checks
    if a.ndim>=2:
        err  = 'The input array a should be a 1D array of 1D arrays. This means that '
        err += 'a.shape=(N,) while here '
        err += 'a.shape={}'.format(a.shape)
        raise NameError(err)    
    dims=np.array([e.ndim for e in a])
    Neq0,Ngt2=np.count_nonzero(dims==0),np.count_nonzero(dims>=2)
    if Neq0>0 or Ngt2>0:
        err  = 'The input array should be a 1D array of 1D arrays'
        err += ' in order to be converted into a 2D array.\n Some'
        err += ' of the sub-array has dim>=2 or dim==0 (ie not an array):\n'
        err += '  -> Nubmer of d==0 element: {}\n'.format(Neq0)
        err += '  -> Number of d>=2 element: {}\n'.format(Ngt2)
        raise NameError(err)
    
    # kwargs
    val,size,dtype=np.nan,None,'float32'
    if 'dtype' in kwargs: dtype=kwargs['dtype']
    if 'size'  in kwargs: size=kwargs['size']
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


def all_pairs_nd(a,b,axis=1):
    '''
    Compute all possible pairs along a given axis.

    This function performs the list of all possible pairs along a given axis
    of the between the two arrays a and b. The typical use case it the following:
    there are Nevts events with two collections of 5 vectors {r_i} and 10 vector 
    {q_j} (where each vector q,r=(px,py,pz)), and the pair (q,r) with the smallest 
    distance is wanted. In that case, one has:
      a.shape=(Nevts, 5,3)
      b.shape=(Nevts,10,3)
      all_pairs_nd(a,b,axis=1).shape=(Nevts,50,2,3)
    
    NB1: If a and b are the same arrays, the unordered/unrepeated combinations 
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
        (e.g. [px,py,pz,E,btagg,iso]): l must be equal to k and b.shape=(Nevt,Nobj_b,k)
    axis: int
        The dimension along which the pairing is done (axis=1 if not specified since
        the most common HEP array is (Nevt,Nobj,k)).
    
    Returns
    -------
    pairs: nd.ndarray
        For each event (element along axis=0), the output array has Npairs of 2 objects,
        meaning that output.shape=(Nevt, Nobj_a x Nobj_b, 2, k).
                 
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
    >>> all_pairs_nd(a,a)
    >>> array([
        [
         [[0, 1],[2, 3]],

         [[0, 1],[4, 5]],

         [[2, 3],[4, 5]]
        ]
      ])
    '''
    
    # Sanity check
    good_shape=np.array_equal(np.delete(a.shape,axis),np.delete(b.shape,axis))
    if not good_shape:
        err  = 'The shape along all dimensions but the one of axis={}'.format(axis)
        err += ' should be equal, while here:\n'
        err += '  -> shape of a is {} \n'.format(a.shape)
        err += '  -> shape of b is {} \n'.format(b.shape)
        raise NameError(err)
    
    # Individual indices
    a,b=np.asarray(a),np.asarray(b)
    ia,jb=np.arange(a.shape[axis]),np.arange(b.shape[axis])
    
    # Pairs of indicies
    dt=np.dtype([('', np.intp)]*2)
    if np.array_equal(a,b): 
        ij=np.fromiter(itertools.combinations(ia,2),dtype=dt)
    else: 
        ij=np.fromiter(itertools.product(ia,jb),dtype=dt)
    ij=ij.view(np.intp).reshape(-1, 2)
    
    # Array of all pairs
    ipair,jpair=ij[:,0],ij[:,1]
    return np.stack([a.take(ipair,axis=axis),b.take(jpair,axis=axis)],axis=axis+1)


def df2array(df,variables,**kwargs):
    '''
    Convert a pandas dataframe column into a regular 2D array using
    square_jagged_2Darray(a,**kwargs).

    Parameters
    ----------
    df: pandas.DataFrame
        Inital DataFrame containing the information
    variables: list of string
        List of column name to extract
    
    keyword arguments
    -----------------
    The same as for square_jagged_2Darray(a,**kwargs) function
    
    Returns
    -------
    output: np.array
        3D array given by square_jagged_2Darray(a,**kwargs)
    '''
    return np.stack([square_jagged_2Darray(df[v].values,**kwargs) for v in variables],axis=2)