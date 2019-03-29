'''

---------------------
composition_matrix.py
---------------------

 This python module allows to compute and plot the composition matrix
 along two dimensions. Typically, if some data can be classified according
 to two different criterias (e.g. physics processes and type of instrumental bkg), 
 the 3 built matrices shows percentages describing:
  1. how each physics process are distributed across of types of bkg
  2. how each type of background are distributed across of physics processes
  3. fraction of a given type of background for a given process wrt total background

 Usage
 -----
 - `import composition_matrix.py` will load all needed functions
 - plot_matrices(yields, errors) plot the composition matrices

 Dependences
 -----------
 - numpy
 - matplotlib

'''

import numpy as np
import matplotlib.pyplot as plt


def binomial_uncertainty(n1, n2, e1, e2):
    '''
    Binomial uncertainty on n1/(n1+n2)
    '''
    den = (n1+n2)**4
    if den == 0:
        return 0.
    drdn1_2 = n2**2/den
    drdn2_2 = n1**2/den
    return np.sqrt(drdn1_2*e1**2 + drdn2_2*e2**2)


def get_fractions(yields, errors, sum_type):
    '''
    Compute fraction over different axis, together with the associated uncertainty
    return fractions(np.array), errors (np.array)
    '''

    # Sanity check
    if yields.shape != errors.shape:
        err = 'Shape of yields {} and error {} are different'.format(yields.shape, errors.shape)
        raise NameError(err)
    
    # Create containers for the results
    res_yields = np.zeros(shape=yields.shape)
    res_errors = np.zeros(shape=yields.shape)
    Nx, Ny = yields.shape
    
    # Define how to remove the correct element to get n2 (depending on sum type)
    if sum_type == 'x-cats':
        clean = lambda ar, x, y: np.delete(yields[:, y], x)
    elif sum_type == 'y-cats':
        clean = lambda ar, x, y: np.delete(yields[x, :], y)
    elif sum_type == 'all':
        clean = lambda ar, x, y: np.delete(yields, [x, y])
    else:
        err = '{} is not supported, use \'x-cats\', \'y-cats\' or \'all\''.format(sum_type)
        raise NameError(err)
    
    # Loop over elements
    for ix in np.arange(Nx):
        for iy in np.arange(Ny):

            # Get independent variables n1, n2 and the associated errors
            n1, e1 = yields[ix, iy], errors[ix, iy]
            n2, e2 = np.sum(clean(yields, ix, iy)), np.sum(clean(errors, ix, iy)**2)**0.5

            # Get fraction (in case Ntot!=0)
            if n1+n2 == 0:
                res_yields[ix, iy] = 0
            else:
                res_yields[ix, iy] = n1/(n1+n2)

            # Get uncertainty
            res_errors[ix, iy] = binomial_uncertainty(n1, n2, e1, e2)

    # Cleaning
    res_yields[np.isnan(res_yields)] = 0
    res_errors[np.isnan(res_errors)] = 0

    # Result in %
    return res_yields*100, res_errors*100


def plot_matrices(yields, errors, SoverB=None, cmap='Blues',
                  xcat_name=None, ycat_name=None,
                  xlabels=None, ylabels=None, added_title=None, grid=True, 
                  figsize=None, label_size=20, number_size=19, title_size=30):

    '''
    Plot three matrices showing the decomposition of 
     1. y-categories into x-categories
     2. x-categories into y-categories
     3. each (x, y) categories - ie fraction wrt the total
    Note: this is done using mpl.imshow() which put the origin at the top left corner.

    Parameters
    ----------
    yields [2d numpy]: array of numbers (x-yields in axis=0 and y-yields in axis=1), ie:
      [ 
        [yield(x=0, y=0), yield(x=0, y=1), ...],
        [yield(x=1, y=0), yield(x=1, y=1), ...],
        ...
      ]
    errors [2d numpy]: array of number uncertainties (x-yields in axis=0 and y-yields in axis=1)
    SoverB [float (optional)]: S/sqrt(B)
    cmap [string]: color map among those available in matplotlib
    xcat_name [string]: name of x-axis type (used for the title)
    ycat_name [string]: name of y-axis type (used for the title)
    xlabels [list of string]: list of names of each x types (used for the x-axis)
    ylabels [list of string]: list of names of each y types (used for the y-axis)
    added_title [string]: Additional title coming after Ntot
    grid [bool]: display the grid or not (default: True)
    figsize [(float, float)]: width, height of the figure (inch)
    label_size [int]: font size for axis labels
    number_size [int]: font size for numbers
    title_size [int]: font size for title

    Return
    ------
    None
    '''
    
    # Compute all types of fractions
    sum_x = get_fractions(yields, errors, sum_type='x-cats')
    sum_y = get_fractions(yields, errors, sum_type='y-cats')
    sum_a = get_fractions(yields, errors, sum_type='all')

    # Number of bins
    Nx, Ny = yields.shape

    # Figure an plot
    w = 2.3*Nx+23
    h = 0.85*Ny+6
    if figsize:
        w, h = figsize
    plt_style = {'nrows': 1, 'ncols': 3, 'figsize': (w, h), 'facecolor': 'white'}
    fig, (ax1, ax2, ax3) = plt.subplots(**plt_style)

    # Loop over plots (ax objects)
    for iax, (ax, data_error) in enumerate(zip([ax1, ax2, ax3], [sum_x, sum_y, sum_a])):

        # Get data to plot
        data, error = data_error
        
        # Plot the data
        ax.imshow(data.T, cmap=cmap)

        # Axis position
        ax.xaxis.tick_top()

        # Axis ticks
        ax.set_xticks(np.arange(Nx))
        ax.set_yticks(np.arange(Ny))
        ax.set_xticks(np.arange(-0.5, Nx, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, Ny, 1), minor=True)

        # Axis labels
        if not xlabels:
            xlabels = ['x-cat. {}'.format(i+1) for i in range(Nx)]
        if not ylabels:
            ylabels = ['y-cat. {}'.format(i+1) for i in range(Ny)]
        ax.set_xticklabels(xlabels, fontsize=label_size, rotation=25)
        ax.set_yticklabels(ylabels, fontsize=label_size)

        # Axis grid
        if grid:
            grid_style = {'color': 'tab:blue', 'alpha': 0.7,
                          'linewidth': 2, 'linestyle': ':',
                          'which': 'minor'}
            if iax == 0:
                ax.yaxis.grid(**grid_style)
            if iax == 1:
                ax.xaxis.grid(**grid_style)

        # Loop over data dimensions and create text annotations.
        text_option = {'ha': 'center', 'va': 'center', 'color': 'black', 'fontsize': number_size}
        data_max = np.max(data)
        for i in range(Nx):
            for j in range(Ny):
                if data[i, j] > 0.7*data_max:
                    text_option['color'] = 'white'
                elif data[i, j] < 1:
                    text_option['color'] = 'lightgrey'
                else:
                    text_option['color'] = 'black'
                val = '{:.0f}$\\,\\pm\\,{:.0f}$'.format(data[i, j], np.sqrt(error[i, j]))
                ax.text(i, j, val, **text_option)

    # Titles
    if not xcat_name:
        xcat_name = 'x-cat.'
    if not ycat_name:
        ycat_name = 'y-cat.'
    title = 'Decomposition [%] of {} (left), {} (middle), both (right) $-$ '.format(ycat_name, xcat_name)
    title += '$N_{tot}='+'{:.1f}$'.format(np.sum(yields))
    if SoverB:
        title += ' $-$ $S/\\sqrt{B}'
        title += '={:.2f}$'.format(SoverB)
    if added_title:
        title += ' $-$ ' + added_title

    fig.suptitle(title, fontsize=title_size)

    return
