"""
Creates plots using the matplotlib.pyplot module.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pyPartAnalysis.transform as tf
import pyPartAnalysis.stripes as st
import pyPartAnalysis.twiss as tw

def det_plot_scale(ps_df,cutoff = 0.9):
    """Gives the scalings for a physical particle distribution
    
    Takes the 6 dimensional phase space and returns the associated scaling.

    Parameters
    ----------
    ps_df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    cutoff : float
        Indicates the cutoff value for scale, e.g. 0.75 means 
        that a value of 750 will return a scale of 3 instead 
        of 0. Must be greater than 0 and equal to or less 
        than 1.
    
    Returns
    -------
    dict of {str : dict of {str : int} and {str : str}}
        A dictionary with the scaling for each dimension. 
        Can be accessed using the the coordinate name at the first 
        level ['x','xp','y','yp','z','delta'].
    """
    
    scaleSteps = 3
    maxExtents = abs(pd.concat([ps_df.max(axis=0), ps_df.min(axis=0)],axis=1))
    maxExtent = maxExtents.max(axis=1)
    maxExtent[maxExtent==0] = 1
    scale = np.floor(np.log10(maxExtent)*cutoff)
    scale = scaleSteps*np.floor(scale/scaleSteps);
    
    scale_info = {name:scale.iloc[idx]
              for idx, name in enumerate(ps_df.columns.values)}  

    return scale_info

def make_phase_space_axis_labels(dim,scale_info,force_latex=False):
    """Makes labels for axis formatted based on scaling.
    
    Note that for the transverse angle labels, we assume 
    the paraxial approximation.
    
    Parameters
    ----------
    dim : {'x', 'xp', 'y', 'yp', 'z', 'delta'} 
        Dimension for which the label is returned.
    scale_info : {0,-3,-6,-9,-12,-15}
        Exponential Factor for scaling axis.
        
    
    Returns
    -------
    String
        Label for the specified axis, formatted 
        based on the scaling.
    """
    space_labels = { 0:'(m)',
                    -3:'(mm)',
                    -6:'(um)',
                    -9:'(nm)',
                    -12:'(pm)',
                    -15:'(fm)'}
    transverse_angle_labels = {0:'(rad)',
                               -3:'(mrad)',
                               -6:'(urad)',
                               -9:'(nrad)',
                               -12:'(prad)',
                               -15:'(frad)'}

    # prevent scalings greater than 0
    scale_info = np.min([0,scale_info])
    
    if(dim=='x'):
        if plt.rcParams['text.usetex'] is not True and not force_latex:
            label=f"x {space_labels[scale_info]}"
        else:
            label=f"$x \: {space_labels[scale_info]}$"
    elif(dim=='xp'):
        if plt.rcParams['text.usetex'] is not True and not force_latex:
            label=f"x' {transverse_angle_labels[scale_info]}"
        else:
            label=f"$x^{{\\prime}} \: {transverse_angle_labels[scale_info]}$"
    if(dim=='y'):
        if (plt.rcParams['text.usetex'] is not True) and not force_latex:
            label=f"y {space_labels[scale_info]}"
        else:
            label=f"$y \: {space_labels[scale_info]}$"
    elif(dim=='yp'):
        if plt.rcParams['text.usetex'] is not True and not force_latex:
            label=f"y' {transverse_angle_labels[scale_info]}"
        else:
            label=f"$y^{{\\prime}} \: {transverse_angle_labels[scale_info]}$"
    elif(dim=='z'):
        if plt.rcParams['text.usetex'] is not True and not force_latex:
            label=f"z {space_labels[scale_info]}"
        else:
            label=f"$z \: {space_labels[scale_info]}$"
    elif(dim=='delta'):
        if plt.rcParams['text.usetex'] is not True and not force_latex:
            label=f"delta x 10^{int(-scale_info)}"
        else:
            label=f"$\\delta \\times 10^{int(-scale_info)}$"
            
    return label

def plot_phase_space_density(df,dim,num_bins,ax=None,cutoff = 0.9,**plt_kwargs):
    """Plots the specfied phase space with scaling and appropriate labels
    
    The defaults can be overwritten using the keyword arguments for 
    **plt_kwargs that a the same as those supplied to matplotlib.pyplot.hist2d.
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    dim : {'x', 'y', 'z'} 
        Dimension to be plotted
    ax : matplotlib.axes, optional, default None
        Axes the plot will be plotted on        
    cutoff: float
        Cutoff value for scaling of the axes. A value between 0 and 1.
        See det_plot_scale for more info. 
    **plt_kwargs
        Extra arguments for matplotlib.pyplot.hist2d
        
    
    Returns
    -------
    matplotlib.axes
        Axes to which were plotted
    """
    
    dim_dict = {'x':['x','xp'],
            'y':['y','yp'],
            'z':['z','delta']}
    
    scale_info = det_plot_scale(df,cutoff)

    if ax is None:
        ax = plt.gca()

    dim0 = dim_dict[dim][0]
    dim1 = dim_dict[dim][1]
    
    ax.set(xlabel=make_phase_space_axis_labels(dim0,scale_info[dim0]),
           ylabel=make_phase_space_axis_labels(dim1,scale_info[dim1]))             
    ax.hist2d(df[dim0]*10**-scale_info[dim0], 
               df[dim1]*10**-scale_info[dim1],
               bins = num_bins,
               **plt_kwargs)
        
    return ax

# Finish documentation for below function

def plot_transverse_density(df,num_bins,ax=None,cutoff = 0.9,**plt_kwargs):
    """Plots the transverse density plot with scaling and appropriate labels
    
    The defaults can be overwritten using the keyword arguments for 
    **plt_kwargs that a the same as those supplied to matplotlib.pyplot.hist2d.
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    num_bins : int
        Number of bins are the same in both dimensions by default
    ax : matplotlib.axes, optional, default None
        Axes the plot will be plotted on        
    cutoff: float
        Cutoff value for scaling of the axes. A value between 0 and 1.
        See det_plot_scale for more info. 
    **plt_kwargs
        Extra arguments for matplotlib.pyplot.hist2d
        
    Returns
    -------
    matplotlib.axes
        Axes to which were plotted
    """
    
    scale_info = det_plot_scale(df,cutoff)

    if ax is None:
        ax = plt.gca()

    ax.set(xlabel=make_phase_space_axis_labels('x',scale_info['x']),
           ylabel=make_phase_space_axis_labels('y',scale_info['y']))
    ax.hist2d(df['x']*10**-scale_info['x'], 
              df['y']*10**-scale_info['y'],
              bins = num_bins,
              **plt_kwargs)
    
    return ax

def plot_transverse_angle_density(df,num_bins,ax=None,cutoff = 0.9,**plt_kwargs):
    """Plots the transverse angle density with scaling and appropriate labels
    
    The defaults can be overwritten using the keyword arguments for 
    **plt_kwargs that a the same as those supplied to matplotlib.pyplot.hist2d.
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    num_bins : int
        Number of bins are the same in both dimensions by default
    matplotlib.axes : matplotlib.axes, optional, default None
        Axes the plot will be plotted on
    ax : matplotlib.axes, optional, default None
        Axes the plot will be plotted on        
    cutoff: float
        Cutoff value for scaling of the axes. A value between 0 and 1.
        See det_plot_scale for more info. 
    **plt_kwargs
        Extra arguments for matplotlib.pyplot.hist2d
        
    
    Returns
    -------
    matplotlib.axes
        Axes to which were plotted
    """
    
    scale_info = det_plot_scale(df,cutoff)

    if ax is None:
        ax = plt.gca()

    ax.set(xlabel=make_phase_space_axis_labels('xp',scale_info['xp']),
           ylabel=make_phase_space_axis_labels('yp',scale_info['yp']))
    ax.hist2d(df['xp']*10**-scale_info['xp'], 
              df['yp']*10**-scale_info['yp'],
              bins = num_bins,
              **plt_kwargs)
    
    return ax

def scatter_phase_space_corr(df1,dim1,df2,dim2,corr,ax=None,cutoff = 0.9,**scatter_kwargs):
    """Scatter plot with color coding for correlation examination
    
    The defaults can be overwritten using the keyword arguments for 
    **scatter_kwargs that a the same as those supplied to matplotlib.pyplot.scatter.
    
    Parameters
    ----------
    df1 : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
        For the x axis of plot.
    dim1 : {'x', 'xp', 'y', 'yp', 'z', 'delta'} 
        Dimension to be plotted along x
    df2 : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
        For the y axis of the plot.
    dim2 : {'x', 'xp', 'y', 'yp', 'z', 'delta'} 
        Dimension to be plotted along y
    corr : array_like, shape=(N,)
        Values used for color coding the scatter plot.
        Same length as number of rows of df1 and df2.
    ax : matplotlib.axes, optional, default None
        Axes the plot will be plotted on        
    cutoff: float
        Cutoff value for scaling of the axes. A value between 0 and 1.
        See det_plot_scale for more info.       
    **scatter_kwargs
        Extra arguments for matplotlib.pyplot.scatter.
        
    
    Returns
    -------
    matplotlib.axes
        Axes to which were plotted
    """
    
    temp1=tf.make_mean_zero(df1)
    temp2=tf.make_mean_zero(df2)
    
    if ax is None:
        ax = plt.gca()
    
    scale_info_output1 = det_plot_scale(temp1,cutoff)
    scale_info_output2 = det_plot_scale(temp2,cutoff)

    ax.scatter(temp1[dim1]*10**-scale_info_output1[dim1],
             temp2[dim2]*10**-scale_info_output2[dim2],
             c=corr,
              **scatter_kwargs)

    ax.set_xlabel(make_phase_space_axis_labels(dim1,scale_info_output1[dim1]))
    ax.set_ylabel(make_phase_space_axis_labels(dim2,scale_info_output2[dim2]))
    
    return ax

def plot_color_coded_stripes(df,dim1,dim2,ax=None,cutoff=0.9,**plt_kwargs):
    """Plot Stripes color coded by stripe id
    
    The defaults can be overwritten using the keyword arguments for 
    **plt_kwargs that a the same as those supplied to matplotlib.pyplot.plot.
    
    Parameters
    ----------
    df1 : DataFrame
        Multi-level indexed particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
        The 0 level index is the particle id or 'id', while 
        the 1 level index is the stripe id or 'stripe_id'.
    dim1 : {'x', 'xp', 'y', 'yp', 'z', 'delta'} 
        Dimension to be plotted along x
    dim2 : {'x', 'xp', 'y', 'yp', 'z', 'delta'} 
        Dimension to be plotted along y
    cutoff: float
        Cutoff value for scaling of the axes. A value between 0 and 1.
        See det_plot_scale for more info.
    **plt_kwargs
        Extra arguments for matplotlib.pyplot.plot.
        
    
    Returns
    -------
    matplotlib.axes
        Axes to which were plotted.
    """
    
    if ax is None:
        ax = plt.gca()
        
    groups_many = df.groupby(df.index.get_level_values('stripe_id')) 
    
    scale_info_output = det_plot_scale(df,cutoff)

    for name, group in groups_many:
        ax.plot(group[dim1]*10**-scale_info_output[dim1],
                     group[dim2]*10**-scale_info_output[dim2],
                     marker='.',
                     linestyle="",
                     label=name,
                     **plt_kwargs)
    ax.set_xlabel(make_phase_space_axis_labels(dim1,scale_info_output[dim1]))
    ax.set_ylabel(make_phase_space_axis_labels(dim2,scale_info_output[dim2]))
    
    return ax

def plot_bunching_factor_area(df,b0,ax=None,fig=None,cutoff = 0.9,**plt_kwargs):
    """Plot the bunching factor vs transverse bin
    
    Can be used to plot either the magnitude or the phase using
    
        numpy.abs(b0)
        numpy.angle(b0)

    for the b0 argument.
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
        that was used in calculating the bunching factor.
    b0 : ndarray, shape(M,N,1)
        bunching factor at the transverse positions
    ax : matplotlib.axes, optional, default None
        Axes the plot will be plotted on
    fig : matplotlib.figure, optional, default None
        Figure the plot will be plotted on
    cutoff: float
        Cutoff value for scaling of transverse bins. 
        A value between 0 and 1. See det_plot_scale for more info.
    **plt_kwargs
        Extra arguments for matplotlib.pyplot.imshow

    Returns
    -------
    fig : matplotlib.figure
        Figure the plot will be plotted on.
    ax : matplotlib.axes
        Axes the plot will be plotted on.
    """    
    
    scale_info = det_plot_scale(df,cutoff)

    if fig is None:
        fig = plt.figure()    
    if ax is None:
        ax = plt.subplot()
        
    min_x = df["x"].min(axis=0)
    max_x = df["x"].max(axis=0)
    min_y = df["y"].min(axis=0)
    max_y = df["y"].max(axis=0)
    dx = (max_x-min_x)*10**-scale_info['x']
    dy = (max_y-min_y)*10**-scale_info['y']
    extent = [min_x*10**-scale_info['x'],
              max_x*10**-scale_info['x'],
              min_y*10**-scale_info['y'],
              max_y*10**-scale_info['y']]
    
    # flipud is used as imshow inverts the numpy matrix b0.
    im = ax.imshow(np.flipud(b0),interpolation='nearest',extent=extent,aspect=dx/dy,**plt_kwargs)
    ax.set_xlabel(make_phase_space_axis_labels('x',scale_info['x']))
    ax.set_ylabel(make_phase_space_axis_labels('y',scale_info['y']))
    fig.colorbar(im,ax=ax)
    
    return fig, ax
    
def plot_bunching_factor_vs_wavelength(wavelengths,b0,ax=None,logscale = False,ymin=0,cutoff=0.9,**plt_kwargs):
    """Plot the bunching factor vs wavelength
    
    Plots the magnitude of the bunching factor vs the wavelength

    Parameters
    ----------
    b0 : ndarray, float, shape(num_bin_x)
        bunching factor at the transverse positions
    ax : matplotlib.axes, optional, default None
        Axes the plot will be plotted on
    fig : matplotlib.figure, optional, default None
        Figure the plot will be plotted on
    cutoff: float
        Cutoff value for scaling of transverse bins. 
        A value between 0 and 1. See det_plot_scale for more info.
    **plt_kwargs
        Extra arguments for matplotlib.pyplot.plot or 
        matplotlib.pyplot.semilogy depending on whether logscale is true 
        or false

    Returns
    -------
    matplotlib.axes
        Axes the plot will be plotted on
    """  
    
    scale_info_wav = det_plot_scale(pd.DataFrame({'x':wavelengths}),cutoff) 
    
    if ax is None:
        ax = plt.gca()
    
    wavelengths = wavelengths*10**-scale_info_wav['x'];

    wav_label = 'Wavelength ' + make_phase_space_axis_labels('x',scale_info_wav['x']).split()[1]
    ax.set_xlabel(wav_label)
    
    if plt.rcParams['text.usetex'] is not True:
        ax.set_ylabel("|b0| (arb units)")
    else:
        ax.set_ylabel("$$|b_0|\:(arb.\: units)$$")
        
    if(logscale==False):
        ax.plot(wavelengths,b0,**plt_kwargs)
    else:
        ax.semilogy(wavelengths,b0,**plt_kwargs)
        
    if(logscale==True and ymin <= 0):
        ax.set_xlim([min(wavelengths),max(wavelengths)])
        ax.set_ylim([None,None])
    else:
        ax.set_xlim([min(wavelengths),max(wavelengths)])
        ax.set_ylim([ymin,None])
        
    return ax

def plot_bunching_factor_slice(wavelengths,mean_slice,b0,dim_slice,ax=None,fig=None,cutoff = [0.9,0.9],**plt_kwargs):
    """Plot the slice bunching factor magnitude vs wavelength

    Parameters
    ----------
    wavelengths : array_like, shape=(m,)
        Wavelengths for which the bunching factor was 
        calculated (in meters).
    mean_slice : array_like, shape=(n,)
        Wavelengths for which the bunching factor was 
        calculated (in meters).        
    b0 : ndarray, shape(n,m)
        Magnitude of bunching factor at the mean slice 
        positions and wavelengths.
    ax : matplotlib.axes, optional, default None
        Axes the plot will be plotted on
    fig : matplotlib.figure, optional, default None
        Figure the plot will be plotted on
    cutoff: array_like, shape=(2,)
        Cutoff value for scaling of the slice dimensions and wavelengths.  
        A value between 0 and 1. See det_plot_scale for more info.
    **plt_kwargs
        Extra arguments for matplotlib.pyplot.imshow

    Returns
    -------
    fig : matplotlib.figure
        Figure the plot will be plotted on
    ax : matplotlib.axes
        Axes the plot will be plotted on
    """    
    
    scale_info_wav = det_plot_scale(pd.DataFrame({'x':wavelengths}),cutoff[0]) 
    scale_info_slice = det_plot_scale(pd.DataFrame({dim_slice:mean_slice}),cutoff[1]) 
    

    if fig is None:
        fig = plt.figure()    
    if ax is None:
        ax = plt.subplot()
        
    min_x = np.min(wavelengths)
    max_x = np.max(wavelengths)
    min_y = np.min(mean_slice)
    max_y = np.max(mean_slice)
    dx = (max_x-min_x)*10**-scale_info_wav['x']
    dy = (max_y-min_y)*10**-scale_info_slice[dim_slice]
    extent = [min_x*10**-scale_info_wav['x'],
              max_x*10**-scale_info_wav['x'],
              min_y*10**-scale_info_slice[dim_slice],
              max_y*10**-scale_info_slice[dim_slice]]
    
    # flipud is used as imshow inverts the numpy matrix b0.
    im = ax.imshow(np.flipud(b0),interpolation='nearest',extent=extent,aspect=dx/dy,**plt_kwargs)
    #im = ax.imshow(np.flipud(b0),interpolation='nearest',extent=extent,aspect=1,**plt_kwargs)
    wav_label = 'Wavelength ' + make_phase_space_axis_labels('x',scale_info_wav['x']).split()[1]
    ax.set_xlabel(wav_label)
    ax.set_ylabel(make_phase_space_axis_labels(dim_slice,scale_info_slice[dim_slice]))
    # ax.set(**plt_kwargs)
    fig.colorbar(im,ax=ax)
    
    return fig, ax

def plot_mismatch_slice(z,bmag,ax=None,cutoff=0.9,**plt_kwargs):
    """Plot the Twiss mismatch factor vs mean z of slice

    Parameters
    ----------
    z : ndarray, float, shape(num_slice)
        Mean z position of slice
    bmag : ndarray, float, shape(num_slice)
        Twiss mismatch factor of slice
    ax : matplotlib.axes, optional, default None
        Axes the plot will be plotted on
    cutoff : float
        Cutoff value for scaling of z. A value between 0 and 1.
        See det_plot_scale for more info.
    **plt_kwargs
        Extra arguments for matplotlib.pyplot.plot.

    Returns
    -------
    matplotlib.axes
        Axes the plot will be plotted on
    """      
    if ax is None:
        ax = plt.gca()
        
    scale_info = det_plot_scale(pd.DataFrame({'z':z}),cutoff) 
    
    ax.plot(z*10**-scale_info['z'],bmag,**plt_kwargs)
    ax.set_ylabel('MMF')
    ax.set_xlabel(make_phase_space_axis_labels('z',scale_info['z']))
    
    return ax
    
def plot_ellipse_stripe(df,dim,ax=None,cutoff = 0.9,**ax_kwargs):
    """Plot the RMS ellipse of the overal distribution and the central stripe.

    Parameters
    ----------
    ps_df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
        Has multi-level indexing with 'id' for the particle id and 
        'stripe_id' for the stripe stripe number. 
    ax : matplotlib.axes, optional, default None
        Axes the plot will be plotted on
    cutoff : float
        Cutoff value for scaling of the phase space plot.  
        A value between 0 and 1. See det_plot_scale for more info.
    **ax_kwargs
        Extra arguments for matplotlib.axes.

    Returns
    -------
    matplotlib.axes
        Axes the plot will be plotted on
    """   
    
    dim_dict_phase_space = {'x':['x','xp'],
                            'y':['y','yp'],
                            'z':['z','delta']}    
    
    dim_dict = {'x':0,'y':1,'z':2}
    
    scale_info = det_plot_scale(df,cutoff)
    
    if ax is None:
        ax = plt.gca()
    
    dim0 = dim_dict_phase_space[dim][0]
    dim1 = dim_dict_phase_space[dim][1]
    
    stripe_id_center = st.get_center_stripe_id(df,dim=dim)
    
    twiss_tot = tw.get_twiss_parameters(df)
    x_tot,y_tot = tw.twiss_ellipse_parametric(twiss_tot.alpha()[dim_dict[dim]],
                                              twiss_tot.beta()[dim_dict[dim]],
                                              twiss_tot.emit()[dim_dict[dim]])
    
    twiss_stripe = tw.get_twiss_parameters(st.filter_stripe(df,stripe_id_center))
    x_stripe,y_stripe = tw.twiss_ellipse_parametric(twiss_stripe.alpha()[dim_dict[dim]],
                                                    twiss_stripe.beta()[dim_dict[dim]],
                                                    twiss_stripe.emit()[dim_dict[dim]])
    
    plt.plot(x_tot*10**-scale_info[dim0],y_tot*10**-scale_info[dim1])
    plt.plot(x_stripe*10**-scale_info[dim0],y_stripe*10**-scale_info[dim1],color='red')
    
    ax.set(xlabel=make_phase_space_axis_labels(dim0,scale_info[dim0]),
           ylabel=make_phase_space_axis_labels(dim1,scale_info[dim1]),
           **ax_kwargs)   
    
    return ax
    
if __name__ == '__main__':
    pass
