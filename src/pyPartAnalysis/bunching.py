"""
Calculates the bunching factor for particle distribution.
"""

import numpy as np

from numba import jit
from collections.abc import Iterable

import pyPartAnalysis.analysis as an

def bunching_factor_numpy(x, wavelengths):
    """Computes the bunching factor from position data using only Numpy
    
    Slightly slower implementation than the numba version

    Parameters
    ----------
    x : array_like, shape=(M,)
        The position data
    wavelengths : array_like, shape=(N,)
        The wavelengths for which the bunching factor is calculated

    Returns
    -------
    array_like, shape=(N,)
        The complex bunching factor
    """
    
    temp = x.reshape(np.size(x),1)
    b0 = np.sum(np.exp(2j * np.pi * temp / wavelengths),axis=0)/np.size(x)
    
    return b0



@jit(nopython=True,fastmath=True)
def bunching_factor(x, wavelengths):
    """Computes the bunching factor from position data

    Parameters
    ----------
    x : array_like, shape=(M,)
        The position data
    wavelengths : array_like, shape=(N,)
        The wavelengths for which the bunching factor is calculated

    Returns
    -------
    array_like, shape=(N,)
        The complex bunching factor
        
    Examples
    --------
    >>> wavelengths = np.linspace(13.55,13.8,100)*1e-9
    >>> b0 = np.abs(bun.bunching_factor(df_many_stripes["x"].to_numpy(), wavelengths))
    """
    
    b0 = np.zeros(shape=np.shape(wavelengths),dtype=np.complex128)

    for inx, wavelength in enumerate(wavelengths):
        for x_val in x:
            b0[inx] += np.exp(2j * np.pi * x_val / wavelength )
    
    if(np.shape(x)[0] > 0):
        b0 = b0/np.shape(x)[0]
    return b0

@jit(nopython=True,fastmath=True)
def bunching_factor_scalar(x, wavelengths):
    """Computes the bunching factor from position data for scalar wavelength

    Parameters
    ----------
    x : array_like, shape=(M,)
        The position data
    wavelengths : float
        The wavelength for which the bunching factor is calculated

    Returns
    -------
    array_like, shape=(N,)
        The complex bunching factor
    """
    
    b0 = np.zeros(shape=1,dtype=np.complex128)
        
    for x_val in x:
        b0+= np.exp(2j * np.pi * x_val / wavelengths )
                
    if(np.shape(x)[0] > 0):
        b0 = b0/np.shape(x)[0]
    return b0

def weighted_bunching(b0,num_particles_slice):
    """Calculates the slice bunching factor weighted by the fraction 
    of particles in the most populated slice.
    
    Parameters
    ----------
    b0 : numpy.ndarray, complex, shape(M,N)
        Bunching factor of each N slices. Can be for M 
        different frequencies.
    num_particles_slice : numpy.array, shape(N,1)
        Number of particles in each slice

    Returns
    -------
    numpy.ndarray, complex, shape(M,N)
        Weighted Bunching Factor vs slice.
    """  
    
    num_particles_slice = num_particles_slice/np.max(num_particles_slice)
    return b0*num_particles_slice
    
def weighted_bunching_area(b0_area,count_bin):
    """Weights bunching factor area by the transverse bin with the largest 
    number of counts.
    
    Uses the outputs from bunching_factor_area.

    Parameters
    ----------
    b0_area : numpy.ndarray, shape(M,N,n)
        Bunching factor for transverse bins in x and y. Can be for n 
        different frequencies.
    count_bin : numpy.array, shape(M,N)
        Number of particles in each bin.

    Returns
    -------
    ndarray, complex, shape(M,N,n)
        Weighted bunching factor at the transverse bins
    """
    
    count_bin = count_bin/np.max(count_bin)
    return (np.abs(b0_area).T * count_bin.T).T
    
def get_iterable(x):
    """Converts an object to iterable if not already
    
    Parameters
    ----------
    x : array_like
        array to be iterable
    
    Returns
    -------
    array_like
        iterable array 
    """
    
    if isinstance(x, Iterable):
        return x
    else:
        return (x,)
    
def bunching_factor_area(df,wavelengths,num_pixels=[32,32]):
    """Calculates the bunching factor for transverse bins in x and y.
    
    Calculates the bunching factor at the wavelengths 

    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    wavelengths : array_like, shape(N,)
        Must be a positive real number.
    num_pixels : array_like, shape (2,), default [32,32]
        Must be a positive integer.

    Returns
    -------
    b0 : ndarray, complex, shape(num_pixels[0],num_pixels[1],N)
        Bunching factor at the transverse positions
    count_bin : ndarray, shape(num_pixels[0],num_pixels[1])
        Bunching factor for transverse bins in x and y
    """
    
    df_new = an.transverse_bin(df,num_pixels[0],num_pixels[1])
    
    b0_data = (df_new.groupby(['bins_x','bins_y'])["z"]
          .apply(lambda x: bunching_factor(x.to_numpy(), wavelengths)))
    
    b0 = np.zeros((num_pixels[1],num_pixels[0],len(wavelengths)),dtype=complex)

    for idx, df_select in b0_data.groupby(level=[0, 1]):
        nonzero = get_iterable(np.isnan(df_select.values[0]))
        for idz,cond in enumerate(nonzero):
            if(~cond):
                b0[idx[1],idx[0],idz] = df_select.values[0][idz]
    
    count_bin = an.transverse_bin_counts(df,b0.shape[0:2])   
    
    return (b0,count_bin)

def bunching_factor_slice(df,dimSlice,bins,dimBunching,wavelengths):
    """Calculates the bunching factor along slices in specified dimension
    
    Calculates the slice bunching factor at the wavelengths 

    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    dimSlice: {'x', 'y', 'z'}
        Dimension along which the distribution is sliced.
    bins: int
        Number of bins used in slicing.
        Must be a positive integer.
    dimBunching: {'x', 'y', 'z'}
        Dimension along which the bunching is calculated.
    wavelengths : array_like, shape(n,)
        Must be a positive integer.

    Returns
    -------
    b0_mat: ndarray, complex, shape(bins,n)
        Bunching factor along dimBunching for each slice.
    num_particles_slice: array_like, shape(bins,)
        Number of particles in each slice.
    mean_slice: array_like, shape(bins,)
        Mean of bin edges as sliced along dimSlice.
    """
    
    bin_name = "bin_" + dimSlice
    
    df_new = df.copy()
    df_new = an.binning(df_new,dimSlice,bins)
    b0_data = (df_new.groupby([bin_name])[dimBunching]
      .apply(lambda x: bunching_factor(x.to_numpy(), wavelengths)))
    b0_mat=np.stack(b0_data.to_numpy())
    
    num_particles_slice = df_new.groupby([bin_name])[dimBunching].count().to_numpy().reshape(bins,1)
    
    minVal = df_new[dimSlice].min(axis=0)
    maxVal = df_new[dimSlice].max(axis=0)
    bins_edges = np.linspace(minVal,maxVal,bins+1)
    w=2
    mean_slice = np.convolve(bins_edges, np.ones(w), 'valid') / w
    
    return (b0_mat,num_particles_slice,mean_slice)

if __name__ == '__main__':
    pass