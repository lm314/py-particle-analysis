"""
Calculates quantities from beam data.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import pyPartAnalysis.particle_accelerator_utilities as pau
import pyPartAnalysis.bunching as bun
import pyPartAnalysis.twiss as twi
import pyPartAnalysis.convert as cv

def sigma_to_emit(s_matrix,dim):
    """Calculates emittance from sigma matrix
       
    Calculates the emittance in the individual dimensions, 
    the transverse 4D emittance, or the 6D emittance. Note that if the sigma 
    matrix includes the normalization factor as is default for fort.32, 
    that the output will be the normalized emittance. Otherwise, it will 
    be the geometric emittance.
    
    Parameters
    ----------
    s_matrix : ndarray, shape=(N,6,6)
        Sigma matrix, i.e. the covariance matrix for a particle 
        distribution.
    dim: str, {'x', 'y', 'z','xy','xyz'}
        
    Returns
    -------
    array_like, shape=(N,)
        Normalized or geometric emittance.
    """    
    
    index = {'x':np.s_[:,0:2,0:2],
             'y':np.s_[:,2:4,2:4],
             'z':np.s_[:,4:6,4:6],
             'xy':np.s_[:,0:4,0:4],
             'xyz':np.s_[:,0:6,0:6]}
    
    
    return np.sqrt(np.linalg.det(s_matrix[index[dim]]))

def interplane_coupling_ratio(x_emit,y_emit,xy_emit):
    """Calculates interplane coupling ratio in the xy phase space
    
    Calculates the interplane coupling ratio given by:
        
        r = emit_x*emit_y/emit_4Dxy - 1
        
    Emittances can be normalized or geometric as long as all 
    three quantities are the same type. The greater the value,
    the greater the interplane coupling, with 0 indicating no 
    coupling.
    
    Parameters
    ----------
    x_emit : array_like, shape=(N,)
        2D x emittance
    y_emit : array_like, shape=(N,)
        2D y emittance
    xy_emit : array_like, shape=(N,)
        4D xy emittance
        
    Returns
    -------
    array_like, shape=(N,)
        Ratio of interplane coupling
    """   
    
    return x_emit*y_emit/xy_emit-1

def xy_eigen_emit(sigma):
    """Calculates the transverse eigen-emittances from the sigma matrix
    
    If the sigma matrix contains the normalization factor, then the returned 
    value is the normalized eigen-emittance. Otherwise, it is the geometric 
    eigen-emittance.
    
    Parameters
    ----------
    sigma : ndarray, shape=(N,6,6)
        Sigma matrix, i.e. the covariance matrix for a particle 
        distribution
        
    Returns
    -------
    emit1: array_like, shape=(N,)
        First transverse eigen-emittance
    emit2: array_like, shape=(N,)
        Second transverse eigen-emittance
    """   
    
    C = sigma[:,0:4,0:4];
    J = np.array([[0,1,0,0],[-1,0,0,0],[0,0,0,1],[0,0,-1,0]])
    
    trCJ2 = np.trace(np.linalg.matrix_power(np.matmul(C,J), 2),axis1=1,axis2=2)
    arg = np.sqrt(trCJ2**2 - 16*np.linalg.det(C))
    
    emit1 = 0.5*np.sqrt(-trCJ2 - arg)
    emit2 = 0.5*np.sqrt(-trCJ2 + arg)
    
    return (emit1,emit2)

def int_vorticity_area(sigma):
    """Calculates vorticity integrated over the area times the area 
    from the sigma matrix, i.e. WA.
    
    If the sigma matrix includes a normalization 
    component, the returned value WA will be 
    normalized as well. To remove normalization use WA/GBz.
    
    Parameters
    ----------
    sigma : ndarray, shape=(N,6,6)
        Sigma matrix, i.e. the covariance matrix for a particle 
        distribution
        
    Returns
    -------
    array_like, shape=(N,)
        Integrated vorticity
    """ 
        
    #using the default values from fort.32 will give the normalized integrated vorticity
    x2 = sigma[:,0,0]
    y2 = sigma[:,2,2]
    xy = sigma[:,0,2]
    xyp = sigma[:,0,3]
    xpy = sigma[:,2,1]
    xxp = sigma[:,0,1]
    yyp = sigma[:,2,3]
    
    WA = y2*xyp - x2*xpy + xy*(xxp - yyp)

    return WA

def beam_area(sigma):
    """Calculates beam area from the sigma matrix
    
    Parameters
    ----------
    sigma : ndarray, shape=(N,6,6)
        Sigma matrix, i.e. the covariance matrix for a particle 
        distribution
        
    Returns
    -------
    array_like, shape=(N,)
        RMS transverse area of the distribution
    """ 
    
    x2 = sigma[:,0,0]
    y2 = sigma[:,2,2]
    xy = sigma[:,0,2]
    
    A = np.sqrt(x2*y2-xy**2)
      
    return A

def int_vorticity(sigma):
    """Calculates integrated vorticity W from the sigma matrix
    
    If the sigma matrix includes a normalization 
    component, the returned integrated vorticity will be 
    normalized as well. To remove normalization use W/GBz, 
    where GBz can be gotten from the average normalized z 
    momentum in fort.26.
    
    Parameters
    ----------
    sigma : ndarray, shape=(N,6,6)
        Sigma matrix, i.e. the covariance matrix for a particle 
        distribution
        
    Returns
    -------
    array_like, shape=(N,)
        Vorticity
    """ 
    
    return int_vorticity_area(sigma)/beam_area(sigma)

def transverse_bin(df,num_bin_x,num_bin_y):
    """Splits particles into x and y bins
    
    Splits particles into categories based on the x and y positions. 
    The positions of the bins are equidistant and span from the min to the 
    max particle position in each dimension.

    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    num_bin_x : int
        Must be a positive integer.
    num_bin_y : int
        Must be a positive integer.

    Returns
    -------
    Dataframe
        A copy of df but with bins_x and bin_y appended.
    """
    
    df_new = df.copy()
    labels_x = np.arange(0, num_bin_x)
    labels_y = np.arange(0, num_bin_y)
    
    dimension = "x"
    minx = df_new[dimension].min(axis=0)
    maxx = df_new[dimension].max(axis=0)
    bins_x = np.linspace(minx,maxx,num_bin_x+1)
    df_new["bins_x"] = pd.cut(df_new[dimension], bins=bins_x, labels=labels_x, include_lowest=True)
    df_new["bins_x"] = df_new['bins_x'].cat.codes
    
    dimension = "y"
    miny = df_new[dimension].min(axis=0)
    maxy = df_new[dimension].max(axis=0)
    bins_y = np.linspace(miny,maxy,num_bin_y+1)
    df_new["bins_y"] = pd.cut(df_new[dimension], bins=bins_y, labels=labels_y, include_lowest=True)
    df_new["bins_y"] = df_new['bins_y'].cat.codes
    
    return df_new
    
def transverse_bin_counts(df,bins):
    """Plot the bunching factor
    
    Calculates the number of particles in each transverse bin

    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    bins : array_like, shape=(2,)
        Number of bins in each dimension

    Returns
    -------
    count_area : ndarray, shape=(bins[0],bins[1])
        Number of particles in each transverse bins
    """    
    count_data = (transverse_bin(df,bins[0],bins[1]).groupby(['bins_x','bins_y'],observed=True)).count().x
    count_area = np.zeros((bins[0],bins[1]))
    for idx, df_select in count_data.groupby(level=[0, 1],observed=True):
        count_area[idx[1],idx[0]] = df_select.values

    return count_area    

def binning(df,dim,num_bin):
    """Bins Dataframe based on column
    
    Adds a column to the dataframe named "bin_" plus 
    the specified column.
                     
    Parameters
    ----------
    df : DataFrame
        Arbitrary dataframe
    dim: str
        Column of df along which binning will occur
    num_bin : int
        Number of bins to divide along
        
    Returns
    -------
    DataFrame
        Copy of the initial dataframe with an additional 
        column for the binning names "bin_" plus 
        the specified column.
    """
    
    df_copy = df.copy()

    labels = np.arange(0, num_bin)
    bin_name = "bin_" + dim;
    
    minVal = df_copy[dim].min(axis=0)
    maxVal = df_copy[dim].max(axis=0)
    bins = np.linspace(minVal,maxVal,num_bin+1)
    df_copy[bin_name] = pd.cut(df_copy[dim], bins=bins, labels=labels, include_lowest=True)
    
    return df_copy

def get_transport_matrix_SVD(input_df,output_df):
    """Calculates the linear transport matrix for 6d phase space using SVD
    
    Uses sklearn.linear_model.LinearRegression to fit the 6D transport matrix. 
    Is generally slower than get_transport_matrix as the sklearn package uses 
    SVD for the calculation which is more robust but slower.

    Parameters
    ----------
    input_df : DataFrame
        Initial particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    output_df : DataFrame
        Final particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P

    Returns
    -------
    reg.coef_ : float
        6x6 linear transport matrix
    reg : sklearn.linear_model.LinearRegression
        Fitted linear model
    """
    
    idx = input_df.index.intersection(output_df.index)
    idx_id = [ii[0] for ii in idx]
    reg = LinearRegression().fit(input_df.loc[idx_id],output_df.loc[idx_id])
    return [reg.coef_,reg] 

def get_transport_matrix(input_df,output_df):
    """Calculates the linear transport matrix for 6d phase space
    
    Uses numpy least squares to calculate the 6D linear transport matrix. 
    Assumes input_df and output_df are sorted in ascending order by their 
    particle id values.

    Parameters
    ----------
    input_df : DataFrame
        Initial particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    output_df : DataFrame
        Final particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P

    Returns
    -------
    x[0].T : float
        6x6 linear transport matrix
    x : numpy.linalg.lstsq
        All outputs from the function
    """
    
    idx = input_df.index.get_level_values('id').intersection(output_df.index.get_level_values('id'))
    if(len(idx)!=output_df.shape[0] or len(idx)!=input_df.shape[0]):
        x = np.linalg.lstsq(input_df.loc[input_df.index.get_level_values('id').isin(idx),:],
                            output_df.loc[output_df.index.get_level_values('id').isin(idx),:],
                            rcond=None)
    else:
        x = np.linalg.lstsq(input_df,
                            output_df,
                            rcond=None)
        
    return [x[0].T,x]

def get_phase_advance(df_norm,dims = ['x','y','z']):
    """Gets the angle of the distribution
    
    Calculates the betatron phase angle from the particle 
    distribution in normalized coordinates as output by 
    normalized_coord
             
    Parameters
    ----------
    df : DataFrame
        Particle distribution in normalized units:
        x,xp,y,yp,z,deltaGamma/gamma~deltaP/P
    dims: List {'x', 'y', 'z'}
    
    List : 
    
    Returns
    -------
    ndarray, shape(num_bin_x,num_bin_y,n)
        Bunching factor at the transverse positions
    """
    
    dim_dict = {'x':['x','xp'],
            'y':['y','yp'],
            'z':['z','delta']}
    
    df_copy = df_norm.copy()
    
    df_copy = df_copy - df_copy.mean(axis=0)
    deg = np.zeros([df_copy.shape[0],len(dims)])
    for ii,dim in enumerate(dims):
        deg[:,ii] = np.arctan2(df_copy.loc[:,dim_dict[dim][1]],
                               df_copy.loc[:,dim_dict[dim][0]])
    
    return pd.DataFrame(deg, columns = dims)
    
def get_poly_area(df,dim,deg,num_pixels=[32,32],cut_off=1):
    """Polynomial Fit to particles in x and y bins
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    dim : {'x', 'y', 'z'} 
        Phase Space dimension to be fitted.
    deg : int
        Polynomial degree of fitting
    num_pixels : array_like, shape=(2,1)
        number of bins in the x and y directions
    cutoff : int
        The number of particles below which the fitting does not occur
     
    
    Returns
    -------
    numpy.polyfit, shape(num_pixels)
        polynomial fit to phase space
    """
    
    df_new = transverse_bin(df,num_pixels[0],num_pixels[1])
    
    poly_data = (df_new.groupby(['bins_x','bins_y'],observed=True)
               .apply(lambda x: get_poly_fit(x,dim,deg,cut_off)))
    
    poly = np.zeros((num_pixels[0],num_pixels[1],deg+1),dtype=float)

    for idx, df_select in poly_data.groupby(level=[0, 1],observed=True):
        nonzero = bun.get_iterable(np.isnan(df_select.values[0]))
        for idz,cond in enumerate(nonzero):
            if(~cond):
                poly[idx[0],idx[1],idz] = df_select.values[0][idz]
        
    return poly  

def get_poly_fit(df,dim,deg,cut_off):
    """Slices distribution along column and filters slices with low counts.
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    dim : {'x', 'y', 'z'} 
        Phase Space dimension to be fitted.
    deg : int
        Polynomial degree of fitting
    cutoff : int
        The number of particles below which the fitting does not occur
    
    Returns
    -------
    numpy.polyfit
        polynomial fit to phase space
    """

    if(len(df)>cut_off):
        dim_dict = {'x':['x','xp'],
                   'y':['y','yp'],
                   'z':['z','delta']}
            
        x = df.loc[:,dim_dict[dim][1]] - df.loc[:,dim_dict[dim][1]].mean()
        x = x.values.flatten()
        y = df.loc[:,dim_dict[dim][0]] - df.loc[:,dim_dict[dim][0]].mean()
        y = y.values.flatten()
        poly = np.polyfit(x=x,y=y,deg=deg)
    else:
        poly = np.zeros(deg+1,)
    return poly

def current(df: pd.DataFrame,num_bin: int,total_charge: float,Bz: float = []):
    """Calculates the instantaneous current of particle distribution
    
    Assumes that the z coordinate of the dataframe is in physical units.
    Note that if the distribution does not have normalized momentum 
    (i.e. GBx,GBy,GBz) and the Bz is not given as an argument, then 
    it is assumed that the distribution is traveling at the speed of light.
    Bz is ignored if the distribution has normalized momentum.
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units.
    num_bin : int
        Number of bins used slicing the distribution
    total_charge : float
        Total charge in the distribution (C).
    Bz : float, optional
        Normalized z velocity. Should be less than or equal to 1.

    Returns
    -------
    z_mean: numpy.ndarray
        Z positions of the binned distribution (m).
    current: numpy.ndarray
        Instantaneous currents of the binned distribution (A).   
    """ 
    temp = binning(df,dim='z',num_bin=num_bin)
    z_mean = np.convolve(np.linspace(np.min(temp.z),np.max(temp.z),num_bin+1),
                         np.ones(2)/2, 
                         mode='valid')
    deltaZ = z_mean[1] - z_mean[0]
    
    # Assign Bz if not given as input
    if len({'GBx','GBy','GBz'}.intersection(set(df.columns)))==3:
        gammaL = pau.gammabeta2gamma(temp.GBx,temp.GBy,temp.GBz)
        gamma_mean = np.mean(gammaL)
        Bz = np.mean(df.GBz/gammaL)
    elif not Bz:
        Bz = 1
    
    assert Bz <= 1, f'Bz should be less than or equal to 1 but is {Bz:0.3f}.'
    
    c = 299792458
    vz = c*Bz    

    counts = temp.groupby('bin_z',observed=True).count().z.values
    current = counts/temp.count().z*total_charge/deltaZ*vz
   
    return z_mean, current

def sliceMean(value,weights,axis=0):
    """
    Computes the weighted mean of values across slices.

    This function calculates the weighted average of a 1D array of values, 
    typically representing a quantity computed per slice of a particle distribution.

    Parameters
    ----------
    values : array-like
        Array of values to average, one per slice.
 
    weights : array-like
        Array of weights corresponding to each slice. Typically the relative 
        number of particles in each slice. Sum is 1.

    Returns
    -------
    float
        Weighted mean of the input values.
    """

    return np.average(value,weights=weights,axis=axis)

def sliceCovariance(value1,value2,weights):
    """
    Computes the weighted covariance between two slice-wise quantities.

    This function calculates the weighted covariance between two arrays of 
    slice-wise values. It is useful for evaluating correlations between 
    slice-averaged quantities or intra-slice statistical properties.

    Parameters
    ----------
    value1 : array-like
        First array of values, one per slice.

    value2 : array-like
        Second array of values, one per slice.

    weights : array-like
        Array of weights corresponding to each slice. Typically the relative 
        number of particles in each slice. Sum is 1.

    Returns
    -------
    float
        Weighted covariance between the two input arrays.
    """

    mean1 = sliceMean(value1,weights)
    mean2 = sliceMean(value2,weights)
    return sliceMean((value1-mean1)*(value2-mean2),weights)

def get_decomposed_normalized_emittance(df, bins,transverse_dimension='x'): 
    """
    Computes slice-averaged and correlated emittance metrics from a particle distribution.

    This function processes a particle distribution DataFrame by binning it along the 
    longitudinal coordinate `z`, computing slice-wise statistics, and evaluating 
    several emittance-related quantities. It uses normalized momentum coordinates 
    and assumes the input DataFrame includes columns for position and normalized 
    momentum. The decomposition is described in https://arxiv.org/abs/1509.04765.
    Note that the squared sum of the returned components is equal to the square 
    of the projected normalized emittance.

    Parameters
    ----------
    df : pandas.DataFrame
        Particle Distribution with normalized momentum
        x(m),GBX,y(m),GBy,z,GBz
 
    bins : array-like
        Bin edges for slicing the distribution along the `z` axis.
        
    transverse_dimension : {'x', 'y'}, optional
        Specifies the transverse dimension for which to compute the emittance 
        decomposition. Default is 'x'.


    Returns
    -------
    dict
        Dictionary containing the following emittance-related metrics:
 
        - epsilon_mean_slice : float
            Weighted mean of the normalized emittance across slices.

        - epsilon_r : float
            Emittance growth associated with Twiss parameter mismatch 
            between slices.

        - epsilon_int : float
            Emittance growth term associated with a linear correlation 
            between transverse dimension centroid and z.
 
        - epsilon_parallel : float
            Emittance growth term associated with a non-linear correlation 
            between transverse dimension centroid and z.
    """

    if transverse_dimension == 'x':
        emit_coord = 0
        dim_ind = [0,1]
        dim_coord = ['x','xp']
    elif transverse_dimension == 'y':
        emit_coord = 1
        dim_ind = [2,3]
        dim_coord = ['y','yp']
    
    # group beam based off of z slices
    group_bins = pd.cut(df['z'], bins=bins)
    df_groups = df.groupby(group_bins,observed=True)
    
    # fraction of the beam in each z slice
    sliceWeight = df_groups.size().values / df.shape[0]

    # get twiis parameters for each slice
    twissSlice = twi.get_twiss_z_slices(df, bins)
    
    df_ps = cv.GB_to_phase_space(df)
    df_ps_groups = df_ps.groupby(group_bins,observed=True)
    
    # covariance matrix of each z slice
    cov_slice = df_ps_groups.cov().to_numpy().reshape(-1, 6, 6).transpose(1, 2, 0)
    
    # mean values of each z slice
    mean_slice = df_ps_groups.mean(numeric_only=True)

    # mean normalized emittance of the z slices
    epsilon_mean_slice = sliceMean(twissSlice["emitn"][:, emit_coord], weights=sliceWeight)

    # covariance of the mean weighted by counts in slice
    cov_ux_ux = sliceCovariance(mean_slice[dim_coord[0]].values, mean_slice[dim_coord[0]].values, weights=sliceWeight)
    cov_uxp_uxp = sliceCovariance(mean_slice[dim_coord[1]].values, mean_slice[dim_coord[1]].values, weights=sliceWeight)
    cov_ux_uxp = sliceCovariance(mean_slice[dim_coord[0]].values, mean_slice[dim_coord[1]].values, weights=sliceWeight)

    epsilon_r2 = (
        sliceCovariance(twissSlice["emitn"][:, emit_coord], twissSlice["emitn"][:, emit_coord], weights=sliceWeight)
        - sliceCovariance(cov_slice[dim_ind[0], dim_ind[0], :], cov_slice[dim_ind[1], dim_ind[1], :], weights=sliceWeight)
        + sliceCovariance(cov_slice[dim_ind[0], dim_ind[1], :], cov_slice[dim_ind[0], dim_ind[1], :], weights=sliceWeight)
    )

    epsilon_int2 = (
        sliceMean(cov_slice[dim_ind[0], dim_ind[0], :], weights=sliceWeight) * cov_uxp_uxp
        + sliceMean(cov_slice[dim_ind[1], dim_ind[1], :], weights=sliceWeight) * cov_ux_ux
        - 2 * sliceMean(cov_slice[dim_ind[0], dim_ind[1], :], weights=sliceWeight) * cov_ux_uxp
    )

    epsilon_parallel2 = cov_ux_ux * cov_uxp_uxp - cov_ux_uxp ** 2

    return {
        "epsilon_mean_slice": epsilon_mean_slice,
        "epsilon_r": np.sqrt(epsilon_r2),
        "epsilon_int": np.sqrt(epsilon_int2),
        "epsilon_parallel": np.sqrt(epsilon_parallel2),
    }

if __name__ == '__main__':
    pass
