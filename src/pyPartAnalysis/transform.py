"""
For removing correlations or mean shifts in distributions.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
    
def remove_phase_space_corr(df,dim,inds,dimremove=0):
    """Removes correlation using using the specified particles
    
    Removes a linear correlation in phase space for all particles 
    using a  subset specified by the indices

    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
        or with normalized momentum:
        x(m),GBX,y(m),GBy,z(m),GBz
    dim : {'x', 'y', 'z'} 
        Dimension along which correlation is removed
    inds : array_like
        Indices of the particles that are used to calculate the correlation
    dimremove : {0, 1}
        Indicates which dimension of phase space the correlation will be 
        removed from. 0 for the spatial or 1 for the compliment.

    Returns
    -------
    DataFrame
        Particle distribution with correlation removed
    """
    
    if(len({'xp','yp','delta'}.intersection(set(df.columns)))==3):
        dim_dict = {'x':['x','xp'],
                    'y':['y','yp'],
                    'z':['z','delta']}
    elif(len({'GBx','GBy','GBz'}.intersection(set(df.columns)))==3):
        dim_dict = {'x':['x','GBx'],
                    'y':['y','GBy'],
                    'z':['z','GBz']}
    
    ind_y = dimremove
    ind_x = np.mod(dimremove+1,2)
    
    df_new = df.copy()
    fit_df = df_new.loc[inds,:]
    y = fit_df[dim_dict[dim][ind_y]] - fit_df[dim_dict[dim][ind_y]].mean()
    y = y.values.reshape(-1,1)
    x = fit_df[dim_dict[dim][ind_x]] - fit_df[dim_dict[dim][ind_x]].mean()
    x = x.values.reshape(-1,1)
    
    model = LinearRegression()
    model.fit(x, y)
    
    x_total = df_new[dim_dict[dim][ind_x]]
    x_total = x_total.values.reshape(-1,1)
    
    y_predict = model.predict(x_total)
    y_original = df_new[dim_dict[dim][ind_y]]
    y_original = y_original.values.reshape(-1,1)
    y_uncorr = y_original - y_predict

    df_new.loc[:,dim_dict[dim][ind_y]] = y_uncorr
    
    return df_new

def remove_phase_space_corr_poly(df,dim,inds,dimremove=0,degree=1):
    """Removes polynomial correlation using using the specified particles
    
    Removes a polynomial correlation in phase space for all particles using a 
    subset specified by the indices

    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
        or with normalized momentum:
        x(m),GBX,y(m),GBy,z(m),GBz
    dim : {'x', 'y', 'z'} 
        Dimension to be plotted
    inds : array_like
        Indices of the particles that are used to calculate the correlation
    dimremove : {0, 1}
        Indicates which dimension of phase space the correlation will be 
        removed from. 0 for the spatial or 1 for the compliment.
    degree : int > 0
        Degree of the polynomial for which the correlation will be removed. 
        Defaults to a linear correlation.

    Returns
    -------
    DataFrame
        Particle distribution with correlation removed in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    """
    
    if(len({'xp','yp','delta'}.intersection(set(df.columns)))==3):
        dim_dict = {'x':['x','xp'],
                    'y':['y','yp'],
                    'z':['z','delta']}
    elif(len({'GBx','GBy','GBz'}.intersection(set(df.columns)))==3):
        dim_dict = {'x':['x','GBx'],
                    'y':['y','GBy'],
                    'z':['z','GBz']}
    
    ind_y = dimremove
    ind_x = np.mod(dimremove+1,2)
    
    df_new = df.copy()
    fit_df = df_new.loc[inds,:]
    y = fit_df[dim_dict[dim][ind_y]] - fit_df[dim_dict[dim][ind_y]].mean()
    y = y.values.reshape(-1,1)
    x = fit_df[dim_dict[dim][ind_x]] - fit_df[dim_dict[dim][ind_x]].mean()
    x = x.values.reshape(-1,1)
    
    poly = PolynomialFeatures(degree=degree)
    X_ = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(X_,y)
    
    x_total = df_new[dim_dict[dim][ind_x]]
    x_total = x_total.values.reshape(-1,1)
    x_total_ = poly.fit_transform(x_total)
    
    y_predict = model.predict(x_total_)
    y_original = df_new[dim_dict[dim][ind_y]]
    y_original = y_original.values.reshape(-1,1)
    y_uncorr = y_original - y_predict

    df_new.loc[:,dim_dict[dim][ind_y]] = y_uncorr.reshape(-1,)
    
    return df_new

def remove_2D_linear_corr(df,dim,inds,dimremove):
    """Removes multidimensional linear correlation using the specified particles
    
    Removes a multidimensional linear correlation in phase space for all particles using a 
    subset specified by the indices, e.g. xy correlation.

    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    dim : list {'x', 'xp', 'y', 'yp', 'z', 'delta'} 
        Dependent Variable
    inds : array_like
        Indices of the particles that are used to calculate the correlation
    dimremove : {'x', 'xp', 'y', 'yp', 'z', 'delta'} 
        Indicates which dimension of phase space the correlation will be 
        removed from.

    Returns
    -------
    DataFrame
        Particle distribution with correlation removed in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    """   
    
    df_new = df.copy()
    fit_df = df_new.loc[inds,:]
    y = fit_df.loc[:,dimremove] - fit_df.loc[:,dimremove].mean()
    y = y.values.reshape(-1,1)
    x = fit_df.loc[:,dim] - fit_df.loc[:,dim].mean()
    x = x.values.reshape(-1,1)
    
    poly = PolynomialFeatures(degree=2,interaction_only=True,include_bias = False)
    X_ = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(X_,y)
    
    x_total = df_new.loc[:,dim]
    x_total = x_total.values.reshape(-1,1)

    x_total_ = poly.fit_transform(x_total)
    
    y_predict = model.predict(x_total_)
    y_original = df_new.loc[:,dimremove]
    y_original = y_original.values.reshape(-1,1)
    y_uncorr = y_original - y_predict

    df_new.loc[:,dimremove] = y_uncorr
    
    return df_new

def remove_2D_corr_poly(df,dim1,dim2,inds,degree):
    """Removes multidimensional polynomial correlation using the specified particles
    
    Removes a multidimensional polynomial correlation in phase space for all particles using a 
    subset specified by the indices

    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    dim1 : {'x', 'xp', 'y', 'yp', 'z', 'delta'} 
        Dependent Variable
    dim2 : {'x', 'xp', 'y', 'yp', 'z', 'delta'} 
        Indicates which dimension of phase space the correlation will be 
        removed from.
    inds : array_like
        Indices of the particles that are used to calculate the correlation
    degree : int > 0
        Degree of the polynomial for which the correlation will be removed.

    Returns
    -------
    DataFrame
        Particle distribution with correlation removed in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    """   
    
    df_new = df.copy()
    fit_df = df_new.loc[inds,:]
    y = fit_df.loc[:,dim2] - fit_df.loc[:,dim2].mean()
    y = y.values.reshape(-1,1)
    x = fit_df.loc[:,dim1] - fit_df.loc[:,dim1].mean()
    x = x.values.reshape(-1,1)
    
    poly = PolynomialFeatures(degree=degree)
    X_ = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(X_,y)
    
    x_total = df_new.loc[:,dim1]
    x_total = x_total.values.reshape(-1,1)

    x_total_ = poly.fit_transform(x_total)
    
    y_predict = model.predict(x_total_)
    y_original = df_new.loc[:,dim2]
    y_original = y_original.values.reshape(-1,1)
    y_uncorr = y_original - y_predict

    df_new.loc[:,dim2] = y_uncorr
    
    return df_new

def add_phase_space_corr(df,dim,slope):
    """Adds correlation to phase space
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
        or with normalized momentum:
        x(m),GBX,y(m),GBy,z(m),GBz
    dim : {'x', 'y', 'z'} 
        Dimension to be plotted
    slope : float
        Units of z/delta or m/rad

    Returns
    -------
    DataFrame
        Particle distribution with correlation added in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    """
    
    if(len({'xp','yp','delta'}.intersection(set(df.columns)))==3):
        dim_dict = {'x':['x','xp'],
                    'y':['y','yp'],
                    'z':['z','delta']}
    elif(len({'GBx','GBy','GBz'}.intersection(set(df.columns)))==3):
        dim_dict = {'x':['x','GBx'],
                    'y':['y','GBy'],
                    'z':['z','GBz']}
    
    fit_df = df.copy()
    y = fit_df[dim_dict[dim][0]] - fit_df[dim_dict[dim][0]].mean()
    y = y.values.reshape(-1,1)
    x = fit_df[dim_dict[dim][1]] - fit_df[dim_dict[dim][1]].mean()
    x = x.values.reshape(-1,1)
    
    y_predict = slope*x
    y_corr = y - y_predict
    fit_df.loc[:,dim_dict[dim][0]] = y_corr
    
    return fit_df
    
def make_mean_zero(df):
    """Makes the mean of each column zero
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    
    Returns
    -------
    Dataframe
        df but the mean is zero 
    """
    
    # Removes mean from all columns of data frame
    df_new = df.copy()
    return df_new-df_new.mean()
    
def filter_by_counts(df,column,bins,cutoff):
    """Slices distribution along column and filters slices with low counts.
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    column : str
        Name of the column that will be sliced for filter
    bins : int
        Number of bins to slice the distribution
    cutoff : int
        The number of particles below which the slice is removed
    
    Returns
    -------
    Dataframe
        Filtered version of df
    """
    
    df_copy = df.copy()
    df_copy['bin'] = pd.cut(df_copy[column], bins=bins)
    bin_freq = df_copy.loc[:,[column,'bin']].groupby('bin',observed=True).count()
    df_copy = df_copy.loc[:,['delta','bin']].merge(bin_freq, 
                    on='bin', 
                    how='left',
                    suffixes=("_bin", 
                              "_bin_freq"))
    df_copy.columns = [column,'bin','freq']
    ind = df_copy.freq.values > cutoff
    
    return df.iloc[ind,:]
    
if __name__ == '__main__':
    pass
