"""
For splitting particle distributions into spatial stripes.
"""

import numpy as np
import pandas as pd
import math

def filter_stripe(df,stripeNum):
    """Filters out particles not in specified stripes
    
    Parameters
    ----------
    df : DataFrame
        Particle Distribution in normalized coordinates with two-level 
        indexing, the first being the particle id and the second being 
        the stripe id
    stripeNum : list
        The stripe numbers of the particles to be returned
    
    Returns
    -------
    DataFrame
        Particle distribution in physical units for the specified stripes
    """
    
    dist_df = df.copy()
    stripe_mask = np.in1d(df.index.get_level_values('stripe_id'),stripeNum)
    return dist_df.loc[stripe_mask,:]

def add_stripe_id(df,stripe_df):
    """Adds stripe_id for particles
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    stripe_df : DataFrame
        StripeNum and endIndex
    
    Returns
    -------
    Dataframe
        Copy of df but with stripe id appended        
    """
    
    num_particles = stripe_df['endIndex'] - stripe_df['endIndex'].shift(1,fill_value=0)
    df_new = df.copy()
    df_new['stripe_id'] = np.repeat(stripe_df['stripeNum'].values, num_particles)[df_new.index-1]
    return df_new   

def gen_stripe(df, feature_width, dim):
    """Adds stripe number to dataframe for stripes of specified width
    
    The number of stripes is calculated using the difference in the maximum 
    and minimum, dividing by the feature_width and rounding to the largest 
    integer.
    
    The bounds of the stripes is then calculated using numpy.linspace with the 
    bounds given by the max and min and the number of edges given by the 
    previously calculated number of stripes, i.e. the stripes are not 
    necessarily orientated in relation to the origin.
    
    Adds a new column with the stripe number with the name given by stripe_dim, 
    e.g. stripe_x, stripe_y, stripe_z.
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution where the space coordinates are in 
        physical units, e.g. x(m), y(m),z(m).
    feature_width : float
        Width of the stripes in meters.
    dim : {'x', 'y', 'z'} 
        Dimension along which stripe number is generated.
    
    Returns
    -------
    Dataframe
        Copy of df but with the stripe_dim column added.  
    """
    
    df_copy = df.copy()
    
    minx = df_copy[dim].min(axis=0)
    maxx = df_copy[dim].max(axis=0)

    numStripes = math.ceil(abs(maxx-minx)/feature_width)
    labels = np.arange(1, numStripes+1)

    bins = np.linspace(minx,maxx,numStripes+1)

    stripe_id = 'stripe_' + dim
    df_copy[stripe_id] = pd.cut(df_copy[dim], bins=bins, labels=labels, include_lowest=True)
    df_copy[stripe_id] = df_copy[stripe_id].cat.codes + 1

    return df_copy

def even_odd_stripe_mask(stripe_id):
    """Gives mask for even-odd stripes
    
    Even stripes have a value of 1 and odd a value of 0.

    Parameters
    ----------
    stripe_id : array_like
        Stripe id numbers.
    
    Returns
    -------
    array_like, bool
        Mask for even-odd stripes.  
    """
    
    mask = stripe_id % 2 == 0
    return mask

def stripe_mean(df,dim):
    """Returns mean position of each stripe along a dimension.
    
    Takes the mean of particles grouped by stripe number.
       
    Parameters
    ----------
    df : DataFrame
        Multi-level indexed particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
        The 0 level index is the particle id or 'id', while 
        the 1 level index is the stripe id or 'stripe_id'.
    dim : {'x', 'y', 'z'} 
        Dimension along which mean position is calculated.
    
    Returns
    -------
    numpy.ndarray
        Mean position of each stripe.  
    """

        
    return df.groupby(['stripe_id'],observed=True)[dim].mean().values
    
def stripe_count(df):
    """Returns number of particles in each stripe along a dimension.
    
    Takes the mean of particles grouped by stripe number.
       
    Parameters
    ----------
    df : DataFrame
        Multi-level indexed particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
        The 0 level index is the particle id or 'id', while 
        the 1 level index is the stripe id or 'stripe_id'.
    
    Returns
    -------
    numpy.ndarray
        Number of particles in each stripe. 
    """
    return df.groupby(['stripe_id'],observed=True)['x'].count().values
    
def stripe_moving_mean(val,w=2):
    """Calculates the moving average
           
    Parameters
    ----------
    val : array_like
        Array over which moving average is taken.
    w : int 
        Size of window over which moving average is taken.
    
    Returns
    -------
    numpy.ndarray
        Array of moving average.
    """
    return np.convolve(val, np.ones(w), 'valid') / w
    
def stripe_pitch(df,dim):
    """Calculates pitch between succesive stripes.
    
    The pitch is the center-to-center distance of the stripes.
       
    Parameters
    ----------
    df : DataFrame
        Multi-level indexed particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
        The 0 level index is the particle id or 'id', while 
        the 1 level index is the stripe id or 'stripe_id'.
    dim : {'x', 'y', 'z'} 
        Dimension along which stripe pitch is calculated.
    
    Returns
    -------
    numpy.ndarray
        Pitch of successive stripes.  
    """   
    
    mean_pos = stripe_mean(df,dim)
    mean_pos_diff = stripe_moving_mean(mean_pos,w=2)
    pitch = np.diff(mean_pos)
    
    return (mean_pos_diff,pitch)
    
def get_particle_id_from_stripe(df,stripe_id):
    """Return particle id's for the stripes.
    
    Parameters
    ----------
    df : DataFrame
        Multi-level indexed particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
        The 0 level index is the particle id or 'id', while 
        the 1 level index is the stripe id or 'stripe_id'.
    stripe_id : array_like 
        Stripe id's for which the particle id's are returned.
    
    Returns
    -------
    numpy.ndarray
        Particle id's. 
    """   
    return filter_stripe(df,stripe_id).index.get_level_values('id')
     
def get_center_stripe_id(df,dim):
    """Return particle id's for the stripes.
    
    Parameters
    ----------
    df : DataFrame
        Multi-level indexed particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
        The 0 level index is the particle id or 'id', while 
        the 1 level index is the stripe id or 'stripe_id'.
    stripe_id : array_like 
        Stripe id's for which the particle id's are returned.
    
    Returns
    -------
    numpy.ndarray
        Particle id's. 
    """   
    
    mean_pos = stripe_mean(df,dim)
    stripe_id_center = np.argmin(np.abs(mean_pos))
    
    # select stripe with more counts for better fitting
    if filter_stripe(df,stripe_id_center).count().x < filter_stripe(df,stripe_id_center+1).count().x:
        stripe_id_center = stripe_id_center + 1
        
    return stripe_id_center

def stripe_statistics(df,dim,remove_edge = 0.1):
    """Compute mean and standard deviation of successive stripe pitches.
    
    Parameters
    ----------
    df : DataFrame
        Multi-level indexed particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
        The 0 level index is the particle id or 'id', while 
        the 1 level index is the stripe id or 'stripe_id'.
    dim : {'x', 'y', 'z'} 
        Dimension along which stripe counts is calculated.
    remove_edge : float
        Percentage of the max extent used to calculate the statistics.
        A value between 0 and 1.
    
    Returns
    -------
    mean_pitch: numpy.float64
        Mean pitch between succesive stripes.
    std_pitch: numpy.float64
        Standard deviation of the pitch values between succesive stripes.
    """   
        
    mean_pos,pitch = stripe_pitch(df,dim)
    
    # remove particles near edges where counts are smaller are std is larger
    mask_edges = np.abs(mean_pos) < remove_edge*np.max(mean_pos)
    mean_pitch = np.abs(pitch[mask_edges].mean())
    std_pitch = pitch[mask_edges].std()
    
    return (mean_pitch,std_pitch)

if __name__ == '__main__':
    pass