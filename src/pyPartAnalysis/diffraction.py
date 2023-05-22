"""
Applies diffraction to pandas dataframes of particle data and related functions.
"""

import pandas as pd
import numpy as np
  
def read_diffraction_data_miller(filename):
    """Reads diffraction probabilities and Miller indices from file.
    
    The probability of diffraction into a reflection with a given Miller index 
    is calculated from simulation and printed out to a text file.
    
    The first two rows contain information about the simulation settings, e.g.
    
    9.000000000000000e+06   0.000000000000000e+00   0.000000000000000e+00 
    5.431000000000000e-10   5.431000000000000e-10   5.431000000000000e-10  

    indicates the simulation was for 9.0 MeV electrons with an x tilt of and a 
    y tilt of 0 (note that this presentation assumes that the we are dealing 
    with simple cubic crystals where the x and y tilts in inverse space can be 
    mapped directly to the x and y real space tilts). We also have lattice 
    constants of 5.43 angstroms in a,b, and c as indicated by the second row. 
    Again we are assuming simple cubic crystals, so this should correpsond to 
    the x, y, and z dimensions.
    
    The data after the first two rows is the probability followed by the 
    corresponding h,k,l Miller indices, e.g.

    7.455459782573727e-02     0     0     0 
    1.058955621407441e-01     0     4     0 
    4.505832360703049e-03     0     8     0 
    2.576638838330619e-03     0    12     0 
  
    The sum of the probabilities in the first column should be one.
    
    Parameters
    ----------
    filename : str
        File name for the diffraction data.
    
    Returns
    -------
    prob : array_like
        Probability of diffraction into the correpsonding Miller indices.
    dfKick : DataFrame
        h,k,l indices
    lattice : array_like, len=3
        a,b,c lattice constants
    """
    
    dfSimData = pd.read_csv(filename, nrows=2, header=None, sep = "\s+")
    lattice = dfSimData.loc[1].values.flatten().tolist();
    dfKick = pd.read_csv(filename, sep = "\s+", skiprows=2, header=None)
    dfKick.columns = ["prob","h","k","l"]
    prob = dfKick["prob"]
    dfKick = dfKick.drop(columns="prob")
    return (prob,dfKick,lattice)

def rotate_miller_indices(df_miller,angle):
    """Rotates crystal planes with respect to external x,y axes.
    
    The x,y axes of the simulated beamline or distribution may not align with 
    the crystal planes of the simulated diffraction data. This allows rotation 
    for an already performed simulation.
    
    Parameters
    ----------
    df_miller : DataFrame
        h,k,l Miller Indices in the original simulation frame.
    angle: float, {-180,180}
        Rotation of the crystal plane relative to tilt axes (degrees) 
        (ccw positive).
    
    Returns
    -------
    DataFrame
        h,k,l Miller Indices with the rotation applied.
    """
    
    df_miller_copy = df_miller.copy()
    
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    df_miller_copy[['h','k']] = df_miller_copy.loc[:,['h','k']].dot(R)
    
    df_miller_copy.loc[abs(df_miller_copy['h'])<=np.finfo(float).eps,'h'] = 0 
    df_miller_copy.loc[abs(df_miller_copy['k'])<=np.finfo(float).eps,'k'] = 0 
    
    return df_miller_copy

def apply_kick(df,prob,df_miller,lattice,mask=None,rotation=0):
    """Applies momentum kick to particles with a given probability
    
    Assumes a simple cubic crystal structure.
    
    Parameters
    ----------
    df : DataFrame
        Particle Distribution with normalized momentum
        x(m),GBx,y(m),GBy,z(m),GBz
    prob : array_like
        Probability of diffraction into the correpsonding Miller indices.
    dfKick : DataFrame
        h,k,l millerindices
    lattice : float
        a,b,c lattice constants
    mask : array_like, bool, optional
        Mask indicating whether row is to have kick applied, i.e. be 
        diffracted.
    rotation: float, {-180,180}
        Rotation of the crystal plane relative to tilt axes (degrees) 
        (ccw positive).
    
    Returns
    -------
    DataFrame
        Particle Distribution with normalized momentum
        x(m),GBX,y(m),GBy,z,GBz
        with the momentum kicks applied.
    """
    
    # Assume we are in the zeroth order Laue zone so we do not need to include z kicks
    hbar = 6.626*10**-34;     #J*s
    eMass = 9.10938*10**-31;  #kg
    c = 3*10**8;              #m/s

    #randomly assign to reflection
    kicknum = np.random.choice(len(df_miller), len(df), p=prob)
    
    #apply rotation to miller indices to get the correct kick
    df_miller = rotate_miller_indices(df_miller,rotation)
    
    #generate normalized momentum kick along x and y from the reflection indices
    df_miller = df_miller.div(lattice)*hbar/eMass/c

    if mask is None:
        mask = np.full((df.shape[0], 1), True)

    dfNew = df.copy()
    dfNew.loc[mask,'GBx'] = dfNew.loc[mask,'GBx'].add(df_miller.iloc[kicknum[mask],0].values)
    dfNew.loc[mask,'GBy'] = dfNew.loc[mask,'GBy'].add(df_miller.iloc[kicknum[mask],1].values)
    dfNew.loc[mask,'GBz']= np.sqrt(df.loc[mask,'GBx']**2 + df.loc[mask,'GBy']**2 + df.loc[mask,'GBz']**2  \
        - dfNew.loc[mask,'GBx']**2 - dfNew.loc[mask,'GBy']**2)    

    return dfNew
        


def kicked(df_old,df_new):
    """Checks if there is a change in normalized momentum
    
    Parameters
    ----------
    df_old : DataFrame
        Particle Distribution with normalized momentum
        x(m),GBX,y(m),GBy,z,GBz
    df_new : DataFrame
        Particle Distribution with normalized momentum
        x(m),GBX,y(m),GBy,z,GBz
    
    Returns
    -------
    array_like, bool
        False if there was no change in normalized momentum.
    """
    mask_kicked = ((df_new.loc[:,'GBx'] != df_old.loc[:,'GBx']) | 
                   (df_new.loc[:,'GBy'] != df_old.loc[:,'GBy']))
    
    return mask_kicked
    
    
if __name__ == "__main__":
    pass