"""
For writing pandas dataframe particle distributions and related info 
to csv files.
"""

import numpy as np
import pyPartAnalysis.diffraction as diff

def write_particle(df,filename,in_type = False,**kwargs):
    """Writes particle distribution to text file
    
    Note that this can be slow to write if the dataframe is multi-indexed; 
    increased write speeds can be seen be using only one index.
    Appears to be issue as of pandas 1.3.4.
    
    Adding the following lines improves the speed:
    
        df = df.reset_index(level=1)
        df = df.rename(columns={"stripe_id":"bins"}).
        
    If this is to be used as an input for ImpactT/Z with normalized momentum, 
    then the columns should be x(m),GBx,y(m),GBy,z(m),GBz in that order.
            
    Parameters
    ----------
    df : DataFrame
        Particle distribution.
    filename : str
        Name of the file to write the distribution.
    in_type : bool
        Is true if the distribution to be written is to be used as a 
        particle.in file. If true, then the number of particles is appended 
        to the beginning of the file.
    **kwargs
        Extra arguments for pandas.to_csv
    
    Returns
    -------
    pandas.to_csv
        See pandas.to_csv for more information.
    """
    
    if in_type:
        with open(filename, 'w') as f:
             f.write(f'{df.shape[0]}\n')
        mode= 'a'
    else:
        mode = 'w'
        
    return df.to_csv(filename, 
                     header=False, 
                     index=None, 
                     sep='\t',
                     float_format='%0.15e',
                     mode=mode,
                     **kwargs)
    
def write_GPT(df,filename,**kwargs):
    """Writes ASCII file for use with GPT
            
    Includes the column names as the header when writing to file.
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution.
    filename : str
        Name of the file to write the distribution.
    **kwargs
        Extra arguments for pandas.to_csv
    
    Returns
    -------
    pandas.to_csv
        See pandas.to_csv for more information.
    """

    return df.to_csv(filename, 
                     header=True, 
                     index=None, 
                     sep='\t',
                     float_format='%0.15e',
                     **kwargs)
    
def write_stripe_with_index(df,filename,**kwargs):
    """Write stripe data to file
            
    First column is particle id and second column is stripe id.
    
    Parameters
    ----------
    df : DataFrame
        Stripe data where index is the particle id and the column is 
        the stripe id.
    filename : str
        Name of the file to write the distribution.
    **kwargs
        Extra arguments for pandas.to_csv
    
    Returns
    -------
    pandas.to_csv
        See pandas.to_csv for more information.
    """

    return df.to_csv(filename, 
                     header=False, 
                     index=True, 
                     sep='\t',
                     float_format='%d',
                     **kwargs)
    
def write_stripe(stripe_num,filename,**kwargs):
    """Write stripe id to file
            
    Only the stripe id is written to file. It is assumed that the row number 
    (starting from 1) is the particle id.
    
    Parameters
    ----------
    stripe_num : array_like
        Stripe data where index is the particle id and the column is 
        the stripe id.
    filename : str
        Name of the file to write the distribution.
    **kwargs
        Extra arguments for numpy.savetxt
    """
    
    np.savetxt(filename, 
               np.asarray(stripe_num, dtype=np.int64),
               fmt='%i',
               **kwargs)
    
def write_kicked(df_original,df_kicked,filename,**kwargs):
    """Writes whether particle recieved transverse momentum kick to file
            
    In the written file, the first column correpsonds to the particle 
    id, while the second row indicates whether the particle was 
    kicked (1) or not (0).
    
    Parameters
    ----------
    df_original : DataFrame
        Particle distribution with normalized momentum:
        x(m),GBx,y(m),GBy,z(m),GBz.
    df_kicked : DataFrame
        Particle distribution with normalized momentum:
        x(m),GBx,y(m),GBy,z(m),GBz.
        Distribution to which pyPartAnalysis.diffraction.apply_kick is applied.
    filename : str
        Name of the file to write the distribution.
    **kwargs
        Extra arguments for pandas.to_csv
    
    Returns
    -------
    pandas.to_csv
        See pandas.to_csv for more information.
    """
    
    mask_kicked = diff.kicked(df_original,df_kicked)
    mask_kicked = mask_kicked.astype(int)
    
    return mask_kicked.to_csv(filename, 
                              header=False, 
                              index=True, sep='\t',
                              float_format='%d',
                              **kwargs)