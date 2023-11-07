"""
Reads particle distributions and data output by IMPACTZ/T
"""

import os
import pandas as pd
import numpy as np

import pyPartAnalysis.convert as cv

def read_fort_t(filename,file_extension=[],on_bad_lines='skip'):    
    """Reads text file of IMPACTT standard fort outputs
    
    Reads in the fort files with the following extensions (and information):
        fort.18
            reference particle information
        fort.24
            X RMS size info
        fort.25
            Y RMS size info
        fort.26
            Z RMS size info
        fort.27
            Maximum amplitude info
        fort.28
            Load balance and loss diagnostic
        fort.29
            Cubic roots of 3rd moments of beam dist
        fort.30
            Square roots of wth moments of beam dist
            
    Parameters
    ----------
    filename : str
        Name of the text file containing the distribution data, 
        usually fort.**
    file_extension : str
        Extension of the filetype, including the period at the beginning.
        Needed if the name of the file differs from the ones defined in 
        the IMPACT User manual.
    on_bad_lines : str
        keyword argument for read_csv. Determines how to handle lines that do 
        not have the same number of columns as the initial part of the file.
    
    Returns
    -------
    DataFrame
        Columns depend on the fort file extension  
    """
    
    col_names = {'.18':['t','dist','gamma','KE','beta','Rmax','energy_deviation'],
                 '.24':['t','z','avgX','rmsX','avgPx','rmsPx','alphaX','rmsEmitN'],
                 '.25':['t','z','avgY','rmsY','avgPy','rmsPy','alphaY','rmsEmitN'],
                 '.26':['t','z','rmsZ','avgPz','rmsPz','alphaZ','rmsEmitN'],
                 '.27':['t','z','maxX','maxPx','maxY','maxPy','maxZ','maxPz'],
                 '.28':['t','z','minPE','maxPE','totalPart'],
                 '.29':['t','z','X','Px','Y','Py','Z','Pz'],
                 '.30':['t','z','X','Px','Y','Py','Z','Pz'],
                 '.32':['t','z','lscale','s11','s12','s13','s14','s15','s16',\
                                               's22','s23','s24','s25','s26',\
                                                     's33','s34','s35','s36',\
                                                           's44','s45','s46',\
                                                                 's55','s56',\
                                                                       's66']}
    
    if not file_extension:
        _, file_extension = os.path.splitext(filename)
    
    if file_extension in col_names.keys():
        df = pd.read_csv(filename,header=None, delimiter=r"\s+",names=col_names[file_extension],on_bad_lines=on_bad_lines)
    else:
        df = pd.DataFrame()
        
    return df
    
def read_fort_z(filename):    
    """Reads text file of IMPACTZ standard fort outputs
    
    Reads in the fort files with the following extensions (and information):
        fort.18
            reference particle information
        fort.24
            X RMS size info
        fort.25
            Y RMS size info
        fort.26
            Z RMS size info
        fort.27
            Maximum amplitude info
        fort.28
            Load balance and loss diagnostic
        fort.29
            Cubic roots of 3rd moments of beam dist
        fort.30
            Square roots of wth moments of beam dist
        fort.31
            Number of particles for each charge state
            
    Note rmsZ = c*rmsPhase*(pi/180)/(2*pi)/rf_freq
         rmsZemitN = c*rmsPhase*(pi/180)/(2*pi)/rf_freq
    
    Parameters
    ----------
    filename : str
        Name of the text file containing the distribution data, 
        usually fort.**
    
    Returns
    -------
    DataFrame
        Columns depend on the fort file extension  
    """

    
    col_names = {'.18':['dist','absPhase','gamma','KE','beta','Rmax'],
                 '.24':['z','avgX','rmsX','avgPx','rmsPx','alphaX','rmsEmitN'],
                 '.25':['z','avgY','rmsY','avgPy','rmsPy','alphaY','rmsEmitN'],
                 '.26':['z','avgPhase','rmsPhase','avgPz','rmsPz','alphaZ','rmsEmitN'],
                 '.27':['z','maxX','maxPx','maxY','maxPy','maxPhase','maxDelta'],
                 '.28':['z','minPE','maxPE','totalPart'],
                 '.29':['z','X','Px','Y','Py','phase','delta'],
                 '.30':['z','X','Px','Y','Py','phase','delta'],
                 '.31':['z','numPart']}
    
    _, file_extension = os.path.splitext(filename)
    
    if file_extension in col_names.keys():
        df = pd.read_csv(filename,header=None, delimiter=r"\s+",names=col_names[file_extension])
    else:
        df = pd.DataFrame()
        
    return df

def read_GB(filename,skiprows=0):
    """Reads IMPACTZ/T distributions with normalized momentum 
    
    Reads 6 column particle distribution text file with normalized momentum.
    The header can be skipped as particle.in files can start with the number 
    of particles in the file.
    
    Newer versions of ImpactT on the development branch input/output 9 columns,
    where the extra 3 columns are macroparticle charge/mass, charge/macroparticle,
    and global id.
    
    Parameters
    ----------
    filename : str
        Name of the text file containing the distribution data, 
        usually fort.100*
    skiprows : int {0,1}
        Number of rows to skip in distribution file. Should be 0 
        for particle distribution generated by ImpactT and 1 for 
        distributions that have the number of particles as the 
        first line (i.e. particle.in files).
        
    Returns
    -------
    DataFrame
        Particle distribution with normalized momentum:
        x [m] ,GBx [px/gamma/beta], y [m], GBy [py/gamma/beta], 
        z [m], GBz [pz/gamma/beta]
        If 9 columns, then the extra 3 columns are:
        q_macro (units of e charge)/E_rest (eV) 
        q_macro (C)
        particle id
        
    """
    df = pd.read_csv(filename,
                     header=None, 
                     delimiter=r"\s+",
                     skiprows=skiprows)
                     
    if(df.shape[1]==6):
        df.columns = ['x','GBx','y','GBy','z','GBz']
    elif(df.shape[1]==9):
        df.columns = ['x','GBx','y','GBy','z','GBz','q_over_m','charge','id']
        df = df.sort_values(by='id')
        df = df.set_index('id')
    
    return df

def read_GB_to_phase_space(filename,**kwargs):
    """Reads partially normalized IMPACTZ/T into a physical units
    
    Particles are indexed first by the particle id ("id")
    
    Parameters
    ----------
    filename : str
        Name of the text file containing the distribution data, 
        usually fort.100* or fort.40 or fort.50.
    **kwargs
        Extra arguments for read_fort_norm_momentum
    
    Returns
    -------
    DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P    
    """
    
    df = (read_GB(filename,**kwargs)
            .sort_index()
            .pipe(cv.GB_to_phase_space))
    
    df.index = np.arange(1, df.shape[0]+1)
    df.index.name = 'id'
    
    return df

def read_GB_to_phase_space_stripes(filename,stripe_df,**kwargs):
    """Reads partially normalized IMPACTZ/T into a physical units with multi-indexing
    
    Particles are indexed first by the particle id ("id"), then by the 
    stripe id ("stripe_id"). The stripe_df format is different from the 
    one used in the IMPACTZ_analysis version of this function.
    
    Parameters
    ----------
    filename : str
        Name of the text file containing the distribution data, 
        usually fort.100*
    stripe_df : DataFrame
        StripeNum
    **kwargs
        Extra arguments for read_norm
    
    Returns
    -------
    DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P    
    """
    
    df = read_GB_to_phase_space(filename,**kwargs)

    df['stripe_id'] = stripe_df.loc[stripe_df.index.isin(df.index.get_level_values('id')),:]
    df = df.set_index('stripe_id',append=True)

    return df

# def read_IMPACTZ_stripes(filename,stripe_df,kinetic_energy,rf_freq):
    # """Reads IMPACTZ into a physical units with multi-indexing
    
    # Particles are indexed first by the particle id ("id"), then by the 
    # stripe id ("stripe_id")
    
    # Parameters
    # ----------
    # filename : str
        # Name of the text file containing the distribution data, 
        # usually fort.100*
    # stripe_df : DataFrame
        # StripeNum
    
    # Returns
    # -------
    # DataFrame
        # Particle distribution in physical units:
        # x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P    
    # """
    
    # df = (read_IMPACTZ(filename)
            # .sort_index()
            # .pipe(convert.IMPACTZ_to_phase_space,kinetic_energy,rf_freq))

    # df['stripe_id'] = stripe_df.loc[stripe_df.index.isin(df.index.get_level_values('id')),:]
    # df = df.set_index('stripe_id',append=True)

    # return df

def read_IMPACTZ(filename):
    """Reads text file of IMPACTZ particle distributions 
    
    This takes the normalized coordinates that are output by IMPACTZ and 
    converts them to physical phase space coordinates. See read_IMPACTZ_dist 
    for normalized coordinates
    
    Parameters
    ----------
    filename : str
        Name of the text file containing the distribution data, 
        usually fort.100*
    
    Returns
    -------
    DataFrame
        Particle distribution in normalized units:
        x*c/omega,xp*beta*gamma,y*c/omega,yp*beta*gamma,phase(rad),
        -deltaGamma i.e gamma0-gamma, charge/mass, charge, particle id    
    """
    
    col_names = ['x','px','y','py','phase','delta','q_over_m','charge','id']
    
    df = (pd.read_csv(filename,header=None, delimiter=r"\s+",names=col_names)
              .astype({'id': int})
              .set_index('id'))
    return df
 
def read_IMPACTZ_to_phase_space(filename,kinetic_energy,rf_freq):
    """Reads IMPACTZ into a physical units with multi-indexing
    
    Particles are indexed first by the particle id ("id"), then by the 
    stripe id ("stripe_id")
    
    Parameters
    ----------
    filename : str
        Name of the text file containing the distribution data, 
        usually fort.100*
    stripe_df : DataFrame
        StripeNum
    
    Returns
    -------
    DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P    
    """
    
    df = (read_IMPACTZ(filename)
            .sort_index()
            .pipe(cv.IMPACTZ_to_phase_space,kinetic_energy,rf_freq))

    return df
    
# IMPACTZ_analysis
 
def read_IMPACTZ_to_phase_space_stripes(filename,stripe_df,kinetic_energy,rf_freq):
    """Reads IMPACTZ into a physical units with multi-indexing
    
    Particles are indexed first by the particle id ("id"), then by the 
    stripe id ("stripe_id")
    
    Parameters
    ----------
    filename : str
        Name of the text file containing the distribution data, 
        usually fort.100*
    stripe_df : DataFrame
        StripeNum and endIndex
    kinetic_energy : float
        The mean kinetic energy of the distribution in IMPACTZ
    rf_freq : float
        The RF frequency used in IMPACTZ for normalization
    
    Returns
    -------
    DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P    
    """
    
    df = (read_IMPACTZ(filename)
                   .sort_index()
                   .pipe(cv.IMPACTZ_to_phase_space,kinetic_energy,rf_freq))    
    
    df['stripe_id'] = stripe_df.loc[stripe_df.index.isin(df.index.get_level_values('id')),:]
    df = df.set_index('stripe_id',append=True)
    
    return df  

def read_stripe_data(filename):
    """Reads text files of stripe id and particle id
    
    The expected format of the text file is two columns, with the first giving 
    the stripe number and the second giving the particle id of the the last 
    particle in that stripe
    
    Parameters
    ----------
    filename : str
        Name of the text file containing the stripe information
    
    Returns
    -------
    DataFrame
        StripeNum and endIndex
    """
        
    return pd.read_csv(filename, delimiter=",")

def read_slice_info(filename): 
    """Reads text file of IMPACTZ slice information
    
    Reads the slice information for the beam at the specified location in the 
    IMPACTZ lattice. 
    
    Parameters
    ----------
    filename : str
        Name of the text file containing the distribution data, 
        usually fort.20*
    
    Returns
    -------
    DataFrame
        The returned columns are:
                bunch length (m)
                # of particles per slice
                current per slice (A)
                X normalized Emittance per slice (m-rad)
                Y normalized Emittance per slice (m-rad)
                dE/E 
                uncorrelated energy spread per slice (eV)
                x_rms fo each slice (m)
                y_rms fo each slice (m)
                X mismatch factor
                Y mismatch factor 
    """
    col_names = ['bunch_length','num_part','current','x_emit_n','y_emit_n',
                 'dE_E','uncorr_E_spread','x_rms','y_rms','x_mismatch',
                 'y_mismatch']
    
    df = pd.read_csv(filename,header=None, delimiter=r"\s+",names=col_names)

    return df

def read_kicked(filename):
    """Reads text file indicating whether a particle recieved a momentum kick
    
    The text file has two columns: 
        1. Particle id
        2. 1 or 0 with 1 indicating the particle was kicked.
    
    Parameters
    ----------
    filename : str
        Name of the text file containing the kicked data.
    
    Returns
    -------
    DataFrame
        The index is the particle id and the kicked column is a boolean 
        indicating if the particle received a kick (True).
    """
    
    df_kicked = pd.read_csv(filename,
             header=None, 
             delimiter=r"\s+",
             names=['id','kicked'])
    df_kicked['kicked'] = df_kicked['kicked'].astype(bool)
    df_kicked.index = df_kicked.id
    df_kicked = df_kicked.drop(columns = 'id')
    return df_kicked

def read_stripe(filename):
    """Reads text files of stripe id
    
    If there is one column of data in the file it is assumed that the row of 
    the file correpsonds to the particle ID, starting from 1. If there are two 
    columns of data, then the first column is the particle index and the 
    second column in the stripe number.
    
    Parameters
    ----------
    filename : str
        Name of the text file containing the stripe information
    
    Returns
    -------
    DataFrame
        Index is particle id and stripe_id is the stripe number of the particle.
    """
    
    stripe_df = pd.read_csv(filename,header=None, delimiter=r"\s+",dtype='Int64')
    
    assert stripe_df.shape[1] in [1,2], \
            f'{filename} should have 1 or 2 columns but has {stripe_df.shape(1)}.' 
    
    if stripe_df.shape[1] == 2:
        stripe_df.columns = ['id','stripe_id']
        stripe_df.index = stripe_df['id']
        stripe_df = stripe_df.drop(columns = 'id')
    elif stripe_df.shape[1] == 1:
        stripe_df.index = np.arange(1, stripe_df.shape[0]+1)
        stripe_df.columns = ['stripe_id']
        
    return stripe_df
    
if __name__ == '__main__':
    pass
