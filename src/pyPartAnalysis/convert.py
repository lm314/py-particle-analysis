"""
Converts pandas dataframes from one format to another.
"""

import numpy as np

import pyPartAnalysis.particle_accelerator_utilities as pau
import pyPartAnalysis.twiss as tw
import pyPartAnalysis.convert as cv

def GB_to_IMPACTZ(df, rf_freq,charge=-0.160000409601E-18,qoverm = -0.195692801440E-05,id = []):
    """Converts from normalized momentum to ImpactZ format
    
    This converts from the ImpactT format with normalized momentum to the 
    ImpactZ format. Note that omega=2*pi*rf_freq for the transverse 
    real space normalization.
    
    Parameters
    ----------
    df : DataFrame
        Particle Distribution with normalized momentum:
        x(m),GBX,y(m),GBy,z(m),GBz
       
    
    Returns
    -------
    DataFrame
        ImpactZ Particle distribution:
        x/c*omega,GBx,x/c*omega,GBy,phase(radian),deltaGamma/gamma~deltaP/P 

    """
    
    ps_df = df.copy()
    
    col_names = ['x','px','y','py','phase','delta','q_over_m','charge','id']
    
    c = 299792458;
    omega = 2*np.pi*rf_freq;
    gamma = pau.gammabeta2gamma(df.GBx,df.GBy,df.GBz)
    gamma0 = np.mean(gamma)
    beta0 = pau.gamma2beta(gamma0)
    betaZ = df.GBz/gamma
    
    transNorm = c/omega;
    transAngle = gamma0*beta0;
    
    if len(id) > 0:
        ps_df['id'] = id
    else:
        ps_df['id'] = np.arange(1,df.shape[0]+1)
        
        
    ps_df['q_over_m'] = qoverm
    ps_df['charge'] = charge
    
    ps_df['x'] = ps_df['x']/transNorm
    ps_df['y'] = ps_df['y']/transNorm
    
    ps_df['px'] = ps_df['GBx']
    ps_df['py'] = ps_df['GBy']
    ps_df['delta'] = gamma0 - gamma
    
    ps_df['phase'] = -omega/betaZ/c*ps_df['z']
    
    ps_df = ps_df[col_names]
    
    return ps_df


def GB_to_phase_space(df):
    """Converts from normalized momentum to phase space
    
    This takes the normalized momentum coordinates that are output by 
    IMPACT-T and converts them to physical phase space coordinates. 
    Note that the use of radians for x' and y' assumes the paraxial 
    approximation, i.e. x' ~ theta_x.
    
    Parameters
    ----------
    df : DataFrame
        Particle Distribution with normalized momentum
        x(m),GBX,y(m),GBy,z,GBz
    
    Returns
    -------
    DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z(m),deltaGamma/gamma~deltaP/P
    """
    
    df_copy = df.copy()
    
    df_copy['xp'] = df_copy['GBx']/df_copy['GBz']
    df_copy['yp'] = df_copy['GBy']/df_copy['GBz']
    betagamma2 = df_copy.GBx**2+df_copy.GBy**2+df_copy.GBz**2;
    gamma = np.sqrt(1+betagamma2);
    gamma_mean = np.mean(gamma)
    beta_mean = pau.gamma2beta(gamma_mean)
    df_copy['delta'] = (gamma - gamma_mean)/gamma_mean
    df_copy = df_copy.drop(columns=['GBx', 'GBy', 'GBz'])
    
    df_copy = df_copy[['x','xp','y','yp','z','delta']] 
    
    return df_copy

def phase_space_to_GB(df,kinetic_energy):
    """Converts from phase space to normalized momentum
    
    This takes the physical phase space coordinates and converts them to 
    coordinates with normalized momentum.
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P        
    
    Returns
    -------
    DataFrame
        Particle Distribution with normalized momentum:
        x(m),GBX,y(m),GBy,z(m),GBz

    """
    
    df_copy = df.copy()
    
    gamma_mean = pau.KE2gamma(kinetic_energy)
    gamma = df_copy['delta']*gamma_mean + gamma_mean
    beta = pau.gamma2beta(gamma)
    
    Bx = beta*np.sin(df['xp'])
    By = beta*np.sin(df['yp'])
    Bz = np.sqrt(beta**2 - Bx**2 - By**2)
    
    df_copy['GBx'] = gamma*Bx
    df_copy['GBy'] = gamma*By
    df_copy['GBz'] = gamma*Bz
    
    df_copy = df_copy.drop(columns=['xp', 'yp', 'delta'])
    df_copy = df_copy[['x','GBx','y','GBy','z','GBz']]
    
    return df_copy
    
def IMPACTT_to_sigma_matrix(df):
    """Convert contents of fort.32 into sigma matrix format
    
    Converts sigma matrix components from IMPACT-T units to 
    natural units.
    
    Parameters
    ----------
    filename : DataFrame, shape=(N,24)
        DataFrame from read.read_fort for fort.32
    
    Returns
    -------
    ndarray, shape=(N,6,6)
        Sigma matrix, i.e. the covariance matrix for a particle 
        distribution. Note that the determinant gives the 
        normalized emittance.  
    """
    
    temp =  [[df.s11,df.s12,df.s13,df.s14,df.s15,df.s16],\
            [df.s12,df.s22,df.s23,df.s24,df.s25,df.s26],\
            [df.s13,df.s23,df.s33,df.s34,df.s35,df.s36],\
            [df.s14,df.s24,df.s34,df.s44,df.s45,df.s46],\
            [df.s15,df.s25,df.s35,df.s45,df.s55,df.s56],\
            [df.s16,df.s26,df.s36,df.s46,df.s56,df.s66]]
    
    # Convert from IMPACT-T units to natural units
    temp =  np.asarray(temp)*df.lscale.values[np.newaxis, np.newaxis,:]
    
    # Make shape into (N,6,6) for use with numpy det function
    temp = np.swapaxes((np.swapaxes(temp,0,1)),0,2)
    return temp

def phase_space_to_normalized_coord(df_GB):
    """Normalizes phase space distributions to spherical space
    
    This takes phase space coordinates and converts them 
    to normalized coordinates where the betatron phase change 
    is more easily observed, i.e. where alpha=0 and beta=1. 
    Normilization is the following:
    
        w(phi) = u/sqrt(beta) = a*cos(phi)
        
        dw(phi)/dphi = sqrt(beta)*u' + alpha/sqrt(beta)*u
                     = -a*sin(phi)
                     
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units with normalized momentum GB:
        x(m),GBx,y(m),GBy,z(m),GBz
    
    Returns
    -------
    DataFrame
        Particle distribution in normalized units:
        x,xp,y,yp,z,deltaGamma/gamma~deltaP/P
    """
    
    
    
    df = cv.GB_to_phase_space(df_GB)
    df_norm = df.copy()
    
    twiss3d = tw.get_twiss_parameters(df_GB)
    df_norm.x = df.x/np.sqrt(twiss3d.x().beta())
    df_norm.xp = np.sqrt(twiss3d.x().beta())*df.xp + df.x*twiss3d.x().alpha()/np.sqrt(twiss3d.x().beta()) 
    df_norm.y = df.y/np.sqrt(twiss3d.y().beta())
    df_norm.yp = np.sqrt(twiss3d.y().beta())*df.yp + df.y*twiss3d.y().alpha()/np.sqrt(twiss3d.y().beta()) 
    df_norm.z = df.z/np.sqrt(twiss3d.z().beta())
    df_norm.delta = np.sqrt(twiss3d.z().beta())*df.delta + df.z*twiss3d.z().alpha()/np.sqrt(twiss3d.z().beta()) 
    
    return df_norm
    
def IMPACTZ_to_phase_space(df, kinetic_energy, rf_freq):
    """Converts from IMPACTZ to phase space
    
    This takes the normalized coordinates that are output by IMPACTZ and 
    converts them to physical phase space coordinates. See read_IMPACTZ 
    for the normalizations. Note that the conversion assumes the paraxial 
    approximation as xp calculated from removing the normalization Px/mc 
    is assumed to be the angle when calculating betaX and betaY.
    
    Parameters
    ----------
    df : DataFrame
        Particle Distribution in normalized coordinates
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
    
    c = 299792458;
    omega = 2*np.pi*rf_freq;
    gamma0 = pau.KE2gamma(kinetic_energy)
    beta0 = pau.gamma2beta(gamma0)
    
    transNorm = c/omega;
    transAngle = gamma0*beta0;
    
    ps_df = df.drop(['q_over_m','charge'], axis=1)
    ps_df['x'] = transNorm*ps_df['x']
    ps_df['y'] = transNorm*ps_df['y']
    ps_df['px'] = ps_df['px']/transAngle
    ps_df['py'] = ps_df['py']/transAngle
    gamma = gamma0-ps_df['delta'];
    beta = np.sqrt(1-1/gamma**2);
    betaX = beta*np.sin(ps_df['px'])
    betaY = beta*np.sin(ps_df['py'])
    betaZ = np.sqrt(beta**2-betaX**2-betaY**2)
    
    ps_df['phase'] = -betaZ*c*ps_df['phase']/2/np.pi/rf_freq
    ps_df['delta'] = -ps_df['delta']/gamma0/beta0**2
    
    ps_df.rename(columns = {'phase':'z','px':'xp','py':'yp'}, inplace = True)
    
    return ps_df
    
def IMPACTZ_to_GB(df, kinetic_energy, rf_freq):
    """Converts from IMPACTZ to IMPACTT
    
    This takes the normalized coordinates that are output by IMPACTZ and 
    converts them to physical phase space coordinates with normalized 
    momentum. See read_IMPACTZ for the normalizations.
    
    Parameters
    ----------
    df : DataFrame
        Particle Distribution in normalized coordinates
    kinetic_energy : float
        The mean kinetic energy of the distribution in IMPACTZ
    rf_freq : float
        The RF frequency used in IMPACTZ for normalization
    
    Returns
    -------
    DataFrame
        Particle distribution in physical units:
        x(m),GBx,y(m),GBy,z(m),GBz
    """
    
    c = 299792458;
    omega = 2*np.pi*rf_freq;
    gamma0 = pau.KE2gamma(kinetic_energy)
    beta0 = pau.gamma2beta(gamma0)
    
    transNorm = c/omega;
    transAngle = gamma0*beta0;
    
    ps_df = df.drop(['q_over_m','charge'], axis=1)
    ps_df['x'] = transNorm*ps_df['x']
    ps_df['y'] = transNorm*ps_df['y']
    gamma = gamma0-ps_df['delta'].values;
    ps_df['delta'] = np.sqrt(gamma**2-1-ps_df['px'].values**2-ps_df['py'].values**2)
    print(gamma[1])
    print(ps_df['px'].values[1])
    betaZ = ps_df['delta'].values/gamma
    
    ps_df['phase'] = -betaZ*c*ps_df['phase']/2/np.pi/rf_freq
    
    ps_df.rename(columns = {'phase':'z','px':'GBx','py':'GBy','delta':'GBz'}, inplace = True)
    
    return ps_df
    
if __name__ == '__main__':
    pass
