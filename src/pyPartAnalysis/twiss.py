"""
For calculating and work with Twiss parameters.
"""

import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

import pyPartAnalysis.particle_accelerator_utilities as pau
import pyPartAnalysis.convert as cv

class Twiss:
    def __init__(self, **kwargs):
        """Instantiate Twiss parameters in a single dimension
        
        Either the geometric or normalized emittance must be set.
        
        Parameters
        ----------
        alpha : float
            alpha Twiss parameter
        beta : float
            beta Twiss parameter in meters
        emit : float
            geometric emittance
        emitn : float
            normalized emittance
        GBz : float, optional
            mean normalized momentum
        kinetic_energy : float, optional
            mean kinetic energy       
        """
        has_twiss = len({'alpha','beta'}.intersection(set(kwargs.keys())))==2
        has_emit = ('emit' in kwargs.keys()) != ('emitn' in kwargs.keys())
        
        assert has_twiss,'Twiss parameters not defined during instantiation.'
        assert has_emit,'Emittance not defined correctly during instantation.'
        
        self._parameters = kwargs

    def __str__(self):
        return str(self._parameters)
    
    def alpha(self):
        # return Twiss beta
        return self._parameters['alpha']
    
    def beta(self):
        # return Twiss beta
        return self._parameters['beta']
    
    def GBz(self):
        # return Twiss beta
        return self._parameters['GBz']
    
    def kinetic_energy(self):
        # return Twiss beta
        return self._parameters['kinetic_energy']
    
    def emit(self,**kwargs):
        # return geometric emittance
        
        #update if to include normalization factor
        self._parameters.update(kwargs)
        
        if 'emit' in self._parameters.keys():
            return self._parameters['emit']
        elif 'emitn' in self._parameters.keys():
            # if does not exit, normalize geometric emittance
            if 'GBz' in self._parameters.keys():
                norm_fact = self._parameters['GBz']
            elif 'kinetic_energy' in self._parameters.keys():
                kinetic_energy = self._parameters['kinetic_energy']
                gamma_mean = pau.KE2gamma(kinetic_energy)
                beta_mean = pau.gamma2beta(gamma_mean)
                norm_fact = gamma_mean*beta_mean
            else:
                raise Exception("Normalized Emittance defined but normalization factor not provided.\n")
            return self._parameters['emitn']/norm_fact
        else:
            raise Exception("Emittance not defined during instantiation")
            
    def emitn(self,**kwargs):
        # return normalized emittance
        
        # update parameters if new normalization value given
        self._parameters.update(kwargs)
        
        if 'emitn' in self._parameters.keys():
            return self._parameters['emitn']
        elif 'emit' in self._parameters.keys():
            # if does not exit, normalize geometric emittance
            if 'GBz' in self._parameters.keys():
                norm_fact = self._parameters['GBz']
            elif 'kinetic_energy' in self._parameters.keys():
                kinetic_energy = self._parameters['kinetic_energy']
                gamma_mean = pau.KE2gamma(kinetic_energy)
                beta_mean = pau.gamma2beta(gamma_mean)
                norm_fact = gamma_mean*beta_mean
            else:
                raise Exception("Geometric Emittance defined but normalization factor not provided.\n")
            return self._parameters['emit']*norm_fact
        else:
            raise Exception("Emittance not defined during instantiation")

    def rms_space(self):
        #calculate the RMS spatial extent
        return np.sqrt(self.emit()*self.beta())
    
    def rms_angle_waist(self):
        #calculate the RMS minimum angular extent
        return np.sqrt(self.emit()/self.beta())
    
    def gamma(self):
        # calculate the gamma Twiss parameter
        return (1+self.alpha()**2)/self.beta()                           

class Twiss_3d:
    def __init__(self,**kwargs):
        values = np.asarray(list(kwargs.values()))
        self._twiss_list = {}
        dimensions = {0:'x',1:'y',2:'z'}
        for dim in range(0,3):
            self._twiss_list[dimensions[dim]] = Twiss(**dict(zip(kwargs.keys(),values[:,dim])))
    
    def __str__(self):
        return(f'emitn = {self.emitn()}\n'
               f'emit = {self.emit()}\n'
               f'alpha = {self.alpha()}\n'
               f'beta = {self.beta()}\n'
               f'gamma = {self.gamma()}\n')
    
    def x(self):
        return self._twiss_list['x']
    
    def y(self):
        return self._twiss_list['y']
    
    def z(self):
        return self._twiss_list['z']
    
    def emitn(self,**kwargs):
        vals = []
        for dim in self._twiss_list.values():
            vals.append(dim.emitn(**kwargs))
        return vals
            
    def emit(self):
        vals = []
        for dim in self._twiss_list.values():
            vals.append(dim.emit())
        return vals

    def alpha(self):
        vals = []
        for dim in self._twiss_list.values():
            vals.append(dim.alpha())
        return vals
            
    def beta(self):
        vals = []
        for dim in self._twiss_list.values():
            vals.append(dim.beta())
        return vals
    
    def gamma(self):
        vals = []
        for dim in self._twiss_list.values():
            vals.append(dim.gamma())
        return vals    
    
def get_twiss_parameters(df,**kwargs):
    """Calculates the twiss parameters from distribution
    
    To consistantly calculate the emittance at non-relativistic 
    and relativistic energies, the normalized momentum should be 
    used for the angle parts of the phase space pairs.
    
    By default, ImpactT outputs distributions in this format, 
    however to convert ImpactZ outputs to the correct format, 
    the following conversion should be used:
    
        import pyPartAnalysis.convert as cv
        
        df_GB = cv.IMPACTZ_to_GB(df_IMPACTZ, 
                                 kinetic_energy, 
                                 rf_freq)
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution with normalized momentum:
        x(m),GBx,y(m),GBy,z(m),GBz

    Returns
    -------
    Twiss_3d
        Class instance that contains the Twiss parameters 
        in 3 dimensions as well as the emittances.
 
    """ 
    
    assert len({'GBx','GBy','GBz'}.intersection(set(df.columns)))==3,\
           'Distribution needs normalized momentum for Twiss calculation'
           
    df_copy = df.copy()
    kwargs_update = kwargs
   
    gammaL = pau.gammabeta2gamma(df_copy.GBx,df_copy.GBy,df_copy.GBz)
    gamma_mean = np.mean(gammaL)
    betaZ_mean = np.mean(df.GBz/gammaL)
    GBz = gamma_mean*betaZ_mean
    kwargs_update['GBz'] = GBz
      
    cov_mat = cv.GB_to_phase_space(df_copy).cov()
    emit = np.empty((0,3))
    for ii in range(0,3):
        emit = np.append(emit,np.sqrt(np.linalg.det(cov_mat.iloc[(2*ii):(2*ii+2),(2*ii):(2*ii+2)].to_numpy())))
    emitn = emit*GBz
    beta = np.diagonal(cov_mat,offset=0)[0:5:2]/emit
    alpha = -np.diagonal(cov_mat,offset=1)[0:5:2]/emit 
    
    # make vector of length 3 for distribution to each dimension
    kwargs_update = {k:np.repeat(v,[3]) for k,v in kwargs_update.items()}
    
    twiss_3d_dict = {'alpha':alpha,'beta':beta,'emitn':emitn}
    twiss_3d_dict.update(kwargs_update)
    
    return Twiss_3d(**twiss_3d_dict)
    
def get_twiss_z_slices(df,bins,**kwargs):
    """Caulate the alpha,beta,emit,z_mean for z slice of the distribution.
    
    Parameters
    ----------
    df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    bins : float
        Number of z slices
    
    Returns
    -------
    alpha : array_like, shape=(len(bins),)
        Twiss alpha parameter
    beta : array_like, shape=(len(bins),)
        Twiss beta parameter
    emit : array_like, shape=(len(bins),)
        Geometric Emittance
    z_mean : array_like, shape=(len(bins),)
        Mean z value of the slice
    """
    
    group_bins = pd.cut(df['z'],bins=bins)
    df_groups = df.groupby(group_bins)
    
    z_mean = df_groups['z'].mean().values
    
    a = (df_groups.apply(lambda x: get_twiss_parameters(x,**kwargs)))
    twiss_stack = np.hstack(a)
    
    emit = []
    emitn = []
    beta = []
    alpha = []

    for twiss3d in twiss_stack:
        emit.append(twiss3d.emit())
        emitn.append(twiss3d.emitn())
        beta.append(twiss3d.beta())
        alpha.append(twiss3d.alpha())
    
    data = {'z':z_mean,
            'emit':np.asarray(emit),
            'emitn':np.asarray(emitn),
            'beta':np.asarray(beta),
            'alpha':np.asarray(alpha)}
    
    return data
    
def make_sigma_mat(alpha,beta):
    """Creates a sigma matrix from the alpha and beta Twiss parameters
          
    Sigma Matrix is of the following form:
        
        |beta  -alpha|
        |-alpha gamma|  
        
    Parameters
    ----------
    alpha : array_like
        Alpha Twiss Parameter
    beta : array_like
        Beta Twiss Parameter
        
    Returns
    -------
    sigmax : ndarray, shape=(2,2)
        x sigma matrix
    sigmay : ndarray, shape=(2,2)
        y sigma matrix
    sigmaz : ndarray, shape=(2,2)
        z sigma matrix
    """
    
    gamma = (1+alpha**2)/beta
    sigmax = []
    sigmay = []
    sigmaz = []
    
    for b,a,g in zip(beta[:,0:3],alpha[:,0:3],gamma[:,0:3]):
        sigmax.append(np.array([[b[0],-a[0]],[-a[0],g[0]]]))
        sigmay.append(np.array([[b[1],-a[1]],[-a[1],g[1]]]))
        sigmaz.append(np.array([[b[2],-a[2]],[-a[2],g[2]]]))
    
    return sigmax,sigmay,sigmaz

def calculate_twiss_mismatch(sigma,sigma0):
    """Calulates the mismatch of the Twiss parameters.
        
        Note that the inputs are the sigma matrix, which has the form:
        sigma[1,1] = beta
        sigma[1,2] = -alpha
        sigma[2,1] = -alpha
        sigma[2,2] = gamma
        
        The mismatch parameter is greater than or equal to 1, with a 
        perfect match giving 1.
        
    Parameters
    ----------
    sigma : ndarray, shape=(2,2)
        Measured sigma matrix
    sigma_0 : ndarray, shape=(2,2)
        Design sigma matrix
        
    Returns
    -------
    float
        Twiss mismatch
    """
    
    bmag = []
    for sig in sigma:
        bmag.append(0.5*np.trace(sig @ np.linalg.inv(sigma0))) 
        
    return bmag

def twiss_mismatch_slice(twiss_3d_slice):
    sigma_x,sigma_y,sigma_z = make_sigma_mat(twiss_3d_slice['alpha'],twiss_3d_slice['beta'])
    ind_center = np.argmin(np.abs(twiss_3d_slice['z']))
    
    bmag_x= calculate_twiss_mismatch(sigma_x,sigma_x[ind_center])
    bmag_y= calculate_twiss_mismatch(sigma_y,sigma_y[ind_center])
    z = twiss_3d_slice['z']
    
    return z,bmag_x,bmag_y

def twiss_ellipse(alpha,beta,emit,xy=[0,0],scalex=0,scaley=0,**ell_kwargs):
    """Creates a ellipse patch from Twiss parameters
    
    Uses matplotlib.patches.Ellipse to create a patch from the Twiss parameters. 
    Patches can be added to an existing figure using
        
        fig, ax = plt.subplots()
        ax.add_patch(ell)
          
    Parameters
    ----------
    alpha : float
    beta : float (>0)
    emit : float (>0)
    xy : array_like, shape=(2,)
        Center of the ellipse
    scalex : int (>0)
        Scale of the ellipse along x axis (*10**scalex)
    scaley : int (>0)
        Scale of the ellipse along y axis (*10**scalex)
    **ell_kwargs
        kwarg for matplotlib.patches.Ellipse
        
    Returns
    -------
     matplotlib.patches.Ellipse
        Ellipse created from Twiss parameters
    """
    
    gamma = (1+alpha**2)/beta

    xWaist = np.sqrt(emit/beta);
    thetaMax = np.sqrt(emit*beta);
    angle = get_angle_twiss(alpha,beta)

    ell = Ellipse(xy=xy, width=2*xWaist*10**scalex, height=2*thetaMax*10**scaley, angle = 180+angle,**ell_kwargs)
    
    return ell

def twiss_ellipse_parametric(alpha,beta,emit,num_points=100,xy=[0,0]):
    """Creates a ellipse patch from Twiss parameters
    
    Uses matplotlib.patches.Ellipse to create a patch from the Twiss parameters. 
    Patches can be added to an existing figure using
        
        fig, ax = plt.subplots()
        ax.add_patch(ell)
        
    Note that xRMS is the actual RMS size of the beam, while thetaWaist 
    is the minimum size of the beam in angle space. The x dimension is 
    left alone and the correlation is added along the y axis.
          
    Parameters
    ----------
    alpha : float
    beta : float (>0)
    emit : float (>0)
    xy : array_like, shape=(2,)
        Center of the ellipse
        
    Returns
    -------
     x : ndarray
        x coordinates of ellipse
     y : ndarray
        y coordinates of ellipse
    """
    
    xRMS = np.sqrt(emit*beta);
    thetaWaist = np.sqrt(emit/beta);
    m = -alpha/beta
    t = np.linspace(0,2*np.pi,num_points);

    b = thetaWaist
    a = xRMS
    x = a*np.cos(t)
    y = b*np.sin(t)
    y = y + x*m;
    x += xy[0] 
    y += xy[1] 
    
    return x,y

def get_angle_twiss(alpha,beta):
    """Calculates the angle of the distribution in phase space
          
    Parameters
    ----------
    alpha : array_like
        Alpha Twiss Parameter
    beta : array_like
        Beta Twiss Parameter
        
    Returns
    -------
     array_like
        phase space angle
    """
    
    gamma = (1+alpha**2)/beta
    return np.rad2deg(0.5*np.arctan2(2*alpha,gamma-beta))
    
if __name__ == '__main__':
    pass
