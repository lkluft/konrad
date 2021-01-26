# -*- coding: utf-8 -*-
"""
Created on Tuesday, May 22, 2019
Author: Bjorn Stevens (bjorn.stevens@mpimet.mpg.de)
"""
#
import numpy as np
from scipy import interpolate, optimize

gravity = 9.8076

cpd     = 1006.
Rd      = 287.05

Rv      = 461.53    # IAPWS97 at 273.15
cpv     = 1865.01   # ''
cpl     = 4219.32   # '' 
cpi     = 2096.70   # ''
lv0     = 2500.93e3 # ''
lf0     =  333.42e3 #''

eps1     = Rd/Rv
eps2     = Rv/Rd -1.


P0      = 100000.  # Standard Pressure [Pa]
T0      = 273.15   # Standard Temperature [K]
sd00    = 6783     # Dry air entropy at P0, T0
sv00    = 10321    # Water vapor entropy at P0, T0
PvC     = 22.064e6 # Critical pressure [Pa] of water vapor
TvC     = 647.096  # Critical temperature [K] of water vapor
TvT     = 273.16   # Triple point temperature [K] of water
PvT     = 611.655

def thermo_input(x, xtype='none'):
    
    x = np.asarray(x).flatten()
    scalar_input = False
    if x.ndim == 0:
        x = x[None]  # Makes x 1D
        scalar_input = True

    if (xtype == 'Kelvin' and x.max() < 100 ): x = x+273.15
    if (xtype == 'Celcius'and x.max() > 100 ): x = x-273.15
    if (xtype == 'Pascal' and x.max() < 1200): x = x*100.
    if (xtype == 'kg/kg'  and x.max() > 1.0) : x = x/1000.
    if (xtype == 'meter'  and x.max() < 10.0): print('Warning: input should be in meters, max value less than 10, not corrected')

    return x, scalar_input

def es(T,state='liq'):
    """ Returns the saturation vapor pressure of water over liquid or ice, or the minimum of the two,
    depending on the specificaiton of the state variable.  The calculation follows Wagner and Pruss (2002)
    for saturation over planar liquid, and Wagner et al., 2011 for saturation over ice.  The choice
    of formulation was based on a comparision of many many formulae, among them those by Sonntag, Hardy, 
    Romps, Murphy and Koop, and others (e.g., Bolton) just over liquid. The Wagner and Pruss and Wagner
    formulations were found to be the most accurate as cmpared to the IAPWS standard for warm temperatures,
    and the Wagner et al 2011 form is the IAPWS standard for ice.
    >>> es([273.16,290.])
    [611.65706974 1919.87719485]
    """
 
    def esif(T):
        a1 = -0.212144006e+2
        a2 =  0.273203819e+2
        a3 = -0.610598130e+1
        b1 =  0.333333333e-2
        b2 =  0.120666667e+1
        b3 =  0.170333333e+1
        theta = T/TvT
        return PvT * np.exp((a1*theta**b1 + a2 * theta**b2 + a3 * theta**b3)/theta)

    def eslf(T):
        vt = 1.-T/TvC
        return PvC * np.exp(TvC/T * (-7.85951783*vt + 1.84408259*vt**1.5 - 11.7866497*vt**3 + 22.6807411*vt**3.5 - 15.9618719*vt**4 + 1.80122502*vt**7.5))
    
    x,  scalar_input = thermo_input(T, 'Kelvin')

    if (state == 'liq'):
        es = eslf(x)
    if (state == 'ice'):
        es = esif(x)
    if (state == 'mxd'):
        es = np.minimum(esif(x),eslf(x))

    if scalar_input:
        return np.squeeze(es)
    return es

def desdT(T,state='liq'):
    """ Returns the numerically differentiated saturation vapor pressure over planar water
    or ice""
    >>> desdT([273.16,290.])
    [ 44.43669338 121.88180492]
    """   
    x,  scalar_input = thermo_input(T, 'Kelvin')
    dx = 0.01; xp = x+dx/2; xm = x-dx/2
    return (es(xp)-es(xm))/dx
   
def phase_change_enthalpy(Tx,fusion=False):
    """ Returns the enthlapy [J/g] of vaporization (default) of water vapor or 
    (if fusion=True) the fusion anthalpy.  Input temperature can be in degC or Kelvin
    >>> phase_change_enthalpy(273.15)
    2500.8e3
    """  

    TC, scalar_input = thermo_input(Tx, 'Celcius')
    if (fusion):
        el = lf0 + (cpl-cpi)*TC
    else:
        el = lv0 + (cpv-cpl)*TC
 
    if scalar_input:
        return np.squeeze(el)
    return el
    
def pp2mr(pv,p):
    """ Calculates mixing ratio from the partial and total pressure
    assuming both have same units and returns value in units of kg/kg.
    checked 20.03.20
    """

    pv,  scalar_input1 = thermo_input(pv) # don't specify pascal as this will wrongly corrected
    p ,  scalar_input2 = thermo_input(p ,'Pascal')
    scalar_input = scalar_input1 and scalar_input2

    ret = eps1*pv/(p-pv)
    if scalar_input:
        return np.squeeze(ret)
    return ret
   
def mr2pp(mr,p):
    """ Calculates partial pressure from mixing ratio and pressure, if mixing ratio
    units are greater than 1 they are normalized by 1000.
    checked 20.03.20
    """

    mr,  scalar_input1 = thermo_input(mr, 'kg/kg')
    p ,  scalar_input2 = thermo_input(p , 'Pascal')
    scalar_input = scalar_input1 and scalar_input2

    ret = mr*p/(eps1+mr)
    if scalar_input:
        return np.squeeze(ret)
    return ret

def get_mse(T,Q,Z):                 
    """ Calculates moist static energy, with vapor dependent specific heats, and temperature
    dependent vaporization enthalpy.  Input for Q and 
    >>> mse(0,10.1000.)
    311407.29
    """

    T, scalar_input1 = thermo_input(T, 'Kelvin')
    Q, scalar_input2 = thermo_input(Q, 'kg/kg')
    Z, scalar_input3 = thermo_input(Z,'meter')
    scalar_input = scalar_input1 and scalar_input2 and scalar_input3

    mse = (cpd + (cpv-cpd)*Q)*T +  phase_change_enthalpy(T)*Q + gravity*Z

    if scalar_input:
        return np.squeeze(mse)
    return(mse)

def get_theta_e(T,P,qt,formula = 'isentrope'):
    """ Calculates equivalent potential temperature. The default is the real, saturated,
    isentropic theta-e corresponding to Eq. 2.42 in the Clouds and Climate book.  IF the
    formula 'bolton' or 'pseudo' is specified it reverts to the pseudo-adiabat of Bolton
    checked 19.03.20
    """

    TK,  scalar_input1 = thermo_input(T, 'Kelvin')
    PPa, scalar_input2 = thermo_input(P, 'Pascal')
    qt,  scalar_input3 = thermo_input(qt,'kg/kg')
    scalar_input = scalar_input1 and scalar_input2 and scalar_input3

    if (formula == 'bolton' or formula == 'pseudo'):  # Follows Bolton
        rs = pp2mr(es(TK),PPa)
        rv = qt/(1.-qt)
        rv = np.minimum(rv,rs)
        pv = mr2pp(rv,PPa)

        Tl      = 55 + 2840./(3.5*np.log(TK) - np.log(pv/100.) - 4.805)
        theta_e = TK*(P0/PPa)**(0.2854*(1.0 - 0.28*rv)) * np.exp((3376./Tl - 2.54)*rv*(1+0.81*rv))

        #Tl      = 56 + 2840./(3.5*np.log(TK) - np.log(pv/100.) - 4.805)
        #theta_e = TK*(P0/PPa)**(Rd/cpd) * (TK/Tl)**(0.28*rv) * np.exp((3036./Tl - 1.78)*rv*(1+0.448*rv))

    else: # Follows Stevens and Siebesma, Clouds and Climate Book
        ps = es(TK)
        qs = (ps/(PPa-ps)) * (Rd/Rv) * (1. - qt)        
        qv = np.minimum(qt,qs)
        
        Re = (1-qt)*Rd
        R  = Re + qv*Rv
        pv = qv * (Rv/R) *PPa
        RH = pv/ps
        lv = phase_change_enthalpy(TK)
        cpe= cpd + qt*(cpl-cpd)
        omega_e = (R/Re)**(Re/cpe) * RH**(-qv*Rv/cpe)
        theta_e = TK*(P0/PPa)**(Re/cpe)*omega_e*np.exp(qv*lv/(cpe*TK))
        
    if scalar_input:
        return np.squeeze(theta_e)
    return(theta_e)

def get_theta_l(T,P,qt):
#   """ Calculates liquid-water potential temperature.  Following Stevens and Siebesma
#   Eq. 2.44-2.45 in the Clouds and Climate book
#   """

    TK,  scalar_input1 = thermo_input(T, 'Kelvin')
    PPa, scalar_input2 = thermo_input(P, 'Pascal')
    qt,  scalar_input3 = thermo_input(qt,'kg/kg')
    scalar_input = scalar_input1 and scalar_input2 and scalar_input3

    ps = es(TK)
    qs = (ps/(PPa-ps)) * eps1 * (1. - qt)
    qv = np.minimum(qt,qs)
    ql = qt-qv

    R  = Rd + qv*(Rv - Rd)
    Rl = Rd + qt*(Rv - Rd)
    cpl = cpd + qt*(cpv-cpd)
    lv = phase_change_enthalpy(TK)
    
    omega_l = ((1.-ql)*(R/Rl))**(Rl/cpl) * (qt/qv)**(qt*Rv/cpl)
    theta_l = (TK*(P0/PPa)**(Rl/cpl)) *omega_l*np.exp(-ql*lv/(cpl*TK))

    if scalar_input:
        return np.squeeze(theta_l)
    return(theta_l)

def get_theta_s(T,P,qt):
#   """ Calculates entropy potential temperature. This follows the formulation of Pascal
#   Marquet and ensures that parcels with different theta-s have a different entropy
#   """

    TK,  scalar_input1 = thermo_input(T, 'Kelvin')
    PPa, scalar_input2 = thermo_input(P, 'Pascal')
    qt,  scalar_input3 = thermo_input(qt,'kg/kg')
    scalar_input = scalar_input1 and scalar_input2 and scalar_input3

    kappa = Rd/cpd
    e0    = es(T0)
    Lmbd  = ((sv00 - Rv*np.log(e0/P0)) - (sd00 - Rd*np.log(1-e0/P0)))/cpd
    lmbd  = cpv/cpd - 1.
    eta   = 1/eps1
    delta = eps2
    gamma = kappa/eps1
    r0    = e0/(P0-e0)/eta

    ps = es(TK)
    qs = (ps/(PPa-ps)) * (Rd/Rv) * (1. - qt)
    qv = np.minimum(qt,qs)
    ql = qt-qv

    lv = phase_change_enthalpy(TK)

    R  = Rd + qv*(Rv - Rd)
    pv = qv * (Rv/R) *PPa
    RH = pv/ps
    rv = qv/(1-qv)
    
    x1 = (T/T0)**(lmbd*qt) * (P0/PPa)**(kappa*delta*qt) * (r0/rv)**(gamma*qt) * RH**(gamma*ql)
    x2 = (1.+eta*rv)**(kappa*(1+delta*qt)) * (1+eta*r0)**(-kappa*delta*qt)
    theta_s = (TK*(P0/PPa)**(kappa)) * np.exp(-ql*lv/(cpd*TK)) * np.exp(qt*Lmbd) * x1 * x2

    if scalar_input:
        return np.squeeze(theta_s)
    return(theta_s)

def get_theta_rho(T,P,qt):
#   """ Calculates theta_rho as theta_l * (1+Rd/Rv qv - qt)
#   """  

    TK,  scalar_input1 = thermo_input(T, 'Kelvin')
    PPa, scalar_input2 = thermo_input(P, 'Pascal')
    qt,  scalar_input3 = thermo_input(qt,'kg/kg')
    scalar_input = scalar_input1 and scalar_input2 and scalar_input3

    theta_l = get_theta_l(TK,PPa,qt)
    
    ps = es(TK)
    qs = (ps/(PPa-ps)) * (Rd/Rv) * (1. - qt)
    qv = np.minimum(qt,qs)
    theta_rho = theta_l * (1.+ qv/eps1 - qt)
 
    if scalar_input:
        return np.squeeze(theta_rho)
    return(theta_rho)

def T_from_Te(Te,P,qt):
    """ Given theta_e solves implicitly for the temperature at some other pressure,
    so that theta_e(T,P,qt) = Te
	>>> T_from_Te(350.,1000.,17)
	304.4761977
    """
    
    def zero(T,Te,P,qt):
        return  np.abs(Te/get_theta_e(T,P,qt)-1.)
    return optimize.fsolve(zero,   300., args=(Te,P,qt))

def T_from_Tl(Tl,P,qt):
    """ Given theta_e solves implicitly for the temperature at some other pressure,
    so that theta_e(T,P,qt) = Te
	>>> T_from_Tl(282.75436951,90000,20.e-3)
	290.00
    """
    def zero(T,Tl,P,qt):
        return  np.abs(Tl/get_theta_l(T,P,qt)-1.)
    return optimize.fsolve(zero,   280., args=(Tl,P,qt))

def P_from_Te(Te,T,qt):
    """ Given Te solves implicitly for the pressure at some temperature and qt
    so that theta_e(T,P,qt) = Te
	>>> P_from_Te(350.,305.,17)
	100464.71590478
    """
    def zero(P,Te,T,qt):
        return np.abs(Te/get_theta_e(T,P,qt)-1.)
    return optimize.fsolve(zero, 90000., args=(Te,T,qt))

def P_from_Tl(Tl,T,qt):
    """ Given Tl solves implicitly for the pressure at some temperature and qt
    so that theta_l(T,P,qt) = Tl
	>>> T_from_Tl(282.75436951,290,20.e-3)
	90000
    """
    def zero(P,Tl,T,qt):
        return np.abs(Tl/get_theta_l(T,P,qt)-1.)
    return optimize.fsolve(zero, 90000., args=(Tl,T,qt))

def get_Plcl(T,P,qt,iterate=False, ice=False):
    """ Returns the pressure [Pa] of the LCL.  The routine gives as a default the 
    LCL using the Bolton formula.  If iterate is true uses a nested optimization to 
    estimate at what pressure, Px and temperature, Tx, qt = qs(Tx,Px), subject to 
    theta_e(Tx,Px,qt) = theta_e(T,P,qt).  This works for saturated air.
	>>> Plcl(300.,1020.,17)
	96007.495
    """
    
    def delta_qs(P,Te,qt):
        TK = T_from_Te(Te,P,qt)
        if (ice):
            ps = es(TK,state='ice')
        else:
            ps = es(TK)
        qs = (1./(P/ps-1.)) * eps1 * (1. - qt)
        return np.abs(qs/qt-1.)

    TK,  scalar_input1 = thermo_input(T, 'Kelvin')
    PPa, scalar_input2 = thermo_input(P, 'Pascal')
    qt,  scalar_input3 = thermo_input(qt,'kg/kg')
    scalar_input = scalar_input1 and scalar_input2 and scalar_input3

    if (iterate):
        Te   = get_theta_e(TK,PPa,qt)
        if scalar_input:
            Plcl = optimize.fsolve(delta_qs, 80000., args=(Te,qt))
            return np.squeeze(Plcl)
        else:
            if (scalar_input3):
                qx =np.empty(np.shape(Te)); qx.fill(np.squeeze(qt)); qt = qx
            elif len(Te) != len(qt):
                print('Error in get_Plcl: badly shaped input')

        Plcl = np.zeros(np.shape(Te))
        for i,x in enumerate(Te):
            Plcl[i] = optimize.fsolve(delta_qs, 80000., args=(x,qt[i]))
    else: # Bolton
        cp = cpd + qt*(cpv-cpd)
        R  = Rd  + qt*(Rv-Rd)
        pv = mr2pp(qt/(1.-qt),PPa)
        Tl = 55 + 2840./(3.5*np.log(TK) - np.log(pv/100.) - 4.805)
        Plcl = PPa * (Tl/TK)**(cp/R)
        
    return Plcl
