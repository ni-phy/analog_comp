#!/usr/bin/python
###################################################
## Gyrotropic cylinders                          ##
###################################################
__author__ = "Heedong Goh"                       ##
__email__  = "hgoh@gc.cuny.edy"                  ##
###################################################
import numpy as np
import scipy.special as ss
import cmath
###############################################################################
# Scattering coefficient of a PEC cylinder under a plane incident wave
def GetSN_PEC(omega,cHost,radius,N,loss=0.0):
    SN = np.zeros(2*N+1) + 1j*np.zeros(2*N+1)
    kzero = omega/cHost
    for ii in range(2*N+1):
        order = -N+ii
        A = ss.jv(order-1,kzero*radius)-ss.jv(order+1,kzero*radius)
        B = ss.hankel2(order-1,kzero*radius)-ss.hankel2(order+1,kzero*radius)
        SN[ii] = -np.power(1j,-order)*A/B
    return [SN,range(-N,N+1)]
###############################################################################
# Scattering coefficient of a gyrotropic cylinder under a plane incident wave
def GetSN_Gyro(omega,omegap,omegac,cHost,radius,N,loss=0.0):
    SN = np.zeros(2*N+1) + 1j*np.zeros(2*N+1)
    lomega = omega - 1j*loss
    et = 1.0 - omegap*omegap*lomega/omega/(lomega*lomega-omegac*omegac)
    eg = -omegap*omegap*omegac/omega/(lomega*lomega-omegac*omegac)
    effep = (et*et-eg*eg)/et
    kzero = omega/cHost
    sqrteffep = 0.0
    if loss == 0.0:
        if effep >=0:
            sqrteffep = np.sqrt(effep)
        else:
            sqrteffep = -1j*np.sqrt(-effep)
    else:
        sqrteffep = cmath.sqrt(effep)
    rho = kzero*radius*sqrteffep
    for ii in range(2*N+1):
        order = -N+ii
        A = ss.jvp(order,rho)/ss.jv(order,rho)
        B = ss.jvp(order,kzero*radius)/ss.jv(order,kzero*radius)
        C = ss.h2vp(order,kzero*radius)/ss.hankel2(order,kzero*radius)
        SN[ii] = -np.power(1j,-order)*ss.jv(order,kzero*radius)/ss.hankel2(order,kzero*radius)
        SN[ii] *= (A - sqrteffep*B + order*eg/rho/et) / (A - sqrteffep*C + order*eg/rho/et)
    return [SN,range(-N,N+1)]
###############################################################################
# Scattering coefficient of a dielectric cylinder under a plane incident wave
def GetSN_Dielectric(omega,epsilon,cHost,radius,N,loss=0.0):
    SN = np.zeros(2*N+1) + 1j*np.zeros(2*N+1)
    effep = epsilon - 1j*loss
    kzero = omega/cHost
    sqrteffep = cmath.sqrt(effep)
    rho = kzero*radius*sqrteffep
    for ii in range(2*N+1):
        order = -N+ii
        A = ss.jvp(order,rho)/ss.jv(order,rho)
        B = ss.jvp(order,kzero*radius)/ss.jv(order,kzero*radius)
        C = ss.h2vp(order,kzero*radius)/ss.hankel2(order,kzero*radius)
        SN[ii] = -np.power(1j,-order)*ss.jv(order,kzero*radius)/ss.hankel2(order,kzero*radius)
        SN[ii] *= (A - sqrteffep*B) / (A - sqrteffep*C)
    return [SN,range(-N,N+1)]
###############################################################################
# Polarizability tensor (3x3; alpha_m and alpha_e) for a gyrotropic cylinder
def GetAlpha_Gyro(omega,omegap,omegac,cHost,epHost,radius,loss=0.0):
    SN,orders=GetSN_Gyro(omega,omegap,omegac,cHost,radius,1,loss)
    Sneg  = SN[0]
    Szero = SN[1]
    Spos  = SN[2]
    kzero = omega/cHost
    alpha = np.zeros((3,3)) + 1j*np.zeros((3,3))
    alpha[0,0] = 4.0/kzero/kzero        *  1j * Szero
    alpha[1,1] = 4.0*epHost/kzero/kzero *      (Spos-Sneg)
    alpha[1,2] = 4.0*epHost/kzero/kzero * -1j *(Spos+Sneg)
    alpha[2,1] = 4.0*epHost/kzero/kzero *  1j *(Spos+Sneg)
    alpha[2,2] = 4.0*epHost/kzero/kzero *      (Spos-Sneg)
    return alpha
###############################################################################
# Polarizability tensor (3x3; alpha_m and alpha_e) for a PEC cylinder
def GetAlpha_PEC(omega,cHost,epHost,radius,loss=0.0):
    SN,orders=GetSN_PEC(omega,cHost,radius,1,loss)
    Sneg  = SN[0]
    Szero = SN[1]
    Spos  = SN[2]
    kzero = omega/cHost
    alpha = np.zeros((3,3)) + 1j*np.zeros((3,3))
    alpha[0,0] = 4.0/kzero/kzero        * 1j*Szero
    alpha[1,1] = 4.0*epHost/kzero/kzero * Spos*2.0
    alpha[2,2] = 4.0*epHost/kzero/kzero * Spos*2.0
    return alpha
###############################################################################
# Polarizability tensor (3x3; alpha_m and alpha_e) for a dielectric cylinder
def GetAlpha_Dielectric(omega,epsilon,cHost,epHost,radius,loss=0.0):
    SN,orders=GetSN_Dielectric(omega,epsilon,cHost,radius,1,loss)
    Sneg  = SN[0]
    Szero = SN[1]
    Spos  = SN[2]
    kzero = omega/cHost
    alpha = np.zeros((3,3)) + 1j*np.zeros((3,3))
    alpha[0,0] = 4.0/kzero/kzero        * 1j*Szero
    alpha[1,1] = 4.0*epHost/kzero/kzero * Spos*2.0
    alpha[2,2] = 4.0*epHost/kzero/kzero * Spos*2.0
    return alpha
###############################################################################
#Get incident electric field (curl of a scalar plane wave of given angle and phase)
def GetEinc(position,omega,cHost,epHost,angle,phase=0.0):
    dim = 2
    kHost = omega/cHost
    sint = np.sin(angle)
    cost = np.cos(angle)
    common = kHost/epHost/omega*np.exp(-1j*kHost*position[0]*np.cos(angle))*np.exp(-1j*kHost*position[1]*np.sin(angle))*np.exp(-1j*phase)
    Einc = np.zeros(dim)+1j*np.zeros(dim)
    Einc[0] = -common*sint
    Einc[1] =  common*cost
    return Einc
###############################################################################
#Get incident magnetic field (scalar plane wave of given angle and phase)
def GetHinc(position,omega,cHost,epHost,angle,phase=0.0):
    dim = 2
    kHost = omega/cHost
    sint = np.sin(angle)
    cost = np.cos(angle)
    Hz = np.exp(-1j*kHost*position[0]*np.cos(angle))*np.exp(-1j*kHost*position[1]*np.sin(angle))*np.exp(-1j*phase)
    return Hz
###############################################################################
# Green's function
def GetGamma(position1,position2,omega,cHost,epHost):
    kHost = omega/cHost
    x1 = position1[0]
    y1 = position1[1]
    x2 = position2[0]
    y2 = position2[1]
    R = kHost*np.sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) )
    C = kHost*(x1-x2)/R
    S = kHost*(y1-y2)/R
    hzero = ss.hankel2(0,R)
    hone  = ss.hankel2(1,R)
    htwo  = ss.hankel2(2,R)
    gamma = np.zeros((3,3),dtype=np.cdouble)
    gamma[0,0] = -epHost*hzero                       
    gamma[0,1] =  1j*cHost*epHost*S*hone
    gamma[0,2] = -1j*cHost*epHost*C*hone
    gamma[1,0] = -1j/cHost*S*hone
    gamma[1,1] =  S*S*hzero+(C*C-S*S)*hone/R
    gamma[1,2] =  C*S*htwo                           
    gamma[2,0] =  1j/cHost*C*hone
    gamma[2,1] =  C*S*htwo                           
    gamma[2,2] =  C*C*hzero+(S*S-C*C)*hone/R
    gamma = gamma * 0.25*1j*kHost*kHost/epHost
    return gamma
###############################################################################
# Get all dipoles for given polarizabilities of all cylinders
def GetDipoles(alphas,positions,omega,cHost,epHost,angle,phase=0.0):
    n = len(alphas)
    dim = 3
    A = np.eye(dim*n,dtype=np.cdouble)
    b = np.zeros(dim*n,dtype=np.cdouble)
    x = np.zeros(dim*n,dtype=np.cdouble)
    for ii in range(n):
        position1 = positions[ii*2:ii*2+2]
        alpha = alphas[ii]
        Finc = np.zeros(dim,dtype=np.cdouble)
        Finc[0]     = GetHinc(position1,omega,cHost,epHost,angle)
        Finc[1:dim] = GetEinc(position1,omega,cHost,epHost,angle)
        b[ii*dim:ii*dim+dim] = np.matmul(alpha,Finc)
        for jj in range(n):
            if ii == jj:
                continue
            position2 = positions[jj*2:jj*2+2]
            gamma = GetGamma(position1,position2,omega,cHost,epHost)
            A[ii*dim:ii*dim+dim,jj*dim:jj*dim+dim] = -np.matmul(alpha,gamma)
    x = np.linalg.solve(A,b)
    return x
###############################################################################
# Get scattered magnetic and electric fields from dipoles
def GetField(positionR,dipoles,positions,omega,cHost,epHost):
    n = int(len(positions)/2)
    dim = 3
    kHost = omega/cHost
    field = np.zeros(dim,dtype=np.cdouble)
    for ii in range(n):
        dipole = dipoles[ii*dim:(ii+1)*dim]
        position = positions[ii*2:ii*2+2]
        gamma = GetGamma(positionR,position,omega,cHost,epHost)
        field = field + np.matmul(gamma,dipole)
    return field
###############################################################################
# Debugging purpose
def GetHscatFromSN(omega,cHost,SN,orders,angle,x,y):
    kzero = omega/cHost
    r2 = x*x+y*y
    radius  = np.sqrt(r2)
    eiphi  = x/radius + 1j*y/radius
    val = 0.0
    for ii in range(len(SN)):
        order = orders[ii]
        A = ss.hankel2(order,kzero*radius)
        B = np.power(eiphi,order)*np.exp(-1j*order*angle)
        val += A*B*SN[ii]
    return val
###############################################################################
def GetDipolesFromField(alphas,positions,omega,cHost,epHost,incFields):
    n = len(alphas)
    dim = 3
    A = np.eye(dim*n,dtype=np.cdouble)
    b = np.zeros(dim*n,dtype=np.cdouble)
    x = np.zeros(dim*n,dtype=np.cdouble)
    for ii in range(n):
        position1 = positions[ii*2:ii*2+2]
        alpha = alphas[ii]
        Finc = np.zeros(dim,dtype=np.cdouble)
        Finc[0] = incFields[dim*ii  ]
        Finc[1] = incFields[dim*ii+1]
        Finc[2] = incFields[dim*ii+2]
        b[ii*dim:ii*dim+dim] = np.matmul(alpha,Finc)
        for jj in range(n):
            if ii == jj:
                continue
            position2 = positions[jj*2:jj*2+2]
            gamma = GetGamma(position1,position2,omega,cHost,epHost)
            A[ii*dim:ii*dim+dim,jj*dim:jj*dim+dim] = -np.matmul(alpha,gamma)
    x = np.linalg.solve(A,b)
    return x

