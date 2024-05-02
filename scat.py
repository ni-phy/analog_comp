#!/usr/bin/env python
import numpy as np
import design as des
import gyroGen as gyro
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.special as ss
import os
mpl.use('Agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

def GetIncFields(positions,order,omega,cHost,epHost):
    nPoint = int(len(positions)/2)
    nf = 3
    fields = np.zeros(nf*nPoint,dtype=np.cdouble)
    wavenumber = omega/cHost
    for ii in range(nPoint):
        x = positions[2*ii  ]
        y = positions[2*ii+1]
        r = np.sqrt(x*x+y*y)
        if r != 0.0:
            cost = x/r
            sint = y/r
            H = ss.jv(order,wavenumber*r)*np.power(cost+1j*sint,order)
            Ex = order*(-1j*y*y/r/r/r - x*y/r/r/r + 1j/r ) * np.power(cost+1j*sint,order-1) * ss.jv(order,wavenumber*r) + wavenumber*y/2/r*np.power(cost+1j*sint,order)*(ss.jv(order-1,wavenumber*r)-ss.jv(order+1,wavenumber*r))
            Ey = -order*(-x*x/r/r/r - 1j*x*y/r/r/r + 1/r ) * np.power(cost+1j*sint,order-1) * ss.jv(order,wavenumber*r) - wavenumber*x/2/r*np.power(cost+1j*sint,order)*(ss.jv(order-1,wavenumber*r)-ss.jv(order+1,wavenumber*r))
            Ex /= 1j*omega*epHost
            Ey /= 1j*omega*epHost
        else:
            H = ss.jv(order,wavenumber*r)
            Ex = 0.0
            Ey = 0.0
        fields[nf*ii] = H
        fields[nf*ii+1] = Ex
        fields[nf*ii+2] = Ey
    return fields
#    
def GetScatteringMatrix(positions,alphas,omega,cHost,epHost,nPort,obsRadius,normalize,nobs=100):
    scat = np.zeros((nPort,nPort),dtype=np.cdouble)
    nobs *= nPort
    angles = np.linspace(0,2*np.pi,nobs,endpoint=False)
    obsX = obsRadius*np.cos(angles)
    obsY = obsRadius*np.sin(angles)
    dangle = angles[1]-angles[0]
    kr = omega/cHost*obsRadius
    for jPort in range(nPort):
        jOrder = -int(nPort/2) + jPort
        incFields = GetIncFields(positions,jOrder,omega,cHost,epHost)
        dipoles = gyro.GetDipolesFromField(alphas,positions,omega,cHost,epHost,incFields)
        for iPort in range(nPort):
            iOrder = -int(nPort/2) + iPort
            val = 0.0
            for ii in range(nobs):
                sfield = gyro.GetField([obsX[ii],obsY[ii]],dipoles,positions,omega,cHost,epHost)
                val += sfield[0]*np.exp(-1j*iOrder*angles[ii])*dangle
            val /= ss.hankel2(iOrder,kr)
            scat[iPort,jPort] = val
    if normalize:
        norm = np.linalg.norm(scat,ord=2)
        scat /= norm
    return scat
#
def PlotIncField(order,domainWidth,epHost,cHost,omega):
    nn = 100
    xx = np.linspace(-domainWidth,domainWidth,nn)
    yy = np.linspace(-domainWidth,domainWidth,nn)
    xgrid,ygrid = np.meshgrid(xx,yy)
    H = np.zeros((nn,nn),dtype=np.cdouble)
    Ex = np.zeros((nn,nn),dtype=np.cdouble)
    Ey = np.zeros((nn,nn),dtype=np.cdouble)
    for ii in range(nn):
        for jj in range(nn):
            field = GetIncFields([xgrid[ii,jj],ygrid[ii,jj]],order,omega,cHost,epHost)
            H[ii,jj] = field[0]
            Ex[ii,jj] = field[1]
            Ey[ii,jj] = field[2]
    #
    fig, ax = plt.subplots(1,3,figsize=(10.3,4.0),dpi=600)
    maxval = np.max(np.max(np.abs(H)))
    ax[0].pcolormesh(xgrid,ygrid,np.real(H),vmin=-maxval,vmax=maxval)
    maxval = np.max([np.max(np.max(np.abs(Ex))),np.max(np.max(np.abs(Ey)))])
    ax[1].pcolormesh(xgrid,ygrid,np.real(Ex),vmin=-maxval,vmax=maxval)
    ax[2].pcolormesh(xgrid,ygrid,np.real(Ey),vmin=-maxval,vmax=maxval)
    fig.tight_layout()
    fig.savefig(f"F_{order:d}.png")
    plt.close()
#
def printMat(scatAmp,title='test',minval=-1,maxval=1):
    fig, ax = plt.subplots(figsize=(4.3,4.0),dpi=600)
    N = len(scatAmp)
    #minval = -1.0 #np.min(scatAmp)
    #maxval =  1.0 #np.max(scatAmp)
    #minval = np.min(scatAmp)
    #maxval = np.max(scatAmp)
    surface = ax.pcolor(scatAmp,vmin=minval,vmax=maxval)
    fontsizel=20
    fontsizet=16
    ax.tick_params(
        axis='both',  
        which='both', 
        direction='out',
        left=False,   
        right=False,
        top=False,
        bottom=False)
    ax.set_xlim(0,N)
    ax.set_ylim(N,0)
    ticks = np.linspace(0.5,N-0.5,N)
    labels = np.linspace(1,N,N)
    ax.set_xticks(ticks,labels.astype(int),fontsize=fontsizet)
    ax.set_yticks(ticks,labels.astype(int),fontsize=fontsizet)
    '''
    labels = [r'1',r'2',r'3']#,r'4',r'5',r'6']
    labels = [r'1',r'2',r'3']#,r'4',r'5',r'6']
    #ax.set_xlabel(labels)
    #plt.xlabel(r"$x$",fontsize=18)
    #plt.ylabel("Amplitude",fontsize=18)
    '''
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    ccax = divider.append_axes("right", size="4%", pad=0.2)
    cbar = fig.colorbar(surface,cax=ccax)
    fig.tight_layout()
    fig.savefig("./"+title+".png")
    plt.close()
#
def printPos(positions,title,domainR,minval,maxval,wavelength,atype):
    fig, ax = plt.subplots(figsize=(4.3,4.0),dpi=600)
    #minval = -1.0 #np.min(scatAmp)
    #maxval =  1.0 #np.max(scatAmp)
    #minval = np.min(scatAmp)
    #maxval = np.max(scatAmp)
    N = int(len(positions)/2)
    for ii in range(N):
        if atype[ii] == 0:
            ax.plot([positions[2*ii]/wavelength],[positions[2*ii+1]/wavelength],'o',markerfacecolor='none',color='black')
        else:
            ax.plot([positions[2*ii]/wavelength],[positions[2*ii+1]/wavelength],'x',color='black')
    theta = np.linspace(0,2*np.pi,1000,endpoint=False)
    xx = domainR*np.cos(theta)/wavelength
    yy = domainR*np.sin(theta)/wavelength
    ax.plot(xx,yy,':',color='gray')  
    fontsizel=20
    fontsizet=16
    ax.tick_params(
        axis='both',  
        which='both', 
        direction='out',
        left=True,   
        right=False,
        top=False,
        bottom=True)
    ax.set_xlim(minval/wavelength,maxval/wavelength)
    ax.set_ylim(minval/wavelength,maxval/wavelength)
    #ticks = [0.5,1.5,2.5]#,3.5,4.5,5.5]
    #labels = [r'1',r'2',r'3']#,r'4',r'5',r'6']
    #ax.set_xticks(ticks,labels,fontsize=fontsizet)
    #labels = [r'1',r'2',r'3']#,r'4',r'5',r'6']
    #ax.set_yticks(ticks,labels,fontsize=fontsizet)
    #ax.set_xlabel(labels)
    plt.xlabel(r"$x/\lambda$",fontsize=18)
    plt.ylabel(r"$y/\lambda$",fontsize=18)
    ax.set_aspect('equal')
    #divider = make_axes_locatable(ax)
    #ccax = divider.append_axes("right", size="4%", pad=0.2)
    #cbar = fig.colorbar(surface,cax=ccax)
    fig.tight_layout()
    fig.savefig("./"+title+".png")
    plt.close()       
#

if __name__ == '__main__':
    #
    title = 'test'
    # parameters
    vacuumC = 299792458
    vacuumMu = 4.0e-7*np.pi
    vacuumEpsilon = 1.0/vacuumMu/vacuumC/vacuumC
    epHost = vacuumEpsilon
    cHost = vacuumC
    siEpsilon = 12*vacuumEpsilon
    #    
    # cylinders
    #omega = 4.42e9*2.0*np.pi # rad/s
    #    omegap = 5.28e10 # rad/s
    omega = 1e12
    omegap = omega*2.1
    omegac = np.sqrt( (omega/omegap)*(omega/omegap) - 1.0 + 0.25/(omega/omegap)/(omega/omegap) ) * omegap
    print(omegac/omegap)
    wavelength = 2.0*np.pi*cHost/omega
    radius = wavelength * 0.01
    loss = 0.0
    alphaG = gyro.GetAlpha_Gyro(omega,omegap,omegac,cHost,epHost,radius,loss)
    alphaD = gyro.GetAlpha_Dielectric(omega,siEpsilon,cHost,epHost,radius,loss)
    #
    # design parameters
    nAlpha = 5
    alphas = []
    ## atype == 0: dielectric; == 1: gyrotropic
    atype = np.zeros(nAlpha,dtype=np.int32)
    #atype[int(nAlpha/2):] = 1
    #atype = np.random.randint(2,size=nAlpha)
    #atype = np.zeros(nAlpha)
    #atype = np.ones(nAlpha)
    for ii in range(nAlpha):
        if atype[ii] == 0:
            alphas.append(alphaD)
        else:
            alphas.append(alphaG)
    radii = np.ones(nAlpha)*radius
    offset = 0.0
    controlRadius = 2.0*wavelength
    positions = np.zeros(nAlpha*2,dtype=np.double)
    distFlag = 1
    for ii in range(300):
        rr = np.random.rand(nAlpha)*controlRadius
        aa = np.random.rand(nAlpha)*2.0*np.pi
        for jj in range(nAlpha):
            positions[2*jj] = rr[jj]*np.cos(aa[jj])
            positions[2*jj+1] = rr[jj]*np.sin(aa[jj])
        if des.DistanceCheck(positions,radii,offset) == 0:
            distFlag = 0
            break
    if distFlag == 1:
        print("rerun")
        exit()
    # port definition
    nPort = 3
    obsRadius = controlRadius*1.5
    normalize = True
    scat = GetScatteringMatrix(positions,alphas,omega,cHost,epHost,nPort,obsRadius,normalize)
    #for iPort in range(nPort):
    #    print(scat[iPort,:])
    stif = np.linalg.inv(scat)
    maxstif = np.max(np.max(np.abs(stif)))
    printMat(np.real(scat),title+'_re',-1,1)
    printMat(np.imag(scat),title+'_im',-1,1)
    printMat(np.abs(scat),title+'_abs',0,1)
    printMat(np.real(stif),title+'_inv_re',-maxstif,maxstif)
    printMat(np.imag(stif),title+'_inv_im',-maxstif,maxstif)
    printMat(np.abs(stif),title+'_inv_abs',0,maxstif)
    printPos(positions,title+'_pos',controlRadius,-obsRadius,obsRadius,wavelength,atype)
    fout = open(title+"_scat.data","w")
    for ii in range(nPort):
        for jj in range(nPort):
            fout.write(f"{np.real(scat[ii,jj]):20.10e}\t{np.imag(scat[ii,jj]):20.10e}\t")
        fout.write(f"\n")
    fout.close()
    fout = open(title+"_stif.data","w")
    for ii in range(nPort):
        for jj in range(nPort):
            fout.write(f"{np.real(stif[ii,jj]):20.10e}\t{np.imag(stif[ii,jj]):20.10e}\t")
        fout.write(f"\n")
    fout.close()
    fout = open(title+"_pos.data","w")
    for ii in range(nAlpha):
        fout.write(f"{positions[2*ii]:20.10e}\t{positions[2*ii+1]:20.10e}\t{atype[ii]:d}\n")
    fout.close()

    #
