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

def Interpolate(coeff,x,kr=-1.0):
    N = len(coeff)
    val = 0.0
    for ii in range(N):
        order = -int(N/2)+ii
        scale = 1.0
        if kr > 0:
            scale *= ss.hankel2(order,kr) 
        val = val + coeff[ii]*np.exp(1j*order*x)*scale
    return val
#
def GetStif(coeffA,coeffB,coeffC,N,kr=-1.0):
    NS = 4*int(N/2)*10
    x = np.linspace(0,2*np.pi,NS)
    dx = x[1]-x[0]
    stif = np.zeros((N,N),dtype=np.cdouble)
    for ii in range(N):
        m = -int(N/2) + ii
        for jj in range(N):
            n = -int(N/2) + jj
            scale = 1.0/(2.0*np.pi)
            if kr > 0:
                scale *= ss.hankel2(n,kr) / ss.jv(m,kr)
            val = 0.0
            for kk in range(NS-1):
                alpha = Interpolate(coeffA,x[kk])
                beta  = Interpolate(coeffB,x[kk])
                gamma = Interpolate(coeffC,x[kk])
                integrand = (m*n*alpha-1j*n*beta-gamma)*np.exp(-1j*(m-n)*x[kk])
                #integrand = (-m*n*alpha-1j*n*beta-gamma)*np.exp(1j*(m+n)*x[kk])
                val = val + integrand*dx
            val = val * scale
            stif[ii,jj] = val
    return stif
#
def GetRHS(coeffF,N,kr=-1.0):
    NS = 4*int(N/2)*10
    x = np.linspace(0,2*np.pi,NS)
    dx = x[1]-x[0]
    b = np.zeros(N,dtype=np.cdouble)
    for ii in range(N):
        m = -int(N/2) + ii
        scale = 1.0/(2.0*np.pi)
        if kr > 0:
            scale *=  1.0 / ss.jv(m,kr)
        val = 0.0
        for kk in range(NS-1):
            f = Interpolate(coeffF,x[kk])
            integrand = np.exp(-1j*m*x[kk])*f
            #integrand = np.exp(1j*m*x[kk])*f
            val = val + integrand*dx
        val = val * scale
        b[ii] = val
    return b
#    
def printMat(scatAmp,title='test',minval=0,maxval=0):
    fig, ax = plt.subplots(figsize=(4.3,4.0),dpi=600)
    N = len(scatAmp)
    if minval == maxval:
        maxval = np.max(np.max(np.abs(scatAmp)))
        minval = -maxval
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
def printFunc(coeff,title='test',polar=True,kr=-1.0):
    N = len(coeff)
    NS = 4*int(N/2)*10
    x = np.linspace(0,2*np.pi,NS)
    fun = np.zeros(NS,dtype=np.cdouble)
    for ii in range(NS):
        fun[ii] = Interpolate(coeff,x[ii],kr)
    if polar:    
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},figsize=(4.3,4.0),dpi=600)
    else:
        fig, ax = plt.subplots(figsize=(4.3,4.0),dpi=600)
    ax.plot(x,np.real(fun),'-',color='black',label="real")
    ax.plot(x,np.imag(fun),'--',color='black',label="imag")
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
    '''
    ax.set_xlim(0,N)
    ax.set_ylim(N,0)
    ticks = np.linspace(0.5,N-0.5,N)
    labels = np.linspace(1,N,N)
    ax.set_xticks(ticks,labels.astype(int),fontsize=fontsizet)
    ax.set_yticks(ticks,labels.astype(int),fontsize=fontsizet)
    '''
    '''
    labels = [r'1',r'2',r'3']#,r'4',r'5',r'6']
    labels = [r'1',r'2',r'3']#,r'4',r'5',r'6']
    #ax.set_xlabel(labels)
    #plt.xlabel(r"$x$",fontsize=18)
    #plt.ylabel("Amplitude",fontsize=18)
    '''
    if polar:    
        angle = np.deg2rad(67.5)
        ax.legend(loc="lower left", bbox_to_anchor=(.5 + np.cos(angle)/2, .5 + np.sin(angle)/2))
    else:
        ax.legend()
    #ax.set_aspect('equal')
    #divider = make_axes_locatable(ax)
    #ccax = divider.append_axes("right", size="4%", pad=0.2)
    #cbar = fig.colorbar(surface,cax=ccax)
    fig.tight_layout()
    fig.savefig("./"+title+"_fun.pdf")
    plt.close()
#

def plot_A(A, title):
    N = len(A)
    fontsizel=20
    fontsizet=16
    labels = np.linspace(1,N+1,N+1)
 
    p1 = np.real(A)
    p2 = np.imag(A)

    x = np.arange(0, len(p1), 1)
    y = np.arange(0, len(p2), 1)
    X, Y =np.meshgrid(x, y)

    p1_norm = (p1 - np.min(p1)) / (np.max(p1) - np.min(p1))  # Shift and scale p1
    p2_norm = (p2 - np.min(p2)) / (np.max(p2) - np.min(p2))  # Shift and scale p2

    # Create an RGB image where p1 controls Red, p2 controls Blue, and Green indicates negativity
    color_plot = np.zeros((p1.shape[0], p1.shape[1], 3))  # Initialize a 3-channel image

    # For the Red channel: positive values of p1
    color_plot[..., 1] = p1_norm  # Red channel (from p1)

    # For the Blue channel: positive values of p2
    color_plot[..., 2] = p2_norm  # Blue channel (from p2)

    # Create the 2D color map (for visualization of all combinations of p1 and p2)
    p1_values = np.linspace(np.min(p1), np.max(p1), 256)
    p2_values = np.linspace(np.min(p2), np.max(p2), 256)
    p1_grid, p2_grid = np.meshgrid(p1_values, p2_values)

    # Create a color map where p1 controls red, p2 controls blue, and negative values control green
    color_map_2d = np.zeros((p1_grid.shape[0], p1_grid.shape[1], 3))
    color_map_2d[..., 1] = (p1_grid - np.min(p1_grid)) / (np.max(p1_grid) - np.min(p1_grid))  # Red for p1
    color_map_2d[..., 2] = (p2_grid - np.min(p2_grid)) / (np.max(p2_grid) - np.min(p2_grid))  # Blue for p2

    # Plot the RGB image and the 2D color map
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].tick_params(
        axis='both',  
        which='both', 
        direction='out',
        left=False,   
        right=False,
        top=False,
        bottom=False)
    # ax[0].set_xlim(0,N)
    # ax[0].set_ylim(N,0)
    ticks = np.linspace(0.5,N-0.5,N+1)
    ax[0].set_xticks(ticks,labels.astype(int),fontsize=fontsizet)
    ax[0].set_yticks(ticks,labels.astype(int),fontsize=fontsizet)

    # Main plot with RGB colors based on p1 and p2
    im = ax[0].imshow(color_plot, extent=[x.min(), x.max(), y.min(), y.max()])
    ax[0].set_title("Visualisation of A Real and Imaginary Parts", fontsize=fontsizel)

    ax[0].set_aspect('equal')
    divider = make_axes_locatable(ax[0])

    # 2D color map for reference (shows p1 vs p2 relationship, including negative values)
    ax[1].imshow(color_map_2d, extent=[np.min(p1), np.max(p1), np.min(p2), np.max(p2)], origin='lower')
    ax[1].set_title("2D Colormap", fontsize=fontsizel)
    ax[1].set_xlabel("Real A", fontsize=fontsizel)
    ax[1].set_ylabel("Imaginary A", fontsize=fontsizel)

    plt.tight_layout()
    plt.savefig(title)

if __name__ == '__main__':
    #
    title = 'square_gyro26'
    # parameters
    vacuumC = 299792458
    vacuumMu = 4.0e-7*np.pi
    vacuumEpsilon = 1.0/vacuumMu/vacuumC/vacuumC
    epHost = vacuumEpsilon
    cHost = vacuumC
    omega = 1e12
    wavelength = 2.0*np.pi*cHost/omega
    controlRadius = 1.0*wavelength
    kr = -10*omega/cHost*controlRadius

    maxorder = 2
    
    N = 2*maxorder + 1

    L = 2.0*np.pi

    
    #coeffA = [(1.0-2*1j)/L/L, (2+1j)/L/L, 5/L/L, (2-1j)/L/L, (1.0+2*1j)/L/L]
    #coeffB = [0,0,0,0,0]
    #coeffC = [1-2*1j, 2-1j, 0, 2+1j, 1+2*1j]

    coeffA = [(1.0+2*1j)/L/L, (2+1j)/L/L, 5/L/L-1j/L/L, -(2-1j)/L/L, (1.0+2*1j)/L/L]
    coeffB = [0,0,0,0,0]
    coeffC = [-1-2*1j, 2+1j, 0, 2+1j, 1+2*1j]
    coeffF = [-2,(1+1j)/np.sqrt(2),2.0,(2+1j)/np.sqrt(2),1]

    # coeffA = [ 0.07944097+0.06780868j,  0.20090331+0.08486717j, -0.23107647+0.04118409j,
    #             0.70399919+0.13738002j,  0.57112592-0.77705891j]
    # coeffB = [0,0,0]
    # coeffC =   [0.02839489+0.12051616j, 0.55713239+0.14004572j, 0.75497644+1.41474772j,
    #             0.49373484+0.15436184j, 0.15268701+0.23522389j]

    with open("coeffA26.data") as f:
        data = [[float(num) for num in line.split()] for line in f]
    data = data[0]
    coeffA = np.zeros((N),dtype=np.cdouble)
    for ii in range(N):
            coeffA[ii] = data[2*ii]+1j*data[2*ii+1]
    
    with open("coeffC26.data") as f:
        data = [[float(num) for num in line.split()] for line in f]
    data = data[0]
    coeffC = np.zeros((N),dtype=np.cdouble)
    for ii in range(N):
            coeffC[ii] = data[2*ii]+1j*data[2*ii+1]
    print('C: ', coeffC)
    
    coeffF = [(1+1j)/np.sqrt(2),1.0,(1-1j)/np.sqrt(2)]

    A     = GetStif(coeffA,coeffB,coeffC,N,kr)
    norm  = np.linalg.norm(np.linalg.inv(A),ord=2)
    A    *= norm

    print()
    maxval = np.max(np.max(A))
    minval = np.min(np.min(A))
    print(maxval,minval)
    print()
    
    invA  = np.linalg.inv(A)
    #norm  = np.linalg.norm(invA,ord=2)
    #invA /= norm
    maxval = np.max(np.max(invA))
    minval = np.min(np.min(invA))
    print(maxval,minval)
    print()
    
    printMat(np.real(A),title=title+'_re')
    printMat(np.imag(A),title=title+'_im')
    printMat(np.abs(A),title=title+'_abs')

    printMat(np.real(invA),title=title+'_inv_re')
    printMat(np.imag(invA),title=title+'_inv_im')
    printMat(np.abs(invA),title=title+'_inv_abs')


    printFunc(coeffA,title=title+'_alpha')
    printFunc(coeffB,title=title+'_beta')
    printFunc(coeffC,title=title+'_gamma')
    printFunc(coeffF,title=title+'_fn')

    plot_A(A, title+'_2Dmap')
    plot_A(invA, title+'inv_2Dmap')

    b = GetRHS(coeffF,N,kr)
    
    #norm = np.linalg.norm(b,ord=2)
    #b /= norm
    
    x = np.linalg.solve(A,b)
    for item in x:
        print(np.abs(item))
    printFunc(x,title=title+'_sol',kr=kr)

    fout = open(title+"_invA.data","w")
    for ii in range(N):
        for jj in range(N):
            fout.write(f"{np.real(invA[ii,jj]):20.10e}\t{np.imag(invA[ii,jj]):20.10e}\t")
        fout.write(f"\n")
    fout.close()
    fout = open(title+"_A.data","w")
    for ii in range(N):
        for jj in range(N):
            fout.write(f"{np.real(A[ii,jj]):20.10e}\t{np.imag(A[ii,jj]):20.10e}\t")
        fout.write(f"\n")
    fout.close()



    
