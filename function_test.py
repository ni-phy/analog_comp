from equation import Interpolate, GetRHS
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
title = 'A_test'

 
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},figsize=(5,5.8),dpi=600)


def printFunc(coeff,r_m, i_m,c,label,polar=True,kr=-1.0):
    N = len(coeff)
    NS = 4*int(N/2)*10
    x = np.linspace(0,2*np.pi,NS)
    fun = np.zeros(NS,dtype=np.cdouble)
    for ii in range(NS):
        fun[ii] = Interpolate(coeff,x[ii],kr)
    # if polar:    
    #     fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},figsize=(4.3,4.0),dpi=600)
    # else:
    #     fig, ax = plt.subplots(figsize=(4.3,4.0),dpi=600)
    ax.plot(x,np.real(fun),r_m,color=c,label=r"Re("+label+")")
    ax.plot(x,np.imag(fun),i_m,color=c,label=r"Im("+label+")")
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


nPort = 5
maxorder = 2
N = 2*maxorder + 1

vacuumC = 299792458
vacuumMu = 4.0e-7*np.pi
vacuumEpsilon = 1.0/vacuumMu/vacuumC/vacuumC
epHost = vacuumEpsilon
cHost = vacuumC
omega = 1e12
wavelength = 2.0*np.pi*cHost/omega
controlRadius = 1.0*wavelength
kr = -10*omega/cHost*controlRadius

# targetName = '10by10/gyro_26_v3_A.data'
# targetName = '7by7/square_gyro_30_v4_A.data'
# targetName = '10by10/gyro_26_v3_A.data'
targetName = 'square_gyro26_A.data'
with open(targetName) as f:
    data = [[float(num) for num in line.split()] for line in f]
target = np.zeros((nPort,nPort),dtype=np.cdouble)
for ii in range(nPort):
    for jj in range(nPort):
        target[ii,jj] = data[ii][2*jj]+1j*data[ii][2*jj+1]

# multiplierName = '../analog/GyroPDE10/no_epi_v3_26_multiplier.data'
# multiplierName = '../analog/GyroPDE7/no_epi_v230_multiplier.data'
# multiplierName ='size_test/no_epi_results/no_epi_gyro_28_multiplier.data'
multiplierName = '../analog/GyroPDE10/no_epi_trial26_multiplier.data'

with open(multiplierName) as f:
    data = [[float(num[1:-1]) for num in line.split()] for line in f]
multiplier = float(data[0][0])
print('m', multiplier)

# scatterName = '../analog/GyroPDE10/no_epi_v3_26_stif.data'
# scatterName = '../analog/GyroPDE7/no_epi_v230_stif.data'
# scatterName = '../analog/GyroPDE10/no_epi_v3_scat.data'
# scatterName = 'size_test/no_epi_results/no_epi_gyro_28_scat.data'
scatterName = '../analog/GyroPDE10/no_epi_trial26_stif.data'
with open(scatterName) as f:
    data = [[float(num) for num in line.split()] for line in f]
scattering = np.zeros((nPort,nPort),dtype=np.cdouble)
for ii in range(nPort):
    for jj in range(nPort):
        scattering[ii,jj] = data[ii][2*jj]+1j*data[ii][2*jj+1]

coeffF = [-2,(1+1j)/np.sqrt(2),2.0,(2+1j)/np.sqrt(2),1]
# coeffF = [(1+1j)/np.sqrt(2),(1-1j)/np.sqrt(2),2.0j,(2+1j)/np.sqrt(2),(2-1j)/np.sqrt(2),(1+2j)/np.sqrt(2),(1)/np.sqrt(2)]
# coeffF = [2j,(1)/np.sqrt(2),(1j)/np.sqrt(2),(2+1j)/np.sqrt(2),-1+2.0j,(2j)/np.sqrt(2),(2+1j)/np.sqrt(2),(2)/np.sqrt(2),(1j)/np.sqrt(2),-2]
coeffF = [0.5+1j,(1+1j)/np.sqrt(2),1.0+0.5j,(2+1j)/np.sqrt(2),(1-1j)/np.sqrt(2)]
b = GetRHS(coeffF,N,kr)

x = np.linalg.solve(target,b)
for item in x:
    print(np.abs(item))
printFunc(x, '-', '--', c='k', label='$u_{ref}$', kr=kr)

x = np.linalg.solve(scattering/multiplier,b)
for item in x:
    print(np.abs(item))
printFunc(x,'.', ':', c='b', label='$u_h$',kr=kr)

# fig.tight_layout()
plt.savefig("./"+title+"_compare_fun_5by5_v2.pdf")
plt.close()
