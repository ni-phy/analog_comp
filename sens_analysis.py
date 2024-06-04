#!/usr/bin/env python
import numpy as np
import design as des
import scat as scat
import gyroGen as gyro
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.special as ss
from scipy import optimize
import os
import math

def sens_analysis(positions, index, controlRadius, params, perturb):
    obj1 = des.Objective(positions, params)
    
    newPositions = positions
    newPositions[index] = positions[index] + controlRadius*perturb
    obj2 = des.Objective(newPositions, params)
    
    return (obj1-obj2)/perturb


mpl.use('Agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

#
title = '../analog/sensitivity'
targetName = 'test_g_stif.data'
posName = 'test_g_pos.data'
# parameters
vacuumC = 299792458
vacuumMu = 4.0e-7*np.pi
vacuumEpsilon = 1.0/vacuumMu/vacuumC/vacuumC
epHost = vacuumEpsilon
cHost = vacuumC
siEpsilon = 12*vacuumEpsilon
omega = 1e12
omegap = omega*2.1
omegac = np.sqrt( (omega/omegap)*(omega/omegap) - 1.0 + 0.25/(omega/omegap)/(omega/omegap) ) * omegap
print(omegac/omegap)
wavelength = 2.0*np.pi*cHost/omega
radius = wavelength * 0.01
loss = 0.0
alphaG = gyro.GetAlpha_Gyro(omega,omegap,omegac,cHost,epHost,radius,loss)
alphaD = gyro.GetAlpha_Dielectric(omega,siEpsilon,cHost,epHost,radius,loss)
# design parameters
nAlpha = 5
alphas = []
atype = np.zeros(nAlpha,dtype=np.int32)

## atype == 0: dielectric; == 1: gyrotropic

#importing positions and adding random element
positions = np.zeros(nAlpha*2, dtype=np.double)
with open(posName) as f:
    data = [[float(num) for num in line.split()] for line in f]
for jj in range(nAlpha):
        positions[2*jj] = data[jj][0]
        positions[2*jj+1] = data[jj][1]
        atype[jj] = data[jj][2]

# atype = np.random.randint(2,size=nAlpha)
atype = np.zeros(nAlpha)
atype[0] = 1
# atype = np.ones(nAlpha)
for ii in range(nAlpha):
    if atype[ii] == 0:
        alphas.append(alphaD)
    else:
        alphas.append(alphaG)
radii = np.ones(nAlpha)*radius
offset = 0.0
controlRadius = 2.0*wavelength
distFlag = 1

distFlag = des.DistanceCheck(positions,radii,offset)
if distFlag == 1:
    print("rerun")
    exit()
# port definition
nPort = 5
obsRadius = controlRadius*1.5
normalize = True
#
with open(targetName) as f:
    data = [[float(num) for num in line.split()] for line in f]
target = np.zeros((nPort,nPort),dtype=np.cdouble)
for ii in range(nPort):
    for jj in range(nPort):
        target[ii,jj] = data[ii][2*jj]+1j*data[ii][2*jj+1]

params = [alphas,omega,cHost,epHost,nPort,obsRadius,normalize,target]
perturb = radius*1.0e-11
maxIter = 100
clength = radius*1.0e-1

plt.figure()
plt.plot()

plt.title('Sensitivity of the Cost Function \n to Displacement for Each Cylinder')

for i in range(len(positions)):
    delta = sens_analysis(positions, i, controlRadius, params, 1e-5)
    plt.scatter(int(i/2), np.log10(np.abs(delta)), c='k', marker='x' if atype[int(i/2)]==1 else 'o')

plt.ylabel('$Log_{10} \dfrac{ \Delta F}{ \Delta x}$')
plt.xlabel('Particle Number (x/y)')
plt.savefig(title+'_plot.pdf')
plt.close()

