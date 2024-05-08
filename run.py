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
import nlopt

mpl.use('Agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

#
title = 'des_mix'
targetName = 'test_stif.data'
posName = 'test_pos.data'
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
        positions[2*jj] = data[jj][0]+float(np.random.rand(1))*10e-6
        positions[2*jj+1] = data[jj][1]+float(np.random.rand(1))*10e-6
        atype[jj] = data[jj][2]

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
distFlag = 1
# for ii in range(300):
#     rr = np.random.rand(nAlpha)*controlRadius
#     aa = np.random.rand(nAlpha)*2.0*np.pi
#     for jj in range(nAlpha):
#         positions[2*jj] = rr[jj]*np.cos(aa[jj])
#         positions[2*jj+1] = rr[jj]*np.sin(aa[jj])
#     if des.DistanceCheck(positions,radii,offset) == 0:
#         distFlag = 0
#         break
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
perturb = radius*1.0e-8
maxIter = 1000
clength = radius*1.0e-1

for i in range(5):
    algorithm = nlopt.LD_MMA
    solver = nlopt.opt(algorithm, len(positions))
    solver.set_min_objective(lambda x, grad: des.Objective(x, params))
    solver.set_maxeval(maxIter)
    solver.set_initial_step(perturb)
    solver.set_ftol_rel(10**(-i))
    x = solver.optimize(positions)
    
newPositions = x
obj = des.Objective(newPositions, params)
flag = 1
# how about using basin hopping?
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html
print(obj)
print(flag)
print(newPositions)
smat = scat.GetScatteringMatrix(newPositions,alphas,omega,cHost,epHost,nPort,obsRadius,normalize)
print(smat)
stif = np.linalg.inv(smat)

plt.figure()
plt.plot()

mult = np.matmul(smat,target)
maxstif = np.max(np.max(np.abs(stif)))
scat.printMat(np.real(smat),title+'_re',-1,1)
scat.printMat(np.imag(smat),title+'_im',-1,1)
scat.printMat(np.abs(smat),title+'_abs',0,1)
scat.printMat(np.real(mult),title+'_mult',0,1)
scat.printMat(np.real(stif),title+'_inv_re',-maxstif,maxstif)
scat.printMat(np.imag(stif),title+'_inv_im',-maxstif,maxstif)
scat.printMat(np.abs(stif),title+'_inv_abs',0,maxstif)
scat.printPos(newPositions,title+'_pos',controlRadius,-obsRadius,obsRadius,wavelength,atype)
fout = open(title+"_mult.data","w")
for ii in range(nPort):
    for jj in range(nPort):
        fout.write(f"{np.real(mult[ii,jj]):20.10e}\t{np.imag(mult[ii,jj]):20.10e}\t")
    fout.write(f"\n")
fout.close()
fout = open(title+"_scat.data","w")
for ii in range(nPort):
    for jj in range(nPort):
        fout.write(f"{np.real(smat[ii,jj]):20.10e}\t{np.imag(smat[ii,jj]):20.10e}\t")
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
