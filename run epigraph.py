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
title = '../analog/des_g_norm'
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
# with open(posName) as f:
#     data = [[float(num) for num in line.split()] for line in f]
# for jj in range(nAlpha):
#         positions[2*jj] = data[jj][0]+float(np.random.rand(1)-0.5)*10e-5
#         positions[2*jj+1] = data[jj][1]+float(np.random.rand(1)-0.5)*10e-5
#         atype[jj] = data[jj][2]

#atype = np.random.randint(2,size=nAlpha)
# atype = np.zeros(nAlpha)
# atype[0] = 1
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

for ii in range(300):
    rr = np.random.rand(nAlpha)*controlRadius
    aa = np.random.rand(nAlpha)*2.0*np.pi
    for jj in range(nAlpha):
        positions[2*jj] = rr[jj]*np.cos(aa[jj])
        positions[2*jj+1] = rr[jj]*np.sin(aa[jj])
    if des.DistanceCheck(positions,radii,offset) == 0:
        distFlag = 0
        break
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

eval_history = []
gl_hist = []
eval_it = [0]
max_f0 = [1e10]
best_position = positions

def global_obj(x, params):
    f0 = des.Objective(x,params)
    
    print(f0)
    gl_hist.append(f0)

    if f0<max_f0[0]:
        max_f0[0] = f0
        global best_position
        best_position = np.copy(x)

    return f0

#Starting Global
x = np.array(positions)

global_algo = nlopt.GN_ESCH#GD_STOGO_RAND

solver = nlopt.opt(global_algo, len(positions))
solver.set_lower_bounds(-controlRadius*np.ones(len(positions)))
solver.set_upper_bounds(controlRadius*np.ones(len(positions)))
solver.set_min_objective(lambda a, g: global_obj(a, params))
solver.set_maxeval(1000)
solver.set_ftol_rel(1e-6)
x[:] = solver.optimize(positions)

plt.figure()
plt.plot(np.log10(gl_hist))
plt.savefig(title+'_eval_history')
plt.close()

#Starting Local

x = np.insert(best_position, 0, 0)

algorithm = nlopt.LD_MMA

def f(x, grad):
    t = x[0]  # "dummy" parameter
    v = x[1:]  # design parameters
    if grad.size > 0:
        grad[0] = 1
        grad[1:] = 0
    return t

def c(result, x, gradient, params, perturb):
    t = x[0]  # dummy parameter
    v = x[1:] # design parameters

    f0 = des.Objective(v, params)
    my_grad = des.Gradient(v, params, f0, perturb) 

    print('Eval', eval_it, 'Cost', f0, flush=True)
    eval_history.append(f0)
    if f0<max_f0[0]:
        max_f0[0] = f0
        global best_position
        best_position = np.copy(v)

    # Assign gradients
    if gradient.size > 0:
        gradient[0,0] = -1  # gradient w.r.t. "t"
        gradient[0,1:] = my_grad*1e-5 # gradient w.r.t. objective

    result[:] = np.real(f0) - t
    eval_it[-1] += 1

run_num = 0

#while max_f0[0]>1:
#    if run_num > 0:
#        for ii in range(300):
#            rr = np.random.rand(nAlpha)*controlRadius
#            aa = np.random.rand(nAlpha)*2.0*np.pi
#            for jj in range(nAlpha):
#                positions[2*jj] = rr[jj]*np.cos(aa[jj])
#                positions[2*jj+1] = rr[jj]*np.sin(aa[jj])
#            if des.DistanceCheck(positions,radii,offset) == 0:
#                distFlag = 0
#                break
#        distFlag = des.DistanceCheck(positions,radii,offset)
#        if distFlag == 1:
#            print("rerun")
#            exit()
    
for i in range(0,10):
    print('Optimization Number: ', i)
    if i>0:
        x[1:] = best_position+np.random.randint(0,10, size=len(best_position))*10e-7
    solver = nlopt.opt(algorithm, len(positions) + 1)
    solver.set_min_objective(f)
    solver.set_maxeval(maxIter)
    solver.set_ftol_rel(1e-4)
    solver.add_inequality_mconstraint(
        lambda r, x, g: c(r, x, g, params, perturb), np.array([1e-4])
    )
    x[:] = solver.optimize(x)
run_num += 1

newPositions = best_position
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
plt.plot(np.log10(eval_history))
plt.savefig(title+'_eval_history')
plt.close()

plt.figure()
plt.plot()

mult = np.matmul(smat,target)
maxmult= np.max(np.real(mult))
maxstif = np.max(np.max(np.abs(stif)))
scat.printMat(np.real(smat),title+'_re',-1,1)
scat.printMat(np.imag(smat),title+'_im',-1,1)
scat.printMat(np.abs(smat),title+'_abs',0,1)
scat.printMat(np.real(mult)/maxmult,title+'_mult',0,1)
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
    fout.write(f"{positions[2*ii]:20.10e}\t{positions[2*ii+1]:20.10e}\t{atype[ii]}\n")
fout.close()
