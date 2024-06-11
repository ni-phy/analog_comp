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

gl_hist = []
#
title = '../analog/Fixed2/des_fixed'
targetName = 'test_g2_stif.data'
posName = 'test_g2_pos.data'
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

def global_obj(x, params, perturb):
    w = fixed_gyro(x, positions, atype)
    f0 = des.Objective(w,params)
    
    print('iter: ', len(gl_hist), "Cost: ", f0, flush=True)
    gl_hist.append(f0)

    if f0<max_f0[0]:
        max_f0[0] = f0
        global best_position
        best_position = np.copy(x)

    return f0

def c(result, x, gradient, params, perturb, atype, positions):
    t = x[0]  # dummy parameter
    v = x[1:] # design parameters

    w_gyro = fixed_gyro(v, positions, atype)
    f0 = des.Objective(w_gyro, params)
    my_grad_tot = des.Gradient(w_gyro, params, f0, perturb) 

    my_grad = remove_gyro(my_grad_tot , atype) #This also works for the gradients
    
    print('Eval', eval_it, 'Cost', f0, flush=True)

    eval_history.append(f0)
    if f0<max_f0[0]:
        max_f0[0] = f0
        global best_position
        best_position = np.copy(v)

    # Assign gradients
    if gradient.size > 0:
        gradient[0,0] = -1  # gradient w.r.t. "t"
        gradient[0,1:] = my_grad # gradient w.r.t. objective

    result[:] = np.real(f0) - t
    eval_it[-1] += 1

def fixed_gyro(opt_var, positions, atype):
    total_pos = np.zeros(len(positions))
    num_gyro = 0
    for i in range(len(atype)):
        if atype[i]==1:
            total_pos[2*i] = positions[2*i]
            total_pos[2*i+1] = positions[2*i+1]
            num_gyro += 1
        elif atype[i]==0:
            total_pos[2*i] = opt_var[2*i-2*num_gyro]
            total_pos[2*i+1] = opt_var[2*i-2*num_gyro+1]
        else:
            exit()
    
    return total_pos

def remove_gyro(positions, atype):
    diel_pos = []

    for i in range(len(atype)):
        if atype[i]==0:
            diel_pos.append(positions[2*i])
            diel_pos.append(positions[2*i+1])

    return np.array(diel_pos)



## atype == 0: dielectric; == 1: gyrotropic

#importing positions and adding random element
init_positions = np.zeros(nAlpha*2, dtype=np.double)
with open(posName) as f:
    data = [[float(num) for num in line.split()] for line in f]
for jj in range(nAlpha):
        init_positions[2*jj] = data[jj][0]
        init_positions[2*jj+1] = data[jj][1]
        atype[jj] = data[jj][2]

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

positions = np.zeros(nAlpha*2, dtype=np.double)
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
perturb = radius*1.0e-10
maxIter = 100
clength = radius*1.0e-1

eval_history = []
gl_hist = []
eval_it = [0]
max_f0 = [1e10]
diel_pos = remove_gyro(positions, atype)
best_position = diel_pos

# global_algo = nlopt.GN_ESCH#GD_STOGO_RAND#

# solver = nlopt.opt(global_algo, len(best_position))
# solver.set_lower_bounds(-controlRadius*np.ones(len(best_position)))
# solver.set_upper_bounds(controlRadius*np.ones(len(best_position)))
# solver.set_min_objective(lambda a, g: global_obj(a, params, perturb))
# solver.set_maxeval(5000)
# solver.set_ftol_rel(1e-5)
# x = solver.optimize(best_position)

# plt.figure()
# plt.plot(np.log10(gl_hist))
# plt.savefig(title+'_gl_eval_history')
# plt.close()

#Starting Local

x = np.insert(best_position, 0, 0)

algorithm = nlopt.LD_MMA
lb = -controlRadius*np.ones(len(best_position))
ub = controlRadius*np.ones(len(best_position))

lb = np.insert(lb, 0, -np.inf)
ub = np.insert(ub, 0, 0)

def f(x, grad):
    t = x[0]  # "dummy" parameter
    v = x[1:]  # design parameters
    if grad.size > 0:
        grad[0] = 1
        grad[1:] = 0
    return t

for i in range(0,10):
    print('Optimization Number: ', i)
    if i>0:
        x[1:] = best_position+(np.random.randint(0,10, size=len(best_position))-0.5)*controlRadius*10e-3

    solver = nlopt.opt(algorithm, len(diel_pos) + 1)
    solver.set_lower_bounds(lb)
    solver.set_upper_bounds(ub)
    solver.set_min_objective(f)
    solver.set_maxeval(maxIter)
    solver.set_ftol_rel(1e-5)
    solver.add_inequality_mconstraint(
        lambda r, x, g: c(r, x, g, params, perturb, atype, init_positions), np.array([1e-5])
    )
    x[:] = solver.optimize(x)

newPositions = fixed_gyro(best_position, init_positions, atype)
print(newPositions-np.array(init_positions))
obj = des.Objective(newPositions, params)
flag = 1
# how about using basin hopping?
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html
print('obj', obj)
print(flag)
print(fixed_gyro(diel_pos, init_positions, atype))
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
    fout.write(f"{positions[2*ii]:20.10e}\t{positions[2*ii+1]:20.10e}\t{atype[ii]}\n")
fout.close()
