#!/usr/bin/python
###################################################
## Gyrotropic cylinders                          ##
###################################################
__author__ = "Nikolas Hadjiantoni"               ##
__email__  = "nxh700@student.bham.ac.uk"         ##
###################################################

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
import threading 
from autograd import numpy as npa
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
import queue

def Singular_Gradient(trial, params, obj, perturb, ii):
    N = len(trial)
    gradient = npa.zeros(N,dtype=np.double)

    positions = npa.copy(trial)
    positions[ii] +=   perturb
    objA = des.Objective(positions, params)
    gradient[ii] = (objA-obj)/perturb
    return gradient

class GradientWrapper:
    def __init__(self, v, f0):
        self.v = v
        self.f0 = f0

    def singular_gradient(self, pos):
        return Singular_Gradient(self.v, params, self.f0, perturb, pos)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

mpl.use('Agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

nAlpha = 26

#
title = '../analog/GyroPDE10/no_epi_trial'+str(nAlpha)
if rank == 0:
    print(title) #To keep track of slurm jobs
targetName = 'square_gyro26_A.data'
# posName = '../analog/GyroPDE/square_gyro_pos.data'
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
wavelength = 2.0*np.pi*cHost/omega
radius = wavelength * 0.01
loss = 0.0
alphaG = gyro.GetAlpha_Gyro(omega,omegap,omegac,cHost,epHost,radius,loss)
alphaD = gyro.GetAlpha_Dielectric(omega,siEpsilon,cHost,epHost,radius,loss)
# design parameters
alphas = []
atype = np.zeros(nAlpha,dtype=np.int32)

## atype == 0: dielectric; == 1: gyrotropic

#importing positions and adding random element
positions = np.zeros(nAlpha*2, dtype=np.double)
# with open(posName) as f:
#     data = [[float(num) for num in line.split()] for line in f]
# for jj in range(nAlpha):
#         positions[2*jj] = data[jj][0]#+float(np.random.rand(1)-0.5)*10e-5
#         positions[2*jj+1] = data[jj][1]#+float(np.random.rand(1)-0.5)*10e-5
#         atype[jj] = data[jj][2]

#atype = np.random.randint(2,size=nAlpha)
atype = np.zeros(nAlpha)
atype[:20] = 1
#atype = np.ones(nAlpha)
for ii in range(nAlpha):
    if atype[ii] == 0:
        alphas.append(alphaD)
    else:
        alphas.append(alphaG)
radii = np.ones(nAlpha)*radius
offset = 0.0
controlRadius = 10*wavelength
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
normalize = False
#
with open(targetName) as f:
    data = [[float(num) for num in line.split()] for line in f]
target = np.zeros((nPort,nPort),dtype=np.cdouble)
for ii in range(nPort):
    for jj in range(nPort):
        target[ii,jj] = data[ii][2*jj]+1j*data[ii][2*jj+1]

params = [alphas,omega,cHost,epHost,nPort,obsRadius,normalize,target]

if rank ==0:
    obj = des.Objective(np.insert(positions,0,1), params)
    print('pre-opt obj ', obj)

positions = comm.bcast(positions,root=0)
params = comm.bcast(params,root=0)

perturb = radius*1.0e-10
maxIter = 1000
clength = radius*1.0e-1

eval_history = []
gl_hist = []
eval_it = [0]
max_f0 = [1e10]
best_position = positions
best_mult = [1]
best_array = []
trailing_av_array = []

def global_obj(x, params, perturb):

    f0 = des.Objective(x,params)
    
    print('iter: ', len(gl_hist), "Cost: ", f0, flush=True)
    gl_hist.append(f0)
    trailing_av_array.append(np.average(gl_hist))

    if f0<max_f0[0]:
        max_f0[0] = f0
        global best_position
        best_position = np.copy(x[1:])
        best_mult = np.copy(x[0])
    
    best_array.append(max_f0[0])
    
    return f0

f0 = des.Objective(np.insert(positions,0,1), params)

if rank==0:
    #Starting Global
    x = np.array(positions) 

    global_algo = nlopt.GN_ESCH#

    lb = -controlRadius*np.ones(len(positions))
    ub = controlRadius*np.ones(len(positions))

    x = np.insert(x, 0, 1)
    lb = np.insert(lb, 0, 0.5)
    ub = np.insert(ub, 0, 3)

    # i=0
    solver = nlopt.opt(global_algo, len(x))
    solver.set_lower_bounds(lb)
    solver.set_upper_bounds(ub)
    solver.set_min_objective(lambda a, g: global_obj(a, params, perturb))
    solver.set_maxeval(2000)
    solver.set_ftol_rel(1e-4)
    x[:] = solver.optimize(x)
    # i += 1

    plt.figure()
    plt.plot(np.log10(best_array), label='Best Value')
    plt.plot(np.log10(trailing_av_array), label='Trailing Average')
    plt.legend()
    plt.savefig(title+'_gl_eval_history')
    plt.close()

    #Intermediately Plots

    smat = scat.GetScatteringMatrix(best_position,alphas,omega,cHost,epHost,nPort,obsRadius,normalize)
    stif = np.linalg.inv(smat)

    plt.figure()
    plt.plot()

    smat = scat.GetScatteringMatrix(best_position,alphas,omega,cHost,epHost,nPort,obsRadius,normalize)
    stif = np.linalg.inv(smat)

    maxabs = np.max(np.abs(smat))
    mult = np.zeros((nPort, nPort), dtype=np.complex128)#np.matmul(best_mult[0]*smat,target)
    for ii in range(nPort):
        val = best_mult*np.matmul(target,smat[:,ii])#-np.identity(nPort)[:,ii]
        mult[ii] = val


    maxmult= np.max(np.real(mult))
    maxstif = np.max(np.max(np.abs(stif)))
    maxsmat = np.max(np.max(np.abs(smat)))
    print('multiplier', best_mult)
    scat.printMat(np.real(smat),title+'_re',-maxsmat,maxsmat)
    scat.printMat(np.imag(smat),title+'_im',-maxsmat,maxsmat)
    scat.printMat(np.abs(smat),title+'_abs',0,)
    scat.printMat(np.real(mult),title+'_mult',0,1)
    scat.printMat(np.real(stif),title+'_inv_re',-maxstif,maxstif)
    scat.printMat(np.imag(stif),title+'_inv_im',-maxstif,maxstif)
    scat.printMat(np.abs(stif),title+'_inv_abs',0,maxstif)
    scat.printPos(best_position,title+'_pos',controlRadius,-obsRadius,obsRadius,wavelength,atype)
    
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
        fout.write(f"{best_position[2*ii]:20.10e}\t{best_position[2*ii+1]:20.10e}\t{atype[ii]}\n")
    fout.close()

    ###

    #Starting Local


def c(v, gradient, params, perturb):
    f0 = des.Objective(v, params)
    
    # In your main function:
    if __name__ == '__main__':
        with MPIPoolExecutor() as pool:
            wrapper = GradientWrapper(v, f0)
            my_grad = np.sum(list(pool.map(wrapper.singular_gradient, range(len(v)))), axis=0)
    # my_grad = des.Gradient(x, params, f0, perturb)
    if gradient.size > 0:
        gradient[:] = my_grad
        print(gradient)
    print('Eval', eval_it, 'Cost', f0, flush=True)
    eval_history.append(f0)
    if f0<max_f0[0]:
        max_f0[0] = f0
        global best_position
        best_position = np.copy(v[1:])
        best_mult[0] = v[0]

    eval_it[-1] += 1
    return f0


if rank==0:

    algorithm = nlopt.LD_MMA

    for i in range(0,7):
        print('Optimization Number: ', i)
        if i>0:
            x[1:] = best_position+np.random.randint(-10,10, size=len(best_position))*controlRadius*10e-4
            for i in range(1, len(x)):
                if x[i] > controlRadius:
                    x[i] -= np.random.randint(-10,0, size=1)*controlRadius*10e-3
                elif x[i] < -controlRadius:
                    x[i] += np.random.randint(0,10, size=1)*controlRadius*10e-3
        solver = nlopt.opt(algorithm, len(x))
        solver.set_min_objective(lambda a, g: c(a, g, params, perturb))
        solver.set_lower_bounds(lb)
        solver.set_upper_bounds(ub)
        solver.set_maxeval(maxIter)
        solver.set_ftol_rel(1e-4)
        x[:] = solver.optimize(x)

        newPositions = np.copy(best_position)
        obj = des.Objective(np.insert(newPositions, 0, best_mult[0]), params)
        flag = 1
        # how about using basin hopping?
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html
        print('obj', obj)
        print(flag)
        print(newPositions)
        smat = scat.GetScatteringMatrix(newPositions,alphas,omega,cHost,epHost,nPort,obsRadius,normalize)
        stif = np.linalg.inv(smat)

        plt.figure()
        plt.plot(np.log10(eval_history))
        plt.savefig(title+'_eval_history')
        plt.close()

        plt.figure()
        plt.plot()

        maxabs = np.max(np.abs(smat))
        print('multiplier', best_mult)

        mult = np.zeros((nPort, nPort), dtype=np.complex128)#np.matmul(best_mult[0]*smat,target)
        for ii in range(nPort):
            val = best_mult*np.matmul(target,smat[:,ii])#-np.identity(nPort)[:,ii]
            mult[ii] = val

        maxmult= np.max(np.real(mult))
        maxstif = np.max(np.max(np.abs(stif)))
        scat.printMat(np.real(smat),title+'_re',-0.4,0.4)
        scat.printMat(np.imag(smat),title+'_im',-0.4,0.4)
        scat.printMat(np.abs(smat),title+'_abs',0,)
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
            fout.write(f"{newPositions[2*ii]:20.10e}\t{newPositions[2*ii+1]:20.10e}\t{atype[ii]}\n")
        fout.close()
        fout = open(title+"_multiplier.data","w")
        fout.write(f"{best_mult}\n")
        fout.close()

