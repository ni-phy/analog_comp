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
import queue

def Singular_Gradient(trial, params, obj, perturb, ii, q):
    N = len(trial)
    gradient = npa.zeros(N,dtype=np.double)

    positions = npa.copy(trial)
    positions[ii] +=   perturb
    objA = des.Objective(positions, params)
    #positions[ii] -= 2*perturb
    #objB = Objective(positions, params)
    #gradient[ii] = (objA-objB)/(2.0*perturb)
    gradient[ii] = (objA-obj)/perturb
    q.put(gradient)
    return gradient

def par_grad(v, params, f0, perturb):
    a = {}
    gradients = npa.array(np.zeros(len(v)))

    q = queue.Queue()
    for i in range(len(v)):
        a[i] = threading.Thread(target=Singular_Gradient, args=[v, params, f0, perturb, i, q])

    for j in range(len(v)):
        task = a[j]
        task.start()
        grad = q.get()
        gradients[j] = grad[j]
    
    for k in range(len(v)):
        task = a[k]
        task.join()
    
    return gradients

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:


    mpl.use('Agg')
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['mathtext.rm'] = 'serif'

    #
    title = '../analog/trial/des_'
    print(title) #To keep track of slurm jobs
    targetName = 'test_A.data'
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
    wavelength = 2.0*np.pi*cHost/omega
    radius = wavelength * 0.01
    loss = 0.0
    alphaG = gyro.GetAlpha_Gyro(omega,omegap,omegac,cHost,epHost,radius,loss)
    alphaD = gyro.GetAlpha_Dielectric(omega,siEpsilon,cHost,epHost,radius,loss)
    # design parameters
    nAlpha = 20
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
    atype = np.zeros(nAlpha)
    # atype[:5] = 1
    #atype = np.ones(nAlpha)
    for ii in range(nAlpha):
        if atype[ii] == 0:
            alphas.append(alphaD)
        else:
            alphas.append(alphaG)
    radii = np.ones(nAlpha)*radius
    offset = 0.0
    controlRadius = 6*wavelength
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

    def global_obj(x, params, perturb):
        f0 = des.Objective(x,params)
        
        print('iter: ', len(gl_hist), "Cost: ", f0, flush=True)
        gl_hist.append(f0)

        if f0<max_f0[0]:
            max_f0[0] = f0
            global best_position
            best_position = np.copy(x)

        return f0

    f0 = des.Objective(positions, params)

    print('par_grad',par_grad(positions, params, f0, perturb))

    #Starting Global
    x = np.array(positions) 

    global_algo = nlopt.GN_ESCH#

    i=0
    # while i<5:
    solver = nlopt.opt(global_algo, len(positions))
    solver.set_lower_bounds(-controlRadius  *np.ones(len(positions)))
    solver.set_upper_bounds(controlRadius*np.ones(len(positions)))
    solver.set_min_objective(lambda a, g: global_obj(a, params, perturb))
    solver.set_maxeval(50)
    solver.set_ftol_rel(1e-5)
    x[:] = solver.optimize(x)
    # i += 1

    plt.figure()
    plt.plot(np.log10(gl_hist))
    plt.savefig(title+'_gl_eval_history')
    plt.close()

    #Intermediately Plots

    smat = scat.GetScatteringMatrix(best_position,alphas,omega,cHost,epHost,nPort,obsRadius,normalize)
    stif = np.linalg.inv(smat)

    plt.figure()
    plt.plot()

    maxabs = np.max(np.abs(smat))
    mult = np.matmul(smat,target)
    maxmult= np.max(np.real(mult))
    maxstif = np.max(np.max(np.abs(stif)))
    scat.printMat(np.real(smat),title+'_re',-1,1)
    scat.printMat(np.imag(smat),title+'_im',-1,1)
    scat.printMat(np.abs(smat),title+'_abs',0,)
    scat.printMat(np.real(mult),title+'_mult',-maxmult,maxmult)
    scat.printMat(np.real(stif),title+'_inv_re',-maxstif,maxstif)
    scat.printMat(np.imag(stif),title+'_inv_im',-maxstif,maxstif)
    scat.printMat(np.abs(stif),title+'_inv_abs',0,maxstif)
    scat.printPos(best_position,title+'_pos',controlRadius,-obsRadius,obsRadius,wavelength,atype)

    ###

    #Starting Local

    x = np.insert(best_position, 0, 1e2)

    algorithm = nlopt.LD_MMA
    lb = -controlRadius*np.ones(len(best_position))
    ub = controlRadius*np.ones(len(best_position))

    lb = np.insert(lb, 0, 0)
    ub = np.insert(ub, 0, np.inf)

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
        my_grad = par_grad(v, params, f0, perturb) 

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

    run_num = 0


    for i in range(0,10):
        print('Optimization Number: ', i)
        x[1:] = best_position+np.random.randint(0,10, size=len(best_position))*controlRadius*10e-4
        solver = nlopt.opt(algorithm, len(positions) + 1)
        solver.set_min_objective(f)
        solver.set_lower_bounds(lb)
        solver.set_upper_bounds(ub)
        solver.set_maxeval(maxIter)
        solver.set_ftol_rel(1e-5)
        solver.add_inequality_mconstraint(
            lambda r, x, g: c(r, x, g, params, perturb), np.array([1e-3])
        )
        x[:] = solver.optimize(x)

    newPositions = best_position
    obj = des.Objective(newPositions, params)
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
    mult = np.matmul(smat,target)
    maxmult= np.max(np.real(mult))
    maxstif = np.max(np.max(np.abs(stif)))
    scat.printMat(np.real(smat),title+'_re',-1,1)
    scat.printMat(np.imag(smat),title+'_im',-1,1)
    scat.printMat(np.abs(smat),title+'_abs',0,)
    scat.printMat(np.real(mult),title+'_mult',-maxmult,maxmult)
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
