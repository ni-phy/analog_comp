#!/usr/bin/env python
from scipy import optimize
import scat as scat
import numpy as np
from autograd import numpy as npa
import gyroGen as gyro
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')


#########################################################################
## INVERSE ROUTINES   ###################################################
#########################################################################
#

def ObjGrad(trial, params, perturb):
    obj= Objective(trial, params)
    return obj, Gradient(trial, params, obj, perturb)

def Objective(trial, params):
    positions = trial
    alphas = params[0]
    omega = params[1]
    cHost = params[2]
    epHost = params[3]
    nPort = params[4]
    obsRadius = params[5]
    normalize = params[6]
    target = params[7]
    smat = scat.GetScatteringMatrix(positions,alphas,omega,cHost,epHost,nPort,obsRadius,normalize)
    obj = 0.0
    norm = np.max(np.real(np.matmul(target,smat)))
    for ii in range(nPort):
        val = np.matmul(target,smat[:,ii])-np.identity(nPort)[:,ii]
        obj += np.real(np.vdot(val,val))
    obj *= 0.5
    return obj
#
def Gradient(trial, params, obj, perturb):
    N = len(trial)
    gradient = npa.zeros(N,dtype=np.double)
    for ii in range(N):
        positions = npa.copy(trial)
        positions[ii] +=   perturb
        objA = Objective(positions, params)
        #positions[ii] -= 2*perturb
        #objB = Objective(positions, params)
        #gradient[ii] = (objA-objB)/(2.0*perturb)
        gradient[ii] = (objA-obj)/perturb
    return gradient
#
def NormalizedDistance(position1, position2, radius1, radius2):
    res = (position1[0]-position2[0])*(position1[0]-position2[0])
    res = res + (position1[1]-position2[1])*(position1[1]-position2[1])
    res = np.sqrt(res)/(radius1+radius2)
    return res
#
def DistanceCheck(positions,radii,offset=0):
    dim = 2
    n = int(len(positions)/dim)
    flag = 0
    if offset < 0:
        return flag
    for ii in range(n):
        for jj in range(ii+1,n):
            val = NormalizedDistance(positions[ii*dim:ii*dim+dim], positions[jj*dim:jj*dim+dim], radii[ii],radii[jj])
            if val < 1.0+offset:
                flag = 1
            if flag == 1:
                break
        if flag == 1:
            break
    return flag
#
def Inverse(trial,params,radii,offset,perturb,
            maxIter = 100000,clength = 1e-6,refRate=100,errTol=1.0e-7,
            title="test"):
    flag = 0
    nParam = len(trial)
    obj = Objective(trial,params)
    objMin = obj
    objZero = 1.0
    iiter = 0
    #print("Iter\tObjective\tBKIter")
    print(iiter,f"\t{obj/objZero:e}\t",".",f"\t{nParam:d}")
    lengthA = 1.0
    cgRef = True
    iterMin = 0
    obj_arr = []
    for iiter in range(1,maxIter):
        if obj/objZero < errTol:
            flag = 1
            break
        if iiter%refRate == 0:
            cgRef = True
        gradA = Gradient(trial,params,obj,perturb)
        if cgRef:
            direcA = -gradA
            lengthBaseA = clength/np.max(np.abs(direcA))
            cgRef = False
        else:
            cgBetaA = np.dot(gradA,gradA)/np.dot(gradAOld,gradAOld)
            direcA = -gradA + cgBetaA*direcA
        gradAOld = np.copy(gradA)
        lengthA = lengthBaseA
        for bkIter in range(10):
            newTrial = trial + lengthA*direcA
            newObj = Objective(newTrial,params)
            if newObj < obj and DistanceCheck(newTrial,radii,offset) == 0:
                break
            else:
                lengthA = lengthA * 0.9
        print(iiter,f"\t{newObj/objZero:e}\t{bkIter}\t{nParam}")
        obj_arr.append(newObj/objZero)
        if bkIter < 9:
            trial = np.copy(newTrial)
            obj = newObj
        else:
            break

    plt.figure()
    plt.plot(range(0,len(obj_arr)), obj_arr)
    plt.savefig(title+'_cost')
    plt.close()

    return trial,obj,flag
#########################################################################



