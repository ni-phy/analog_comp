import numpy as np
import design as des
import gyroGen as gyro
import scat
import matplotlib.pyplot as plt

num_particles = []
cost = []
mult = [2.024320459191347, 2.4182981895177122, 2.0541379107884494,1.2496516335310068, 
        1.3179330726905498, 0.8417425801507624, 0.8118545461844427, 1.925821210012961,
        0.7433214598486603, 0.7295588604274161]

targetName = 'trial_gyro_A.data'
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
controlRadius = 7.5*wavelength

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

diag_err = []
worst_err = []

for i in range(2,13):
    posName = '../analog/GyroPDE/w_ga'+str(i)+'_gyro_pos.data'

    with open(posName) as f:
        data = [[float(num) for num in line.split()] for line in f]
    
    nAlpha = len(data)
    alphas = []

    matrixName= '../analog/GyroPDE/w_ga'+str(i)+'_gyro_mult.data'
    with open(matrixName) as f:
        data = [[float(num) for num in line.split()] for line in f]
    matrix = np.zeros((nPort,nPort),dtype=np.cdouble)
    for ii in range(nPort):
        for jj in range(nPort):
            matrix[ii,jj] = data[ii][2*jj]+1j*data[ii][2*jj+1]
    print(np.shape(matrix))
    # atype = np.zeros(nAlpha,dtype=np.int32)

    # positions = np.zeros(nAlpha*2, dtype=np.double)


    # for jj in range(nAlpha):
    #     positions[2*jj] = data[jj][0]
    #     positions[2*jj+1] = data[jj][1]
    #     atype[jj] = data[jj][2]

    # for ii in range(nAlpha):
    #     if atype[ii] == 0:
    #         alphas.append(alphaD)
    #     else:
    #         alphas.append(alphaG)
    
    # params = [alphas,omega,cHost,epHost,nPort,obsRadius,normalize,target]

    # obj = des.Objective(np.insert(positions,0,0.7615706505091185), params) # Adding a multiplier=1

    num_particles.append(nAlpha)
    # cost.append(obj)

    # matrix = []
    # smat = scat.GetScatteringMatrix(positions,alphas,omega,cHost,epHost,nPort,obsRadius,normalize)
    # for ii in range(nPort):
    #     val = 0.7615706505091185*np.matmul(target,smat[:,ii])-np.identity(nPort)[:,ii]
    #     matrix.append(val)

    diag = 0
    min_diag = 1
    for ii in range(nPort):
        diag += np.real((matrix[ii, ii] - 1))**2
        if np.real(matrix[ii, ii])<min_diag:
            min_diag = matrix[ii, ii]
    diag_err.append(diag)
    
    max_off_diag = 0
    for ii in range(nPort):
        for jj in range(nPort):
            if ii != jj and np.real(matrix[ii,jj])>max_off_diag:
                max_off_diag = matrix[ii,jj]
    
    worst_err.append(np.abs(min_diag)-np.abs(max_off_diag))

plt.plot(num_particles, diag_err)
plt.title('RMSE of Re(Diag) to I')
plt.savefig("diag_rmse")
plt.close()

plt.plot(num_particles, worst_err)
plt.title(' Min re(Diag) - Max re(off Diag)')
plt.savefig("diag_err")
plt.close()
