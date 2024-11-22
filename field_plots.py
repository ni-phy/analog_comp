import numpy as np
import design as des
import gyroGen as gyro
import scat
import matplotlib.pyplot as plt

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


scat.PlotIncField(1,obsRadius,epHost,cHost,omega)

# posName = '../analog/GyroPDE/w_ga'+str(i)+'_gyro_pos.data'

# with open(posName) as f:
#     data = [[float(num) for num in line.split()] for line in f]

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