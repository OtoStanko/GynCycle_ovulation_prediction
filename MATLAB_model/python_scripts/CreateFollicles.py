import numpy as np
import pandas as pd

def CreateFollicles(parafoll, paraPoi, tb, te):
    # Create normal distributed fsh sensitivities for each follicle
    with open('FSH.txt', 'w+') as fileID2:
        fileID2.write('Number    FSH\n')
        num_FSH_sensitivity_values = 10000
        FSHdistri = np.random.normal(loc=parafoll[7],
                                     scale=parafoll[8],
                                     size=num_FSH_sensitivity_values)
        
        for i in range(num_FSH_sensitivity_values):
            fsh = FSHdistri[i]
            fileID2.write(f'{i+1} {fsh}\n')

    # Create poisson distributed starting times for each follicle
    with open('StartTimesPoiss.txt', 'w+') as fileID:
        fileID.write('Start time\n')
        TotIntervall = te - tb
        timevec = np.random.poisson(lam=paraPoi[0], size=int(TotIntervall))
        
        for time in timevec:
            fileID.write(f'{time}\n')

    # Load StartNumbers and FSH Sensitivities from File
    data = pd.read_csv('StartTimesPoiss.txt', sep='\s+', skiprows=1)
    data2 = pd.read_csv('FSH.txt', sep='\s+', skiprows=1)

    NumValStart = data.shape[0]
    StartVec = np.zeros(NumValStart)
    for i in range(NumValStart):
        StartVec[i] = data.iloc[i, 0]

    NumValFSH = data2.shape[0]
    FSHVec = np.zeros(NumValFSH)
    for i in range(NumValFSH):
        FSHVec[i] = data2.iloc[i, 1]

    return FSHVec, StartVec

