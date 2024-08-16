import numpy as np
import pandas as pd
from joblib import Parallel, delayed

def Parallel_StartSimulation():
    runnum = 100
    ShowStuff = 0
    SaveStuff = 0
    SaveFoll = 0
    SavePlotStuff = 1
    DirStuff = 'D:/MUNI/FI/_mgr/erasmus_03/research/GynCycle_newVersion/NormalRuns/'
    NormalCycle = 1
    LutStim = 0
    FollStim = 0
    DoubStim = 0

    M = 4
    Parallel(n_jobs=M)(delayed(run_simulation)(runind) for runind in range(1, runnum + 1))

def run_simulation(runind):
    tb = 0
    te = 150

    para = np.array([0, 17]).reshape(-1, 1)

    parafoll = np.array([
        2, 0.04 / 2, 25, 1, 0.065 / (25 ** 2),
        0.01, 18, 3 / 10, 0.1 * 1.05, 25,
        5, 0.01, 0.1, 2, 25
    ]).reshape(-1, 1)

    paraPoi = np.array([10 / 14, 0.25]).reshape(-1, 1)

    Par = np.zeros((74, 1))
    Par[0] = 16
    Par[1] = 3
    Par[2] = 2
    Par[3] = 10
    Par[4] = 120
    Par[5] = 0.0056
    Par[6] = 2
    Par[7] = 100
    Par[8] = 9.6000
    Par[9] = 1
    Par[10] = 322.1765
    Par[11] = 644.3530
    Par[12] = 0.4475
    Par[13] = 3.2218
    Par[14] = 32.2176
    Par[15] = 8.9493e-05
    Par[16] = 32.2176
    Par[17] = 0.0895
    Par[18] = 32.2176
    Par[19] = 3.2218
    Par[20] = 0.0089
    Par[21] = 192.2041
    Par[22] = 5
    Par[23] = 18
    Par[24] = 5
    Par[25] = 1.8275e+03
    Par[26] = 7.3099e+03
    Par[27] = 2.3708
    Par[28] = 1
    Par[29] = 0.25
    Par[30] = 1.25
    Par[31] = 10
    Par[32] = 0.1904
    Par[33] = 0.0003
    Par[34] = 5
    Par[35] = 5
    Par[36] = 74.851
    Par[37] = 1.6e+04
    Par[38] = 2
    Par[39] = 2
    Par[40] = 15
    Par[41] = 5
    Par[42] = 0.02
    Par[43] = 0.12
    Par[44] = 0.00025
    Par[45] = 3
    Par[46] = 0.02
    Par[47] = 114.2474
    Par[48] = 500
    Par[49] = 3.5289
    Par[50] = 0.1000
    Par[51] = 61.0292
    Par[52] = 138.3032
    Par[53] = 5
    Par[54] = 15
    Par[55] = 20
    Par[56] = 0.25
    Par[57] = 120
    Par[58] = 0.06
    Par[59] = 15
    Par[60] = 80
    Par[61] = 4
    Par[62] = 20
    Par[63] = 0.2
    Par = Par.flatten()

    yInitial = pd.read_csv('yInitial.txt', delimiter=';', header=None).values.flatten()

    y0Foll = 4
    StartValues = np.array([y0Foll] + yInitial.tolist()).reshape(1, -1)

    FSHVec, StartVec = CreateFollicles(parafoll, paraPoi, tb, te)

    Stim = 0
    if NormalCycle:
        Simulation(para, paraPoi, parafoll, Par, tb, te, StartValues, StartVec, FSHVec, ShowStuff, SaveStuff, SavePlotStuff, DirStuff, Stim, LutStim, FollStim, DoubStim, runind)

    if LutStim:
        Stim = 1
        Par[63] = 0
        Par[64] = 13.387 / 2.6667
        Par[65] = 9.87
        Par[66] = 0.42
        Par[67] = 2.14
        Par[68] = 6.04
        Par[69] = 3.199
        Par[70] = 150
        Par[71] = Par[70] + 15
        Par = Par.flatten()
        Simulation(para, paraPoi, parafoll, Par, tb, te, StartValues, StartVec, FSHVec, ShowStuff, SaveStuff, SaveFoll, DirStuff, Stim, LutStim, FollStim, DoubStim, runind)

    if FollStim:
        Stim = 1
        Par[63] = 0
        Par[64] = 13.387 / 2.6667
        Par[65] = 9.87
        Par[66] = 0.42
        Par[67] = 2.14
        Par[68] = 6.04
        Par[69] = 3.199
        Par[70] = 150
        Par[71] = Par[70] + 15
        Par = Par.flatten()
        Simulation(para, paraPoi, parafoll, Par, tb, te, StartValues, StartVec, FSHVec, ShowStuff, SaveStuff, SaveFoll, DirStuff, Stim, LutStim, FollStim, DoubStim)

    if DoubStim:
        Stim = 1
        paraPoi[0] = 5 / 14
        y0Foll = 4
        StartValues = np.array([y0Foll] + yInitial.tolist()).reshape(1, -1)
        FSHVec, StartVec = CreateFollicles(parafoll, paraPoi, tb, te)

        Par[63] = 0
        Par[64] = 13.387 / 2.6667
        Par[65] = 9.87
        Par[66] = 0.42
        Par[67] = 2.14
        Par[68] = 6.04
        Par[69] = 3.199
        Par[70] = 150
        Par[71] = Par[70] + 15
        Par = Par.flatten()
        Simulation(para, paraPoi, parafoll, Par, tb, te, StartValues, StartVec, FSHVec, ShowStuff, SaveStuff, SaveFoll, DirStuff, Stim, LutStim, FollStim, DoubStim, runind)

