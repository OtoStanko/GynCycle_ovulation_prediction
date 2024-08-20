import numpy as np
import os
import CreateFollicles

from CreateFollicles import CreateFollicles
from Simulation import Simulation

def StartSimulation():
    runnum = 30
    ShowPlots = 1
    SaveSim = 0
    SavePlotStuff = 0
    SavePop = 0
    DirStuff = os.path.join(os.getcwd(), "..")
    #print(DirStuff)
    
    NormalCycle = 1
    LutStim = 0
    FollStim = 0
    DoubStim = 0
    Foll_ModelPop = 0
    Horm_ModelPop = 0
    
    global ModelPop_Params
    ModelPop_Params = []
    global ModelPop_CycleInfo
    ModelPop_CycleInfo = []

    for runind in range(1, runnum + 1):
        # technical params
        tb = 0
        te = 10
        # follicle params
        para = np.array([0, 17])#.reshape(-1, 1)
        parafoll = np.array([
            2,
            0.04 / 2,
            25,
            1,
            0.065 / (25 ** 2),  # k - strength of competition
            0.01,
            18,
            3 / 10,
            0.1,
            25,
            5,
            0.01,
            0.1,
            2,
            25
        ])#.reshape(-1, 1)
        # poisson distr params
        paraPoi = np.array([10/14, 0.25])#.reshape(-1, 1)
        #print(paraPoi)
        # ODE params
        Par = np.zeros((77, 1))
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
        Par[31] = 3
        Par[32] = 12
        Par[33] = 10
        Par[34] = 0.1904
        Par[35] = 0.0003
        Par[36] = 5
        Par[37] = 5
        Par[38] = 74.851
        Par[39] = 1.6e+04
        Par[40] = 2
        Par[41] = 2
        Par[42] = 15
        Par[43] = 5
        Par[44] = 0.02
        Par[45] = 0.12
        Par[46] = 0.00025
        Par[47] = 3
        Par[49] = 0.02
        Par[50] = 114.2474
        Par[51] = 500
        Par[52] = 3.5289
        Par[53] = 0.1000
        Par[54] = 61.0292
        Par[55] = 138.3032
        Par[56] = 5
        Par[57] = 15
        Par[58] = 20
        Par[59] = 0.25
        Par[60] = 120
        Par[61] = 0.06
        Par[62] = 15
        Par[72] = 80
        Par[73] = 4
        Par[74] = 20
        Par[75] = 0.2
        Par[76] = 0
        Par = Par.flatten()

        # init values
        file = 'yInitial.txt'
        delimiterIn = ';'
        fullFileName = os.path.join(DirStuff, file)
        yInitial = np.genfromtxt(fullFileName, delimiter=delimiterIn, skip_header=0)

        # init follicles
        y0Foll = 4
        #print(y0Foll)
        #print(yInitial)
        #StartValues = np.array([y0Foll] + yInitial.tolist()).reshape(1, -1)
        StartValues = np.concatenate(([y0Foll], yInitial))
        #print(StartValues)

        if Foll_ModelPop or Horm_ModelPop:
            FSHVec = np.genfromtxt('FSHS.txt', delimiter=',', skip_header=1)
            StartVec = np.genfromtxt('StartTimesPoiss.txt', delimiter=',', skip_header=1)
        else:
            FSHVec, StartVec = CreateFollicles(parafoll, paraPoi, tb, te)


        """
            Normal cycle
        """
        if NormalCycle:
            Stim = 0
            Simulation(para, paraPoi, parafoll, Par,
                       tb, te, StartValues, StartVec,
                       FSHVec, ShowPlots, SaveSim,
                       SavePlotStuff, DirStuff, Stim,
                       LutStim, FollStim, DoubStim,
                       Foll_ModelPop, Horm_ModelPop, runind)
        """
            Luteal phase stimulation
        """
        if LutStim:
            Stim = 1
            Par[64] = 0
            Par[65] = 13.387 / 2.6667
            Par[66] = 9.87
            Par[67] = 0.42
            Par[68] = 2.14
            Par[69] = 6.04
            Par[70] = 3.199
            Par[71] = 150
            Par[72] = Par[71] + 15
            Par = Par.T
            Simulation(para, paraPoi, parafoll, Par,
                       tb, te, StartValues, StartVec,
                       FSHVec, ShowPlots, SaveSim,
                       SavePlotStuff, DirStuff, Stim,
                       LutStim, FollStim, DoubStim,
                       Foll_ModelPop, Horm_ModelPop, runind)
        """
            Follicular phase stimulation
        """
        if FollStim:
            Stim = 1
            Par[64] = 0
            Par[65] = 13.387 / 2.6667
            Par[66] = 9.87
            Par[67] = 0.42
            Par[68] = 2.14
            Par[69] = 6.04
            Par[70] = 3.199
            Par[71] = 150
            Par[72] = Par[71] + 15
            Par = Par.T
            Simulation(para, paraPoi, parafoll,Par,
                       tb, te, StartValues, StartVec,
                       FSHVec, ShowPlots, SaveSim,
                       SavePlotStuff, DirStuff, Stim,
                       LutStim, FollStim, DoubStim,
                       Foll_ModelPop, Horm_ModelPop, runind)

        """
            Double stimulation
        """
        if DoubStim:
            Stim = 1
            paraPoi[0] = 5 / 14
            y0Foll = 4
            StartValues = np.array([y0Foll, yInitial]).T
            FSHVec, StartVec = CreateFollicles(parafoll, paraPoi, tb, te)

            Par[64] = 0
            Par[65] = 13.387 / 2.6667
            Par[66] = 9.87
            Par[67] = 0.42
            Par[68] = 2.14
            Par[69] = 6.04
            Par[70] = 3.199
            Par[71] = 150
            Par[72] = Par[71] + 15
            Par = Par.T
            Simulation(para, paraPoi, parafoll, Par,
                       tb, te, StartValues, StartVec,
                       FSHVec, ShowPlots, SaveSim,
                       SavePlotStuff, DirStuff, Stim,
                       LutStim, FollStim, DoubStim,
                       Foll_ModelPop, Horm_ModelPop, runind)

        #
        if Foll_ModelPop:
            Stim = 0
            parafoll[1] = np.random.lognormal(np.log(parafoll[1]), 0.15)
            parafoll[3] = np.random.lognormal(np.log(parafoll[3]), 0.15)
            parafoll[4] = np.random.lognormal(np.log(parafoll[4]), 0.15)
            Par[32] = np.random.lognormal(np.log(Par[32]), 0.15)
            Simulation(para, paraPoi, parafoll, Par,
                       tb, te, StartValues, StartVec,
                       FSHVec, ShowPlots, SaveSim,
                       SavePlotStuff, DirStuff, Stim,
                       LutStim, FollStim, DoubStim,
                       Foll_ModelPop, Horm_ModelPop, runind)

        if Horm_ModelPop:
            Stim = 0
            indices = [1, 2, 5, 6, 8, 9, 22, 24, 26, 27, 28,
                       34, 35, 36, 40, 41, 43, 45, 46, 47, 51, 73]
            for i in indices:
                Par[i] = np.random.lognormal(mean=np.log(Par[i]), sigma=0.15)
            Simulation(para, paraPoi, parafoll, Par,
                       tb, te, StartValues, StartVec,
                       FSHVec, ShowPlots, SaveSim,
                       SavePlotStuff, DirStuff, Stim,
                       LutStim, FollStim, DoubStim,
                       Foll_ModelPop, Horm_ModelPop, runind)

        if SavePop and runind % 10 == 0:
            FileName = 'ModelPopulation_Parameters.txt'
            fullFileName = os.path.join(DirStuff, FileName)
            M = np.loadtxt(fullFileName)
            M = np.column_stack((M, ModelPop_Params))
            np.savetxt(fullFileName, M, delimiter=',')
            ModelPop_Params = []

            FileName = 'ModelPopulation_CycleInfo.txt'
            fullFileName = os.path.join(DirStuff, FileName)
            M = np.loadtxt(fullFileName)
            M = np.column_stack((M, ModelPop_CycleInfo))
            np.savetxt(fullFileName, M, delimiter=',')
            ModelPop_CycleInfo = []

StartSimulation()
