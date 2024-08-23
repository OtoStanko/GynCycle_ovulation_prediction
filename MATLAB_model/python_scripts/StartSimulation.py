import numpy as np
import os
import CreateFollicles

from CreateFollicles import CreateFollicles
from Simulation import Simulation

import parameters
from SimulationSettings import SimulationSettings

def StartSimulation():
    runnum = 1

    settings = SimulationSettings(ShowPlots=1, SaveSim = 0, SavePlotStuff = 0,
                                  SavePop = 0, NormalCycle = 1, LutStim = 0,
                                  FollStim = 0, DoubStim = 0, Foll_ModelPop = 0,
                                  Horm_ModelPop = 0)
    DirStuff = os.path.join(os.getcwd(), "..")
    #print(DirStuff)

    global ModelPop_Params
    ModelPop_Params = []
    global ModelPop_CycleInfo
    ModelPop_CycleInfo = []

    for runind in range(1, runnum + 1):
        # integration time beginning and end
        tb = 0
        te = 40
        # technical params
        # ODE function called to test(0) or not (1)
        # number of non-follcile equations (NO DRUG)
        para = np.array([0, 15])
        # follicle params
        parafoll = np.array([
            2,                  # v - fractal dimension
            0.04 / 2,           # gamma - growth rate
            25,                 # xi - max. diameter of follicles
            1,                  # mu - proportion of self harm
            0.065 / (25 ** 2),  # k - strength of competition
            0.01,               # rho - rate of decline
            18,                 # min. ovulation size
            3 / 10,             # mean for FSH Sensitivity
            0.1,                # std.deviation for FSH Sensitivity %0.55
            25,                 # threshold LH concentration for ovulation
            5,                  # big but not ovulated follicle lifetime
            0.01,               # too slow foll growth
            0.1,                # very slow foll growth
            2,                  # max life time for a small slow growing follicles
            25                  # max follicle life time for a big follicles that start to rest
        ])
        # poisson distr params
        paraPoi = np.array([10/14, 0.25])
        #print(paraPoi)

        # ODE param
        # imported from parameters.py file; Par variable
        Par = parameters.Par

        # init values
        file = 'yInitial.txt'
        delimiterIn = ';'
        fullFileName = os.path.join(DirStuff, file)
        yInitial = np.genfromtxt(fullFileName, delimiter=delimiterIn, skip_header=0)

        # init follicles
        y0Foll = 4
        #print(y0Foll)
        #print(yInitial)
        StartValues = np.concatenate(([y0Foll], yInitial))
        #print(StartValues)

        if settings.foll_modelPop or settings.horm_modelPop:
            FSHVec = np.genfromtxt('FSHS.txt', delimiter=',', skip_header=1)
            StartVec = np.genfromtxt('StartTimesPoiss.txt', delimiter=',', skip_header=1)
        else:
            FSHVec, StartVec = CreateFollicles(parafoll, paraPoi, tb, te)


        """
            Normal cycle
        """
        if settings.normalCycle:
            Stim = 0
            Simulation(para, paraPoi, parafoll, Par,
                       tb, te, StartValues, StartVec,
                       FSHVec, DirStuff, Stim,
                       runind, settings)
        """
            Luteal phase stimulation
        """
        if settings.lut_stim:
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
                       FSHVec, DirStuff, Stim,
                       runind, settings)
        """
            Follicular phase stimulation
        """
        if settings.foll_stim:
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
                       FSHVec, DirStuff, Stim,
                       runind, settings)

        """
            Double stimulation
        """
        if settings.doub_stim:
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
                       FSHVec, DirStuff, Stim,
                       runind, settings)

        #
        if settings.foll_modelPop:
            Stim = 0
            parafoll[1] = np.random.lognormal(np.log(parafoll[1]), 0.15)
            parafoll[3] = np.random.lognormal(np.log(parafoll[3]), 0.15)
            parafoll[4] = np.random.lognormal(np.log(parafoll[4]), 0.15)
            Par[32] = np.random.lognormal(np.log(Par[32]), 0.15)
            Simulation(para, paraPoi, parafoll, Par,
                       tb, te, StartValues, StartVec,
                       FSHVec, DirStuff, Stim,
                       runind, settings)

        if settings.horm_modelPop:
            Stim = 0
            indices = [1, 2, 5, 6, 8, 9, 22, 24, 26, 27, 28,
                       34, 35, 36, 40, 41, 43, 45, 46, 47, 51, 73]
            for i in indices:
                Par[i] = np.random.lognormal(mean=np.log(Par[i]), sigma=0.15)
            Simulation(para, paraPoi, parafoll, Par,
                       tb, te, StartValues, StartVec,
                       FSHVec, DirStuff, Stim,
                       runind, settings)

        if settings.savePop and runind % 10 == 0:
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
