import numpy as np
import os
import CreateFollicles

from CreateFollicles import CreateFollicles
from Simulation import Simulation

import parameters
from SimulationSettings import SimulationSettings

def StartSimulation():
    runnum = 1
    SIM_OUT_NUM = 4

    settings = SimulationSettings(ShowPlots = 1, SaveSim = 0, SavePlotStuff = 1,
                                  SavePop = 0, NormalCycle = 1, LutStim = 0,
                                  FollStim = 0, DoubStim = 0, Foll_ModelPop = 0,
                                  Horm_ModelPop = 0, workDir=os.path.join(os.getcwd(), ".."),
                                  outputDir=os.path.join(os.getcwd(), "../outputDir/"))

    global ModelPop_Params
    ModelPop_Params = []
    global ModelPop_CycleInfo
    ModelPop_CycleInfo = []

    for runind in range(1, runnum + 1):
        # integration time beginning and end
        tb = 0
        te = 3650 + 50
        # technical params
        # ODE function called to test(0) or not (1)
        # number of non-follicle equations (NO DRUG)
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

        # ODE param
        # imported from parameters.py file; Par variable
        Par = parameters.Par

        # init values
        file = 'yInitial.txt'
        delimiterIn = ';'
        fullFileName = os.path.join(settings.workDir, file)
        yInitial = np.genfromtxt(fullFileName, delimiter=delimiterIn, skip_header=0)

        # init follicles
        y0Foll = 4
        StartValues = np.concatenate(([y0Foll], yInitial))

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
                       FSHVec, Stim,
                       SIM_OUT_NUM, settings)

        if settings.savePop and runind % 10 == 0:
            FileName = 'ModelPopulation_Parameters.txt'
            fullFileName = os.path.join(settings.workDir, FileName)
            M = np.loadtxt(fullFileName)
            M = np.column_stack((M, ModelPop_Params))
            np.savetxt(fullFileName, M, delimiter=',')
            ModelPop_Params = []

            FileName = 'ModelPopulation_CycleInfo.txt'
            fullFileName = os.path.join(settings.workDir, FileName)
            M = np.loadtxt(fullFileName)
            M = np.column_stack((M, ModelPop_CycleInfo))
            np.savetxt(fullFileName, M, delimiter=',')
            ModelPop_CycleInfo = []

StartSimulation()
