import os

class SimulationSettings:
    def __init__(self, ShowPlots=0, SaveSim=0, SavePlotStuff=0, SavePop=0,
                 NormalCycle=1, LutStim=0, FollStim=0, DoubStim=0,
                 Foll_ModelPop=0, Horm_ModelPop=0, workDir=None, outputDir=None):
        # save simulation results
        self.showPlots = ShowPlots
        self.saveSim = SaveSim
        self.savePlotsStuff = SavePlotStuff
        self.savePop = SavePop
        # select type of simulation
        self.normalCycle = NormalCycle
        self.lut_stim = LutStim
        self.foll_stim = FollStim
        self.doub_stim = DoubStim
        self.foll_modelPop = Foll_ModelPop
        self.horm_modelPop = Horm_ModelPop
        # directories
        if not os.path.exists(workDir):
            print(f"Directory '{workDir}' doesn't exit. Value of workDir set to None")
            self.workDir = None
        else:
            self.workDir = workDir
        self.outputDir = outputDir
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
            print(f"Directory '{outputDir}' created.")
        else:
            print(f"Directory '{outputDir}' already exists.")

