classdef SimulationSettings
    properties
        % saving settings
        showPlots     = 0;
        saveSim       = 0;
        savePlotStuff = 0;
        savePop       = 0;
        % type of simulation
        normalCycle   = 1;
        stim          = 0;
        lutStim       = 0;
        follStim      = 0;
        doubStim      = 0;
        foll_ModelPop = 0;
        horm_ModelPop = 0;
        % directories settings
        OutputDir = './ModelPopulation/HormPopulation';
        FSHVecPath = './ModelPopulation/ControlRun/FSHS.txt';
        StartVecPath = './ModelPopulation/ControlRun/StartTimesPoiss.txt'
    end
end