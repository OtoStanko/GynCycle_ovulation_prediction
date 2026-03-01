classdef SimulationSettings
    properties
        % saving settings
        showPlots
        saveSim
        savePlotStuff
        savePop
        % type of simulation
        stim
        simulationType
        % directories settings
        OutputDir
        FSHVecPath
        StartVecPath
    end
    methods
        function obj = SimulationSettings(simulationType)
            switch simulationType
                case 'LutStim'
                    obj.simulationType = SimulationType.LutSim;
                case 'FollStim'
                    obj.simulationType = SimulationType.FollStim;
                case 'DoubleStim'
                    obj.simulationType = SimulationType.DoubleStim;
                case 'FollModelPop'
                    obj.simulationType = SimulationType.FollModelPop;
                case 'HormModelPop'
                    obj.simulationType = SimulationType.HormModelPop;
                case 'NormalCycle'
                    obj.simulationType = SimulationType.NormalCycle;
                otherwise
                    obj.simulationType = SimulationType.NormalCycle;
            end
            obj.showPlots     = 1;
            obj.saveSim       = 1;
            obj.savePlotStuff = 1;
            obj.savePop       = 0;
            % type of simulation
            obj.stim          = 0;
            % directories settings
            obj.OutputDir = './hormone_populations';
            obj.FSHVecPath = './ModelPopulation/ControlRun/FSHS.txt';
            obj.StartVecPath = './ModelPopulation/ControlRun/StartTimesPoiss.txt';
        end
    end
end
