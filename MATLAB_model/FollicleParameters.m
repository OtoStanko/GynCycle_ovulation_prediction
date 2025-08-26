classdef FollicleParameters
%
%-----------------------------------------------------------------------
%
% follicle parameters
%
    properties
        %v - fractal dimension
        fractalDim = 2;
        %gamma - growth rate
        gamma = 0.02;
        %xi - max. diameter of follicles
        xi = 25;
        %mu - proportion of self harm
        mu = 1;
        %k - strength of competition
        %k = 0.065 / (xi^fractalDim)
        k = 0.065/(25^2);
        %rho - rate of decline
        rho = 0.01;
        %min. ovulation size
        minOvulationSize = 18;
        %mean for FSH Sensitivity
        meanFSHSensitivity = 0.3;
        %std.deviation for FSH Sensitivity %0.55
        stdFSHSensitivity = 0.1;
        %threshold LH concentration for ovulation
        cLHForOvulation = 25;
        %big but not ovulated follicle livetime
        bigFollLivetime = 5;
        %too slow foll growth
        tooSLowFollGrowth = 0.01;
        %very slow foll growth
        verySlowFollGrowth = 0.1;
        %max life time for a small slow growing follciles
        maxLifeSmallGrowingFolls = 2;
        %max follicle life time for a big follicles that start to rest
        maxLifeBigRestingFolls = 25;
    end
end