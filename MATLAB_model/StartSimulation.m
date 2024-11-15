xfunction StartSimulation
%
%-----------------------------------------------------------------------
%
runnum        = 30;
%save simulation results
ShowPlots     = 0;
SaveSim       = 0;
SavePlotStuff = 0;
SavePop       = 0;
DirStuff      = '/Users/sophie/Documents/GynCycleModel_Pub2021/NonVec_Model/ModelPopulation/HormPopulation';
%select type of simulation
NormalCycle   = 1;
LutStim       = 0;
FollStim      = 0;
DoubStim      = 0;
Foll_ModelPop = 0;
Horm_ModelPop = 0;
%
%-----------------------------------------------------------------------
%
global ModelPop_Params
ModelPop_Params =[];
global ModelPop_CycleInfo
ModelPop_CycleInfo = [];

%
%-----------------------------------------------------------------------
%
for runind = 1:runnum
%
%-----------------------------------------------------------------------
%
%integration time beginning and end
%
    tb = 0;
    te = 300;
%
%-----------------------------------------------------------------------
%
% technical parameters
%
    para    = [];
    para(1) = 0;               %ODE function called to test(0) or not (1)
    para(2) = 17;              %number of non-follcile equations (NO DRUG)
    para    = para';
%
%-----------------------------------------------------------------------
%
%follicle parameters
%
    parafoll     = [];
    parafoll(1)  = 2;                  %v - fractal dimension
    parafoll(2)  = 0.04/2;             %gamma - growth rate
    parafoll(3)  = 25;                 %xi - max. diameter of follicles
    parafoll(4)  = 1;                  %mu - proportion of self harm
    parafoll(5)  = 0.065/ ...          %k - strength of competition
                  (parafoll(3)^parafoll(1));
    parafoll(6)  = 0.01;               %rho - rate of decline
    parafoll(7)  = 18;                 %min. ovulation size
    parafoll(8)  = (3/10);             %mean for FSH Sensitivity
    parafoll(9)  = 0.1;                %std.deviation for FSH Sensitivity %0.55
    parafoll(10) = 25;                 %threshold LH concentration for ovulation
    parafoll(11) = 5;                  %big but not ovulated follicle livetime
    parafoll(12) = 0.01;               %too slow foll growth
    parafoll(13) = 0.1;                %very slow foll growth
    parafoll(14) = 2;                  %max life time for a small slow growing follciles
    parafoll(15) = 25;                 %max follicle life time for a big follicles that start to rest
    parafoll     = parafoll';
%
%-----------------------------------------------------------------------
%
%parameters for poisson distribution
%
    paraPoi    = [];
    paraPoi(1) = 10/14;             %lambda - #Follikels/days
    paraPoi(2) = 0.25;              %intervall per day in which follicles appear
    paraPoi    = paraPoi';
%
%-----------------------------------------------------------------------
%
% ODE parameters
%
    Par     = zeros(74,1);
    Par(1)  = 16;                    %basal GnRH freq
    Par(2)  = 3;                     %T P4 on GnRH freq
    Par(3)  = 2;                     %n P4 on GnRH freq
    Par(4)  = 10;                    %n E2 on GnRH freq
    Par(5)  = 120;                   %T E2 on GnRH freq
    Par(6)  = 0.0056;                %basal GnRH mass
    Par(7)  = 2;                     %n E2 on GnRH mass
    Par(8)  = 100;                   %T E2 on GnRH mass
    Par(9)  = 9.6000;                %T E2 on GnRH mass
    Par(10) = 1;                     %n E2 on GnRH mass
    Par(11) = 322.1765;              %GnRH rezeptor binding constant
    Par(12) = 644.3530;              %GnRH rezeptor release constant
    Par(13) = 0.4475;                %GnRH clearance constant
    Par(14) = 3.2218;                %GnRH rezeptor complex inactivation
    Par(15) = 32.2176;               %GnRH rezeptor complex activation
    Par(16) = 8.9493e-05;            %GnRH receptor activation rate
    Par(17) = 32.2176;               %
    Par(18) = 0.0895;                %
    Par(19) = 32.2176;               %
    Par(20) = 3.2218;                %
    Par(21) = 0.0089;                %
    Par(22) = 192.2041;              %T E2 on LH
    Par(23) = 5;       %10           %n E2 on LH
    Par(24) = 18;                    %T GnRH freq on LH
    Par(25) = 5;                     %n GnRH freq on LH
    Par(26) = 1.8275e+03;            %basal LH production
    Par(27) = 7.3099e+03;            %E2 stimulated LH production
    Par(28) = 2.3708;                %T P4 on LH
    Par(29) = 1;                     %n P4 on LH
    Par(30) = 0.25;                  %scale hm_p4
    Par(31) = 1.25;                  %scale hp_GnRHa on FSH
    Par(32) = 3;                     %n start times
    Par(33) = 12;                    %T start times
    Par(34) = 10; 	           %basal LH release
    Par(35) = 0.1904;                %GnRH stimulated LH release
    Par(36) = 0.0003;                %T GnRH on LH
    Par(37) = 5;                     %n GnRH on LH
    Par(38) = 5;                     %Blood Volume
    Par(39) = 74.851;                %LH clearance rate
    Par(40) = 1.6e+04; 		   %basal FSH production
    Par(41) = 2;                     %T P4 on FSH
    Par(42) = 2;                     %n P4 on FSH
    Par(43) = 15;                    %T GnRH freq on FSH
    Par(44) = 5;                     %n GnRH freq on FSH
    Par(45) = 0.02; 	            %basal FSH release
    Par(46) = 0.12;                  %stimulated FSH release
    Par(47) = 0.00025; 	            %T GnRH on FSH
    Par(48) = 3;                     %n GnRH on FSH
    Par(50) = 0.02;                  %FSH blood to ovaries
    Par(51) = 114.2474;              %FSH clearance rate constant blood
    Par(52) = 500;                   %volumen ovaries
    Par(53) = 3.5289;                %FSH receptor binding constant
    Par(54) = 0.1000;                %FSH clearance rate constant ovaries
    Par(55) = 61.0292;               %FSH receptor release
    Par(56) = 138.3032;              %
    Par(57) = 5;                     %n follcile surface
    Par(58) = 15;                    %T follcile surface
    Par(59) = 20;                    %basal E2
    Par(60) = 0.25;     %0.15        %scale foll size effect
    Par(61) = 120;                   %scale E2 gaussian lutale phase
    Par(62) = 0.06;                  %
    Par(63) = 15;                    %
    Par(73) = 80;                    %T E2 on FSH
    Par(74) = 4;                     %n E2 on FSH
    Par(75) = 20;                    %basal E2 level
    Par(76) = 0.2;                   %basal P4 level
    Par(77) = 0;
    Par     = Par';
%
%-----------------------------------------------------------------------
%
%initial values
    file = 'yInitial.txt';
    delimiterIn=';';
    headerlinesIn=0;
    yInitial=importdata(file,delimiterIn,headerlinesIn);
%
%-----------------------------------------------------------------------
%
%initial follicles
%
    y0Foll = 4;
    StartValues = [y0Foll; yInitial]';

    if Foll_ModelPop || Horm_ModelPop
        FSHVec = csvread('/Users/sophie/Documents/GynCycleModel_Pub2021/NonVec_Model/ModelPopulation/ControlRun/FSHS.txt',1,0);
        StartVec = csvread('/Users/sophie/Documents/GynCycleModel_Pub2021/NonVec_Model/ModelPopulation/ControlRun/StartTimesPoiss.txt', 1, 0);
    else
        [FSHVec, StartVec] = CreateFollicles(parafoll,paraPoi,tb,te);
    end
%
%-----------------------------------------------------------------------
%
%Normal Cycle
%

if NormalCycle
    Stim = 0;
    Simulation(para,paraPoi,parafoll,Par,tb,te,StartValues,StartVec,FSHVec,ShowPlots,SaveSim,SavePlotStuff,DirStuff,Stim,LutStim,FollStim,DoubStim,Foll_ModelPop, Horm_ModelPop,runind);
end
%
%-----------------------------------------------------------------------
%
%Luteal Phase Stimulation: FSH/LH administartion (Menopur)
%
if (LutStim)
    Stim = 1;
    Par(64) = 0;                %set 1 if protocol starts
    Par(65) = 13.387/2.6667;    %D FSH
    Par(66) = 9.87;             %beta FSH
    Par(67) = 0.42;             %clearance rate FSH
    Par(68) = 2.14;             %D LH
    Par(69) = 6.04;             %beta LH
    Par(70) = 3.199;            %clearance rate LH
    Par(71) = 150;              %start of dosing - fiktive Zeitpunkte, werde in der Simulation gesetzt
    Par(72) = Par(71)+15;       %end time of dosing
    Par     = Par';
    Simulation(para,paraPoi,parafoll,Par,tb,te,StartValues,StartVec,FSHVec,ShowPlots,SaveSim,SaveFoll,DirStuff,Stim,LutStim,FollStim,DoubStim,Foll_ModelPop, Horm_ModelPop,runind);
end
%
%-----------------------------------------------------------------------
%
%Follicular Phase Stimulation: FSH/LH administartion (Menopur)
%
if (FollStim)
    Stim = 1;
    Par(64) = 0;                %set 1 if protocol starts
    Par(65) = 13.387/2.6667;      %D FSH
    Par(66) = 9.87;             %beta FSH
    Par(67) = 0.42;             %clearance rate FSH
    Par(68) = 2.14;             %D LH
    Par(69) = 6.04;             %beta LH
    Par(70) = 3.199;            %clearance rate LH
    Par(71) = 150;              %start of dosing - fiktive Zeitpunkte, werde in der Simulation gesetzt
    Par(72) = Par(71)+15;       %end time of dosing
    Par     = Par';
    Simulation(para,paraPoi,parafoll,Par,tb,te,StartValues,StartVec,FSHVec,ShowPlots,SaveSim,SaveFoll,DirStuff,Stim,LutStim,FollStim,DoubStimFoll_ModelPop, Horm_ModelPop,runind);
end
%
%-----------------------------------------------------------------------
%
%Double Stimulation: FSH/LH administartion (Menopur)
%
if (DoubStim)
    Stim = 1;

    paraPoi(1) = 5/14;
    y0Foll = 4;
    StartValues = [y0Foll; yInitial]';
    [FSHVec, StartVec] = CreateFollicles(parafoll,paraPoi,tb,te);

    Par(64) = 0;                %set 1 if protocol starts
    Par(65) = 13.387/2.6667;      %D FSH
    Par(66) = 9.87;             %beta FSH
    Par(67) = 0.42;             %clearance rate FSH
    Par(68) = 2.14;             %D LH
    Par(69) = 6.04;             %beta LH
    Par(70) = 3.199;            %clearance rate LH
    Par(71) = 150;              %start of dosing - fiktive Zeitpunkte, werde in der Simulation gesetzt
    Par(72) = Par(71)+15;       %end time of dosing
    Par     = Par';
    Simulation(para,paraPoi,parafoll,Par,tb,te,StartValues,StartVec,FSHVec,ShowPlots,SaveSim,SaveFoll,DirStuff,Stim,LutStim,FollStim,DoubStim,Foll_ModelPop, Horm_ModelPop,runind);
end
%
%-----------------------------------------------------------------------
%
if (Foll_ModelPop)
    Stim = 0;
    parafoll(2) = lognrnd(log(parafoll(2)),0.15);
    parafoll(4) = lognrnd(log(parafoll(4)),0.15);
    parafoll(5) = lognrnd(log(parafoll(5)),0.15);
    Par(33) = lognrnd(log(Par(33)),0.15);
    Simulation(para,paraPoi,parafoll,Par,tb,te,StartValues,StartVec,FSHVec,ShowPlots,SaveSim,SavePlotStuff,DirStuff,Stim,LutStim,FollStim,DoubStim,Foll_ModelPop,Horm_ModelPop,runind);
end
%
%-----------------------------------------------------------------------
%
if (Horm_ModelPop)
    Stim = 0;
    Par(1)  = lognrnd(log(Par(1)),0.15);
    Par(2)  = lognrnd(log(Par(2)),0.15);
    Par(5)  = lognrnd(log(Par(5)),0.15);
    Par(6)  = lognrnd(log(Par(6)),0.15);
    Par(8)  = lognrnd(log(Par(8)),0.15);
    Par(9)  = lognrnd(log(Par(9)),0.15);
    Par(22) = lognrnd(log(Par(22)),0.15);
    Par(24) = lognrnd(log(Par(24)),0.15);
    Par(26) = lognrnd(log(Par(26)),0.15);
    Par(27) = lognrnd(log(Par(27)),0.15);
    Par(28) = lognrnd(log(Par(28)),0.15);
    Par(34) = lognrnd(log(Par(34)),0.15);
    Par(35) = lognrnd(log(Par(35)),0.15);
    Par(36) = lognrnd(log(Par(36)),0.15);
    Par(40) = lognrnd(log(Par(40)),0.15);
    Par(41) = lognrnd(log(Par(41)),0.15);
    Par(43) = lognrnd(log(Par(43)),0.15);
    Par(45) = lognrnd(log(Par(45)),0.15);
    Par(46) = lognrnd(log(Par(46)),0.15);
    Par(47) = lognrnd(log(Par(47)),0.15);
    Par(51) = lognrnd(log(Par(51)),0.15);
    Par(73) = lognrnd(log(Par(73)),0.15);
    Simulation(para,paraPoi,parafoll,Par,tb,te,StartValues,StartVec,FSHVec,ShowPlots,SaveSim,SavePlotStuff,DirStuff,Stim,LutStim,FollStim,DoubStim,Foll_ModelPop, Horm_ModelPop,runind);
end
%
%-----------------------------------------------------------------------
%
if SavePop && mod(runind, 10) == 0
    FileName = sprintf('ModelPopulation_Parameters.txt');
    fullFileName = fullfile(DirStuff, FileName);
    M = load(fullFileName);
    M = [M ModelPop_Params];
    csvwrite(fullFileName,M);
    ModelPop_Params = [];

    FileName = sprintf('ModelPopulation_CycleInfo.txt');
    fullFileName = fullfile(DirStuff, FileName);
    M = load(fullFileName);
    M = [M ModelPop_CycleInfo];
    csvwrite(fullFileName,M);
    ModelPop_CycleInfo = [];
end
%
%-----------------------------------------------------------------------
%
end
