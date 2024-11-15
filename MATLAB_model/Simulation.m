% Calculation and solving of the ODE
% needs:
%%Parameter(programming, Poisson distribution, ODE calculations),
%%integration starttime, endtime, initial values,
%%poisson distributed starttimes of the follicles, normal distributed FSH
%%sensitivities of the follicles, ShowStuff, SaveStuff, DirStuff
function [res] = Simulation(para,paraPoi,parafoll,Par,tb,te,StartValues,StartTimes,FSHVec,ShowPlots,SaveSim,SavePlotStuff,DirStuff,Stim,LutStim,FollStim,DoubStim,Foll_ModelPop, Horm_ModelPop, runind)
%
%-----------------------------------------------------------------------
%
%integration period
tspan=[tb,te];
%variable for the current time
t=tb;
%Timepoint of last ovulation, initiated as 14 will be change in the cause
%of the simulation
Tovu = 14;
%
%-----------------------------------------------------------------------
%
%initial values ACHTUNG
y0Foll = StartValues(1);    %startsize of the follicle
y0E = StartValues(2);       %startvalue estradiol
y0P4 = StartValues(3);      %startvalue progesterone
y0LH = StartValues(10);      %startvalue LH
y0FSH = StartValues(8);     %startvalue FSH
y0 = StartValues';
%
%-----------------------------------------------------------------------
%
%values for tracking the follicles
FollCounter = 1;

%class to save follicles and their properties
Follicles = FollicleClass(y0Foll,FSHVec(FollCounter),t);
FollCounter = FollCounter + 1;

%values needed for the first integrations
TimeCounter=1;
NextStart=StartTimes(TimeCounter);
TimeFol = t';

%arrays to save times when new follicles emerge or when they can't emerge
NewFollicle = [];
NoNewFollicle = [];
LastYValues = [];
result=zeros(5,2);
%
%-----------------------------------------------------------------------
%
%variables for drug Administration
dd1 = 0;
dosing_events1 = [];
firstExtraction = 0;
%
%-----------------------------------------------------------------------
%
global ModelPop_Params
global ModelPop_CycleInfo
%
%-----------------------------------------------------------------------
%
%tracking the concentrations of important hormone species
%E2 Concentration
E2.Time = t;
E2.Y = y0(end-16);

%P4 Concentration
P4.Time = t;
P4.Y = y0(end-15);

%FSH Concentration
FSH.Time = t;
FSH.Y = y0(end-10);

%FSH Rezeptor
FSHRez.Time = t;
FSHRez.Y = y0(end-14);

%LH Concentration
LH.Time = t;
LH.Y = y0(end-8);

%GnRH Concentration
GnRH.Time = t;
GnRH.Y = y0(end-2);

%GnRH Concentration
GnRHRezA.Time = t;
GnRHRezA.Y = y0(end-5);

%GnRH Concentration
FSHmed.Time = t;
FSHmed.Y = y0(end);

%Yall
solutions.Time = t;
solutions.Y=y0(2:end)';

%
%-----------------------------------------------------------------------
%
while (t<te)

    t

    %follicles recruitment depending on FSH concentration in the system ->
    %FSH window idea
    %HIER MUSS MED FSH WIEDER DAZU
    fshAll = y0(end-10)+y0(end);
    fshimp = fshAll^Par(32)/(fshAll^Par(32) + Par(33)^Par(32));
    timevec=poissonproc(paraPoi(1)+6*paraPoi(1)*fshimp,[t,te]);

    %set integration period for the current follicle
     if (~isempty(timevec))
         NextStart=timevec(1);
         tspan=[t,NextStart];
     else
        tspan=[t,te];
    end

    %determine number of follicles
    NumFollicles=size(y0,1)-para(2);

    %set mass matrix for DAE system
    n=length(y0);
    M=eye(n,n);
    M(NumFollicles+1,NumFollicles+1)=0;     %alg. eq. for E2
    M(NumFollicles+2,NumFollicles+2)=0;     %alg. eq. for P4
    M(NumFollicles+16,NumFollicles+16)=0;     %alg. eq. for LH med
    M(NumFollicles+17,NumFollicles+17)=0;     %alg. eq. for FSH med

    %event function stops the integration, when ever an ovulation takes
    %place within the intervall tspan
    options = odeset('Mass',M,'events',@(t,y)EvaluateFollicle(t,y,para,parafoll,LH));

    %search for dosing events in [tspan(1),tspan(2)]:
    dosing_timeIdx=[];
    if ~isempty(dosing_events1)
        dosing_timeIdx=intersect(find(dosing_events1(1,:)>tspan(1)),find(dosing_events1(1,:)<=tspan(2)));
    end

    %solve differential equations
    para(1) = 0;
    Y=[];
    T=[];
    if ~isempty(dosing_timeIdx)
        tstart=tspan(1);
        tend=tspan(2);
        yInitial=y0;
        for i=1:length(dosing_timeIdx)
            tspan=[tstart;dosing_events1(1,dosing_timeIdx(i))];
            if tspan(1)~=tspan(2)
                [ ti, yi ] = ode15s(@(t,y)FollicleFunction(t,y,Tovu,Follicles,para,parafoll,Par,dd1,Stim,LutStim,FollStim,DoubStim,firstExtraction), tspan, yInitial, options);
                T=[T;ti(2:end)];
                Y=[Y;yi(2:end,:)];
                tstart=T(end);
                yInitial=Y(end,:);
            end
            dd1 = dosing_events1(2,dosing_timeIdx(i));
        end
        tspan=[T(end);tend];
        [ ti, yi ] = ode15s(@(t,y)FollicleFunction(t,y,Tovu,Follicles,para,parafoll,Par,dd1,Stim,LutStim,FollStim,DoubStim,firstExtraction), tspan, yInitial, options);
        T=[T;ti(2:end)];
        Y=[Y;yi(2:end,:)];
    else
        [T,Y] = ode15s(@(t,y)FollicleFunction(t,y,Tovu,Follicles,para,parafoll,Par,dd1,Stim,LutStim,FollStim,DoubStim,firstExtraction),tspan,y0,options);
    end

    for i = 1:Follicles.NumActive
        %saves all times of the foll that was active during last run
        Follicles.Follicle{Follicles.Active(i)}.Time = [Follicles.Follicle{Follicles.Active(i)}.Time; T(2:end)];
        %saves all sizes of the foll that was active during last run
        Follicles.Follicle{Follicles.Active(i)}.Y = [Follicles.Follicle{Follicles.Active(i)}.Y; Y(2:end,i)];
    end

    if (LutStim)
        %Werte f�r die Medikamentengabe setzen
        if Par(71) == Tovu && Par(64) == 0
            for i = 1:Follicles.NumActive
                if Follicles.Follicle{Follicles.Active(i)}.Destiny == -1
                %matrix of dosing times and drugs added: row1: times, row2: drug, row 3, dose
                    if t-Par(71) >= 1 && t-Par(71) < 4
                        Par(71) = ceil(t);
                        Par(72)=Par(71)+15;
                        %Menopur
                        numDoses=Par(72)-Par(71)+1;
                        dosing_events1=[[Par(71):Par(72)];[1:numDoses]];
                        Par(64) = 1;
                    end
                end
            end
        end
    end

    if (FollStim)
        if Par(71)== Tovu && Par(64) == 0
            if t > Par(71)+14
                for i = 1:Follicles.NumActive
                    if Follicles.Follicle{Follicles.Active(i)}.Y(end) >= 14
                        %matrix of dosing times and drugs added: row1: times, row2: drug, row 3, dose
                        Par(71)=ceil(t)+1;
                        Par(72)=Par(71)+14;
                        numDoses=Par(72)-Par(71)+1;
                        dosing_events1=[[Par(71):Par(72)];[1:numDoses]];
                        Par(64) = 1;
                    end
                end
            end
        end
    end

    if(DoubStim)
        if Par(71)== Tovu && Par(64) == 0
            Par(71)=ceil(t)+20;
            Par(72)=Par(71)+15;
            numDoses=Par(72)-Par(71)+1;
            dosing_events1=[[Par(71):Par(72)];[1:numDoses]];
            Par(64) = 1;
        end
    end

    %saves the measuring times of the active foll.
    TimeFol = [TimeFol; T(2:end)];
    %saves last Y values of the follicles
    LastYValues = Y(end,1:end)';

    %save values for E2
    E2.Time = [E2.Time; T(2:end)];
    E2.Y = [E2.Y; Y(2:end,end-16)];
    %save values for P4
    P4.Time = [P4.Time; T(2:end)];
    P4.Y = [P4.Y; Y(2:end,end-15)];
    %save values for LH
    LH.Time = [LH.Time; T(2:end)];
    LH.Y = [LH.Y; Y(2:end,end-8)];
    %save values for FSH
    FSH.Time = [FSH.Time; T(2:end)];
    FSH.Y = [FSH.Y; Y(2:end,end-10)];
    %save values for FSH REzeptor
    FSHRez.Time = [FSHRez.Time; T(2:end)];
    FSHRez.Y = [FSHRez.Y; Y(2:end,end-14)];
    %save values for GnRH
    GnRH.Time = [GnRH.Time; T(2:end)];
    GnRH.Y = [GnRH.Y; Y(2:end,end-2)];
    %GnRH Concentration
    GnRHRezA.Time = [GnRHRezA.Time; T(2:end)];
    GnRHRezA.Y = [GnRHRezA.Y; Y(2:end,end-5)];
    %save values for GnRH
    FSHmed.Time = [FSHmed.Time; T(2:end)];
    FSHmed.Y = [FSHmed.Y; Y(2:end,end)];
    %save solutions
    solutions.Time = [solutions.Time; T(2:end)];
    solutions.Y = [solutions.Y; Y(2:end,NumFollicles+1:end)];

    %no ovulation (=no event) occured
    if T(end)==tspan(2)
        %initialize new follicle
        %Set initial values for new follicle
        Follicle1.Y = y0Foll;
        if( ~isempty(FSHVec) )
            Follicle1.FSHSensitivity = FSHVec(FollCounter);
            FollCounter = FollCounter + 1;
        end
        FSHSSave = Follicles.ActiveFSHS;
        Follicles.ActiveFSHS = [Follicles.ActiveFSHS Follicle1.FSHSensitivity];
        %Test if Follicle(s) could survive
        %(slope of growth-function positive or negative)
        testyvalues = LastYValues(1:(end-para(2)));
        testyvalues = [testyvalues; Follicle1.Y; LastYValues(end+1-para(2):end)];
        para(1) = 1;
        testyslope = FollicleFunction(T(end),testyvalues,Tovu,Follicles,para,parafoll,Par,dd1,Stim,LutStim,FollStim,DoubStim,firstExtraction);
        %if follicle got chance to survive->initiate new follicle and update
        %follicles-vector
        if( testyslope(end-para(2)) > 0 )
            Follicle1.Time = [ T(end) ];
            Follicle1.TimeDecrease = 0;
            Follicle1.Destiny = -1;
            Follicles.Number = Follicles.Number + 1;
            Follicle1.Number = Follicles.Number;
            Follicles.NumActive = Follicles.NumActive + 1;
            Follicles.Active = [Follicles.Active Follicle1.Number];
            Follicles.Follicle = {Follicles.Follicle{1:end} Follicle1};
            NewFollicle = [NewFollicle T(end)];
            LastYValues = testyvalues;
        else
            %no chance to survive save in NoNewFollicle for statistic
            Follicles.ActiveFSHS = FSHSSave;
            NoNewFollicle = [NoNewFollicle T(end)];
        end
        t=T(end);
        TimeCounter=TimeCounter+1;
        %if TimeCounter<=length(StartTimes)
        %    NextStart=StartTimes(TimeCounter);
        %else
        %    NextStart=te;
        %end
    else %ovulation occured
        t=T(end);
    end

    %check on every stop of the integrator if status of follicles changed
    %helping variables
    ActiveHelp = [];
    %determine actual slope of growth of follicles
    para(1) = 0;
    res = FollicleFunction(T(end),LastYValues,Tovu,Follicles,para,parafoll,Par,dd1,Stim,LutStim,FollStim,DoubStim,firstExtraction);
    %reset vector of active FSH sensitivities
    Follicles.ActiveFSHS = [];

    count10 = 0;
    count14 = 0;
    count18 = 0;
    count20 = 0;
    antralcount = 0;
    indexFollGreater8 = [];

    %loop over all active follicles to set new destiny
    for i = 1:Follicles.NumActive
        %Save y-values of i-th (current) follicle
        yCurFoll = LastYValues(i);
        %slope is negative so the follicle is dereasing in size
        if( res(i) <= 0 )
             Follicles.Follicle{Follicles.Active(i)}.Destiny = -2;
        end

        %follicle is big, but doesn't ovulate yet because there is not enough LH
        if(yCurFoll >= (parafoll(7))) && (Y(end,end-8) < parafoll(10) && ...
           Follicles.Follicle{Follicles.Active(i)}.Destiny == -1)
               Follicles.Follicle{Follicles.Active(i)}.Destiny = 3;
               Follicles.Follicle{Follicles.Active(i)}.TimeDecrease=t;
        end

        if Follicles.Follicle{Follicles.Active(i)}.Destiny == 3 && ...
           (t-Follicles.Follicle{Follicles.Active(i)}.TimeDecrease)>=parafoll(11)
                Follicles.Follicle{Follicles.Active(i)}.Destiny =-2;
        end

       % if Follicles.Follicle{Follicles.Active(i)}.Destiny ~= -2 && ...
       %    (Follicles.Follicle{Follicles.Active(i)}.Time(1) - Follicles.Follicle{Follicles.Active(i)}.Time(end)) > 20
       %         Follicles.Follicle{Follicles.Active(i)}.Destiny =-2;
       % end

        %if LH high enough dominant follicle rest until ovulation shortly after LH peak
        if Y(end,end-8) >= parafoll(10)
            if (yCurFoll >= parafoll(7)) && (Follicles.Follicle{Follicles.Active(i)}.Destiny==-1) ||...
               (yCurFoll >= parafoll(7)) && (Follicles.Follicle{Follicles.Active(i)}.Destiny==3)
                th = t-0.5;
                [val, idx] = min(abs(LH.Time-th));
               if (LH.Y(idx)) >= parafoll(10)
                    Follicles.Follicle{Follicles.Active(i)}.Destiny = 4;
                    Follicles.Follicle{Follicles.Active(i)}.TimeDecrease=t;
               end
            end
        end

        %Follicle ovulates
        if (Follicles.Follicle{Follicles.Active(i)}.Destiny == 4) && ...
            Follicles.Follicle{Follicles.Active(i)}.TimeDecrease + 0.5 <= t
                Follicles.Follicle{Follicles.Active(i)}.Destiny = 1;
                Tovu=T(end);
                OvulationNumber = i;
            if (Stim)
                if Tovu > Par(71) && Par(64) == 0
                    Par(71) = Tovu;
                end
            end
        end

        if(Follicles.Follicle{Follicles.Active(i)}.Destiny ~= 1)
            %put the follicle back to the list of actives and its FSH
            ActiveHelp = [ActiveHelp Follicles.Active(i)];
            %sensitivity back in the FSH vector...
            Follicles.ActiveFSHS = [Follicles.ActiveFSHS Follicles.Follicle{Follicles.Active(i)}.FSHSensitivity];
        end

        if(Stim)
            if Par(64) == 1 && t > dosing_events1(1,1)
                %Save y-values of i-th (current) follicle
                if yCurFoll >= 10
                    count10 = count10 + 1;
                end
                if yCurFoll >= 14
                    count14 = count14 + 1;
                end
                if yCurFoll >= 18
                    count18 = count18 + 1;
                end
                if yCurFoll >= 20 && Follicles.NumActive ~= OvulationNumber
                    count20 = count20 + 1;
                end
                if yCurFoll > 8
                    indexFollGreater8 = [indexFollGreater8 i];
                end
                if yCurFoll >= 2 && yCurFoll <= 8
                    antralcount = antralcount + 1;
                end
            end
        end
    end

    %Update list of active follicles
    Follicles.Active = ActiveHelp;
    %find out how many follicles are active...
    Follicles.NumActive = size(ActiveHelp,2);

    %determine new initial values for all differential equations
    y0old = [];
    for i = 1:(Follicles.NumActive)
        y0old = [y0old Follicles.Follicle{Follicles.Active(i)}.Y(end)];
    end
    y0old = y0old';
    y0 = [y0old;LastYValues(end+1-para(2):end)];

    %integration end reached
    t = T(end);
    if( te - t < 0.001 )
        t = te;
    end

    if (LutStim)
        if Par(64) == 1 && t > Par(72)+1 || ...
           Par(64) == 1 && count18 >= 3 || ...
           Par(64) == 1 && count20 >= 1
            break
        end
    end

    if (FollStim)
        if Par(64) == 1 && t > Par(72)+1 || ...
           Par(64) == 1 && count18 >= 3
            break
        end
    end

    if (DoubStim)
        if ~firstExtraction
            if Par(64) == 1
                if Par(72) < t
                    break
                end
                if count18 > 0
                    result(1,1) = count10;
                    result(2,1) = count14;
                    result(3,1) = count18;
                    result(4,1) = Par(71);
                    result(5,1) = t;
                    %change medicaments
                    Par(71) = ceil(t)+1;
                    Par(72) = Par(71) + 20;
                    numDoses = Par(72)-Par(71)+1;
                    dosing_events1=[[Par(71):Par(72)];[1:numDoses]];
                    %change follicle size and destination
                    %for all follicles >8mm
                    for i=1:size(indexFollGreater8,2)
                        currentIndex = indexFollGreater8(1,i);
                        Follicles.Follicle{Follicles.Active(currentIndex)}.Destiny = -3;
                        Follicles.Follicle{Follicles.Active(currentIndex)}.Y(end,1) = 0;
                    end
                    if antralcount >= 2
                        firstExtraction = 1;
                    else
                        break
                    end
                end
            end
        else
            if Par(64) == 1
                if count20 > 0 || count18 >= 3 || Par(72) < t
                    result(1,2) = count10;
                    result(2,2) = count14;
                    result(3,2) = count18;
                    result(4,2) = Par(71);
                    result(5,2) = t;
                    break
                end
            end
        end
    end

end

%plotting
if(ShowPlots)
    hf=figure(1);
    clf;
    widthofline = 2;
    hold on;
end

%vector to save informations about the ovulating follicle
FollOvulInfo=[];
% save t_start t_end destiny of all follicles
FollInfo = [];

for i = 1:Follicles.Number
    %fill follicle information variable...
    help = [Follicles.Follicle{i}.Time(1); Follicles.Follicle{i}.Time(end); Follicles.Follicle{i}.Destiny; Follicles.Follicle{i}.FSHSensitivity; i];
    FollInfo = [FollInfo help];

    FollInfo2 = [Follicles.Follicle{i}.Time Follicles.Follicle{i}.Y];

    %if (SavePlot)
    %   FileName = sprintf('Follicle%d.csv',i);
    %    fullFileName = fullfile(DirStuff, FileName);
    %    csvwrite(fullFileName,FollInfo2)
    %end

    if Follicles.Follicle{i}.Destiny==1 && Follicles.Follicle{i}.Time(1) > 20
        helpFOT=[i;Follicles.Follicle{i}.Time(1);Follicles.Follicle{i}.Time(end);...
            (Follicles.Follicle{i}.Y(end)-Follicles.Follicle{i}.Y(1))/(Follicles.Follicle{i}.Time(end)-Follicles.Follicle{i}.Time(1)-2)];
        FollOvulInfo=[FollOvulInfo helpFOT];
    end

    if(ShowPlots)
        h = plot(Follicles.Follicle{i}.Time,Follicles.Follicle{i}.Y,'Color',[0 0 0],...
                 'DisplayName','x1','LineWidth', widthofline);
    end
end

%%Cycle length
FollOvulInfo
OvuT = FollOvulInfo(3,:);
Cyclelength = diff(OvuT)
Cyclelengthmean = mean(Cyclelength)
Cyclelengthstd = std(Cyclelength)
NumCycles = length(Cyclelength)

FollperCycle =[];
for i = 1:NumCycles
    t1 = OvuT(i);
    t2 = OvuT(i+1);
    count = 0;
    tp = length(FollInfo(1,:));
    for j = 1:tp
        if FollInfo(1,j) > t1 && FollInfo(1,j) < t2
            count = count +1;
        end
    end
    FollperCycle =[FollperCycle count];
end

FollperCyclemean = mean(FollperCycle)

a = sum(FollperCycle);
rest = n - a;

CycleInfo = [[0 Cyclelength]; [rest FollperCycle]; OvuT];

if(ShowPlots)
   %fsh
    hfsh = plot(FSH.Time,FSH.Y,'Color',[1/2 1 1/2],...
         'DisplayName','x1','LineWidth', widthofline);

   %LH
   hLH = plot(LH.Time,LH.Y,'Color',[1 1/4 1/2],...
         'DisplayName','x1','LineWidth', widthofline);

   %P4
    hp4 = plot(P4.Time,P4.Y,'Color',[1 0 1],...
             'DisplayName','x1','LineWidth', widthofline);

    %threshold when you can measure the follicle size
    hTwo=plot(xlim,[4 4],'Color','r');

    %plot for the follicle size, amount of FSH and amount of P4
    h=[h hfsh hTwo hp4 hLH];
    xlabel('time in d','fontsize',15);
    ylabel('follicle diameter in mm','fontsize',15);
    ylim([0 50])
    %fontsize of plot ticks
    ax = gca;
    set(ax, 'Box', 'off' );
    ax.FontSize = 15;
    set(gca,'linewidth',1.5);
    legend(h,{'follicle growth','FSH','measurable','P4', 'LH'},'fontsize',15,...
        'Location','NorthEast');%,'ovulation');

    figure(2);
    plot(P4.Time,P4.Y, FSH.Time,FSH.Y,'LineWidth',2);
    set(gca,'fontsize',24);
    legend({'P4','FSH'},'fontsize',24,...
        'Location','NorthEastOutside');

    figure(3);
    plot(E2.Time,E2.Y, LH.Time, LH.Y, 'LineWidth',2);
    set(gca,'fontsize',24);
    legend({'E2', 'LH'},'fontsize',24,...
        'Location','NorthEastOutside');

    figure(4);
    plot(GnRH.Time,GnRH.Y, 'LineWidth',2);
    set(gca,'fontsize',24);
    legend({'GnRH'},'fontsize',24,...
        'Location','NorthEastOutside');

    %indexes of solutions are number from Model28_ODE + 1
    solutions = [solutions.Time solutions.Y];

    file = '/Users/sophie/Documents/GynCycleModel_Pub2021/NonVec_Model/pfizer_normal.txt';
    delimiterIn='\t';
    headerlinesIn=0;
    Data=importdata(file,delimiterIn,headerlinesIn);
    ID = unique(Data(:,end));

    figure(5)
    hold on
    for i = 1:length(ID)
    Data_LH = [];
        for j = 1:length(Data(:,end))
            if Data(j,end) == ID(i)
                Data_LH = [Data_LH; Data(j,:)];
            end
            Data_LH;
        end
    scatter(Data_LH(:,1),Data_LH(:,2), 'x')
    hold on
    end

    for i = 1:length(FollOvulInfo(end,:))
            Tovu = FollOvulInfo(3,i);
            t1 = Tovu -14;
            t2 = Tovu +14;
            [val,idx1]=min(abs(LH.Time-t1));
            [val,idx2]=min(abs(LH.Time-t2));
            Timenew = LH.Time(idx1:idx2)-t1;
            plot(Timenew,LH.Y(idx1:idx2), 'k--')
            hold on
    end

    figure(6)
    hold on
    for i = 1:length(ID)
    H = [];
        for j = 1:length(Data(:,end))
            if Data(j,end) == ID(i)
                H = [H; Data(j,:)];
            end
            H;
        end
    scatter(H(:,1),H(:,3), 'x')
    hold on
    end

    for i = 1:length(FollOvulInfo(end,:))
            Tovu = FollOvulInfo(3,i);
            t1 = Tovu -14;
            t2 = Tovu +14;
            [val,idx1]=min(abs(FSH.Time-t1));
            [val,idx2]=min(abs(FSH.Time-t2));
            Timenew = FSH.Time(idx1:idx2)-t1;
            plot(Timenew,FSH.Y(idx1:idx2), 'k--')
            hold on
    end

    figure(7)
    hold on
    for i = 1:length(ID)
    H = [];
        for j = 1:length(Data(:,end))
            if Data(j,end) == ID(i)
                H = [H; Data(j,:)];
            end
            H;
        end
    scatter(H(:,1),H(:,4), 'x')
    hold on
    end

    for i = 1:length(FollOvulInfo(end,:))
            Tovu = FollOvulInfo(3,i);
            t1 = Tovu -14;
            t2 = Tovu +14;
            [val,idx1]=min(abs(E2.Time-t1));
            [val,idx2]=min(abs(E2.Time-t2));
            Timenew = E2.Time(idx1:idx2)-t1;
            plot(Timenew,E2.Y(idx1:idx2), 'k--')
            hold on
    end

    figure(8)
    hold on
    for i = 1:length(ID)
    H = [];
        for j = 1:length(Data(:,end))
            if Data(j,end) == ID(i)
                H = [H; Data(j,:)];
            end
            H;
        end
    scatter(H(:,1),H(:,5), 'x')
    hold on
    end

    for i = 1:length(FollOvulInfo(end,:))
            Tovu = FollOvulInfo(3,i);
            t1 = Tovu -14;
            t2 = Tovu +14;
            [val,idx1]=min(abs(P4.Time-t1));
            [val,idx2]=min(abs(P4.Time-t2));
            Timenew = P4.Time(idx1:idx2)-t1;
            plot(Timenew,P4.Y(idx1:idx2), 'k--')
            hold on
    end

    freq = [];
    for i = 1:length(E2.Y)
         yGfreq = Par(1) / ( 1 + ( P4.Y(i) / Par(2) ) ^ Par(3) ) ...
              * ( 1 + E2.Y(i) ^ Par(4) ...
              / ( Par(5) ^ Par(4) + E2.Y(i) ^ Par(4) ) );

          freq = [freq yGfreq];
    end

    figure(9)
    plot(E2.Time, freq)

    figure(10)
    plot(FSHRez.Time, FSHRez.Y)

end

if Par(64) == 1
    [val, idx] = min(abs(E2.Time-(Par(71)-1)));
    E2dm1 = E2.Y(idx);
    [val, idx] = min(abs(E2.Time-(Par(71)+1)));
    E2d1 = E2.Y(idx);
    [val, idx] = min(abs(E2.Time-(Par(71)+5)));
    E2d6 = E2.Y(idx);
    E2dend= E2.Y(end);

    [val, idx] = min(abs(P4.Time-(Par(71)-1)));
    P4dm1 = P4.Y(idx);
    [val, idx] = min(abs(P4.Time-(Par(71)+1)));
    P4d1 = P4.Y(idx);
    [val, idx] = min(abs(P4.Time-(Par(71)+5)));
    P4d6 = P4.Y(idx);
    P4dend= P4.Y(end);

    [val, idx] = min(abs(LH.Time-(Par(71)-1)));
    LHdm1 = LH.Y(idx);
    [val, idx] = min(abs(LH.Time-(Par(71)+1)));
    LHd1 = LH.Y(idx);
    [val, idx] = min(abs(LH.Time-(Par(71)+5)));
    LHd6 = LH.Y(idx);
    LHdend= LH.Y(end);

    sumFSH = FSH.Y + FSHmed.Y;

    [val, idx] = min(abs(FSH.Time-(Par(71)-1)));
    FSHdm1 = sumFSH(idx);
    [val, idx] = min(abs(FSH.Time-(Par(71)+1)));
    FSHd1 = sumFSH(idx);
    [val, idx] = min(abs(FSH.Time-(Par(71)+5)));
    FSHd6 = sumFSH(idx);
    FSHdend= sumFSH(end);

    MedInfo = [count10 count14 E2dm1 E2d1 E2d6 E2dend P4dm1 P4d1 P4d6 P4dend ...
                LHdm1 LHd1 LHd6 LHdend FSHdm1 FSHd1 FSHd6 FSHdend ...
                dosing_events1(1,1) t t-dosing_events1(1,1)]


    count10 - count14
    count14
    count18
    count20
    dosing_events1
    t
end

% FollStarts = FollInfo(1,:)';
% dlmwrite("StartTimesPoiss.txt",FollStarts);
%
% FollSens = FollInfo(4,:)';
% dlmwrite("FSH.txt", FollSens);

if Foll_ModelPop || Horm_ModelPop
    totalcheck = 0;
    if ~isempty(FollOvulInfo)
        for i = 2:length(FollOvulInfo(end,:))
            Tovu = FollOvulInfo(3,i);
            t1 = Tovu - 5;
            t2 = Tovu - 2;
            t3 = Tovu + 0.05;
            t4 = Tovu + 7;
            [val,idx1]=min(abs(FSH.Time-t1));
            [val,idx2]=min(abs(FSH.Time-t2));
            [val,idx3]=min(abs(FSH.Time-t3));
            [val,idx4]=min(abs(FSH.Time-t4));

            check = 0;

            if FSH.Y(idx2) - FSH.Y(idx1) < 0
                check = check + 1;
            end

            if FSH.Y(idx3) - FSH.Y(idx2) > 0
                check = check + 1;
            end

            if FSH.Y(idx4) - FSH.Y(idx3) < 0
                check = check + 1;
            end

            if check == 3
                totalcheck = totalcheck + 1;
            end
        end

        check = 0;

	%better?: 25 - 35
        if Cyclelengthmean > 21 && Cyclelengthmean < 40
            check = check + 1;
        end

        if Cyclelengthstd < 4
            check = check + 1;
        end

        if totalcheck >= (length(FollOvulInfo(end,:)) - 1) * 0.75 && check == 2
            Par(77) = 1;
        end

        if ~isempty(FollOvulInfo)
            H = [Par'; paraPoi; parafoll];
            ModelPop_Params = [ModelPop_Params H];

            F = [Cyclelengthmean; Cyclelengthstd; FollperCyclemean; Par(77)];
            ModelPop_CycleInfo = [ModelPop_CycleInfo F];
        end
    end
end

if (SavePlotStuff)
    FileName = sprintf('%s_%d.csv','E2',runind);
    fullFileName = fullfile(DirStuff, FileName);
    csvwrite(fullFileName,E2.Y)

    FileName = sprintf('%s_%d.csv','FSH',runind);
    fullFileName = fullfile(DirStuff, FileName);
    csvwrite(fullFileName,FSH.Y)

    FileName = sprintf('%s_%d.csv','LH',runind);
    fullFileName = fullfile(DirStuff, FileName);
    csvwrite(fullFileName,LH.Y)

    FileName = sprintf('%s_%d.csv','P4',runind);
    fullFileName = fullfile(DirStuff, FileName);
    csvwrite(fullFileName,P4.Y)

    FileName = sprintf('%s_%d.csv','Time',runind);
    fullFileName = fullfile(DirStuff, FileName);
    csvwrite(fullFileName,E2.Time)

    FileName = sprintf('%s_%d.csv','OvulationInfo',runind');
    fullFileName = fullfile(DirStuff, FileName);
    csvwrite(fullFileName,FollOvulInfo)

    FileName = sprintf('Solutions.csv');
    fullFileName = fullfile(DirStuff, FileName);
    csvwrite(fullFileName,solutions.Y)
end

if (SaveSim)
    FileName = sprintf('%s_%d.csv','DomFolGrowth',runind)
    fullFileName = fullfile(DirStuff, FileName);
    csvwrite(fullFileName,FollOvulInfo(4,:))

    FileName = sprintf('%s_%d.csv','Cyclelength',runind)
    fullFileName = fullfile(DirStuff, FileName);
    csvwrite(fullFileName,CycleInfo(1,:))

    FileName = sprintf('%s_%d.csv','CycleFollCount',runind)
    fullFileName = fullfile(DirStuff, FileName);
    csvwrite(fullFileName,CycleInfo(2,:))

    if Par(64) == 1 && count18 >= 3 || ...
       Par(64) == 1 && count20 >= 1
        FileName = sprintf('MedInfo.csv');
        fullFileName = fullfile(DirStuff, FileName);
        M = load(fullFileName);
        M = [M; MedInfo];
        csvwrite(fullFileName,M);
    end
end

end
