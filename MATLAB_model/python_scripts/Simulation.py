"""
% Calculation and solving of the ODE
% needs: 
%%Parameter(programming, Poisson distribution, ODE calculations),
%%integration starttime, endtime, initial values,
%%poisson distributed starttimes of the follicles, normal distributed FSH
%%sensitivities of the follicles, ShowStuff, SaveStuff, DirStuff
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

from FollicleClass import FollicleClass
from FollicleFunction import FollicleFunction
from EvaluateFollicle import EvaluateFollicle
from Poissonproc import poissonproc


def Simulation(para, paraPoi, parafoll, Par, tb, te,\
               StartValues, StartTimes, FSHVec, ShowPlots,\
               SaveSim, SavePlotStuff, DirStuff, Stim,\
               LutStim, FollStim, DoubStim, Foll_ModelPop,\
               Horm_ModelPop, runind):
    
    # Integration period
    tspan = [tb, te]
    # Variable for the current time
    t = tb
    # Timepoint of last ovulation, initiated as 14 will be changed in the course of the simulation 
    Tovu = 14

    # Initial values
    y0Foll = StartValues[0]    # start size of the follicle
    y0E = StartValues[1]       # start value estradiol
    y0P4 = StartValues[2]      # start value progesterone
    y0LH = StartValues[9]      # start value LH
    y0FSH = StartValues[7]     # start value FSH
    y0 = np.array(StartValues)

    # Values for tracking the follicles 
    FollCounter = 1

    # Class to save follicles and their properties                                          
    Follicles = FollicleClass(y0Foll, FSHVec[FollCounter - 1], t)  
    FollCounter += 1

    # Values needed for the first integrations
    TimeCounter = 0
    NextStart = StartTimes[TimeCounter]
    TimeFol = np.array([t])

    # Arrays to save times when new follicles emerge or when they can't emerge
    NewFollicle = []
    NoNewFollicle = []
    LastYValues = []
    result = np.zeros((5, 2))

    # Variables for drug Administration
    dd1 = 0
    dosing_events1 = []
    firstExtraction = 0

    # Global variables
    global ModelPop_Params
    global ModelPop_CycleInfo

    # Tracking the concentrations of important hormone species
    E2 = {'Time': np.array([t]), 'Y': np.array([y0[-16]])}
    P4 = {'Time': np.array([t]), 'Y': np.array([y0[-15]])}
    FSH = {'Time': np.array([t]), 'Y': np.array([y0[-10]])}
    FSHRez = {'Time': np.array([t]), 'Y': np.array([y0[-14]])}
    LH = {'Time': np.array([t]), 'Y': np.array([y0[-8]])}
    GnRH = {'Time': np.array([t]), 'Y': np.array([y0[-2]])}
    GnRHRezA = {'Time': np.array([t]), 'Y': np.array([y0[-5]])}
    FSHmed = {'Time': np.array([t]), 'Y': np.array([y0[-1]])}
    solutions = {'Time': np.array([t]), 'Y': np.array([y0[1:]])}

    # Main loop
    while t < te:
        print("t:", round(t, 3))

        # Follicles recruitment depending on FSH concentration in the system
        fshAll = y0[-10] + y0[-1]
        fshimp = fshAll**Par[31] / (fshAll**Par[31] + Par[32]**Par[31])
        #print(paraPoi[0], paraPoi[0], fshimp)
        #print([t, te])
        timevec = poissonproc(paraPoi[0] + 6 * paraPoi[0] * fshimp, [t, te])

        # Set integration period for the current follicle
        if timevec.size != 0:
            NextStart = timevec[0]
            tspan = [t, NextStart] 
        else:
            tspan = [t, te]

        # Determine number of follicles
        NumFollicles = len(y0) - para[1] 

        # Set mass matrix for DAE system
        n = len(y0)
        M = np.eye(n)
        M[NumFollicles, NumFollicles] = 0     # alg. eq. for E2
        M[NumFollicles + 1, NumFollicles + 1] = 0     # alg. eq. for P4
        M[NumFollicles + 15, NumFollicles + 15] = 0     # alg. eq. for LH med
        M[NumFollicles + 16, NumFollicles + 16] = 0     # alg. eq. for FSH med

        # Event function stops the integration when an ovulation takes place within the interval tspan
        #print("para", para)
        #print("t", t, "y", y)
        #options = {'mass': M,\
                   #'events': lambda t, y: EvaluateFollicle(t, y, para, parafoll, LH)}
        #print("para after", para)
        options = {'mass': M}

        # Search for dosing events in [tspan[0], tspan[1]]:
        dosing_timeIdx = []
        if dosing_events1:
            dosing_timeIdx = np.intersect1d(\
                np.where(dosing_events1[0] > tspan[0]),\
                np.where(dosing_events1[0] <= tspan[1]))

        # solve differential equations
        para[0] = 0
        Y = np.array([])
        T = np.array([])
        if dosing_timeIdx:
            tstart = tspan[0]
            tend = tspan[1]
            yInitial = y0
            #print("YInitial, supposedly y", yInitial)
            for i in range(len(dosing_timeIdx)):
                tspan = [tstart, dosing_events1[0, dosing_timeIdx[i]]]
                if tspan[0] != tspan[1]:
                    sol = solve_ivp(lambda t, y: FollicleFunction(\
                                        t, y, Tovu, Follicles, para,\
                                        parafoll, Par, dd1, Stim,\
                                        LutStim, FollStim, DoubStim,\
                                        firstExtraction),\
                                    tspan, yInitial, method='LSODA',\
                                    **options)
                    ti = sol.t
                    yi = sol.y.T
                    T = np.concatenate((T, ti[1:]))
                    Y = np.concatenate((Y, yi[1:]))
                    tstart = T[-1]
                    yInitial = Y[-1]
                dd1 = dosing_events1[1, dosing_timeIdx[i]]
            tspan = [T[-1], tend]
            sol = solve_ivp(lambda t, y: FollicleFunction(
                                t, y, Tovu, Follicles, para,\
                                parafoll, Par, dd1, Stim,\
                                LutStim, FollStim, DoubStim,\
                                firstExtraction),\
                            tspan, yInitial, method='LSODA', **options)
            ti = sol.t
            yi = sol.y.T
            T = np.concatenate((T, ti[1:]))
            Y = np.concatenate((Y, yi[1:]))
        else:
            #print("para 160", para)
            print("y0 supposedly y", y0)
            sol = solve_ivp(lambda t, y: FollicleFunction(\
                                t, y, Tovu, Follicles, para,\
                                parafoll, Par, dd1, Stim,\
                                LutStim, FollStim, DoubStim,\
                                firstExtraction),\
                            tspan, y0, method='LSODA', **options)
            T = sol.t
            Y = sol.y.T

        for i in range(Follicles.NumActive):
            # saves all times of the foll that was active during last run
            Follicles.Follicle[Follicles.Active[i]-1]['Time'] = \
                    np.concatenate((Follicles.Follicle[Follicles.Active[i]-1]['Time'], T[1:]))
            # saves all sizes of the foll that was active during last run
            Follicles.Follicle[Follicles.Active[i]-1]['Y'] = \
                    np.concatenate((Follicles.Follicle[Follicles.Active[i]-1]['Y'], Y[1:, i]))

        if LutStim:
            # Werte für die Medikamentengabe setzen
            if Par[70] == Tovu and Par[63] == 0:
                for i in range(Follicles.NumActive):
                    if Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] == -1:
                        # matrix of dosing times and drugs added: row1: times, row2: drug, row 3, dose
                        if t - Par[70] >= 1 and t - Par[70] < 4:
                            Par[70] = round(t)
                            Par[71] = Par[70] + 15
                            # Menopur
                            numDoses = Par[71] - Par[70] + 1
                            dosing_events1 = np.array(\
                                [[*range(Par[70], Par[71] + 1)],\
                                 [*range(1, numDoses + 1)]])
                            Par[63] = 1

        if FollStim:
            if Par[70] == Tovu and Par[63] == 0:
                if t > Par[70] + 14:
                    for i in range(Follicles.NumActive):
                        if Follicles.Follicle[Follicles.Active[i]-1]['Y'][-1] >= 14:
                            # matrix of dosing times and drugs added: row1: times, row2: drug, row 3, dose
                            Par[70] = round(t) + 1
                            Par[71] = Par[70] + 14
                            numDoses = Par[71] - Par[70] + 1
                            dosing_events1 = np.array(\
                                [[*range(Par[70], Par[71] + 1)],\
                                 [*range(1, numDoses + 1)]])
                            Par[63] = 1

        if DoubStim:
            if Par[70] == Tovu and Par[63] == 0:
                Par[70] = round(t) + 20
                Par[71] = Par[70] + 15
                numDoses = Par[71] - Par[70] + 1
                dosing_events1 = np.array(\
                    [[*range(Par[70], Par[71] + 1)],\
                     [*range(1, numDoses + 1)]])
                Par[63] = 1

        # saves the measuring times of the active foll.
        TimeFol = np.concatenate((TimeFol, T[1:]))
        # saves last Y values of the follicles
        LastYValues = Y[-1, :]
        print("Y", Y)
        print("?? not sure is assignet something:", LastYValues)

        # save values for E2 
        E2['Time'] = np.concatenate((E2['Time'], T[1:]))
        E2['Y'] = np.concatenate((E2['Y'], Y[1:, -16]))
        # save values for P4
        P4['Time'] = np.concatenate((P4['Time'], T[1:]))
        P4['Y'] = np.concatenate((P4['Y'], Y[1:, -15]))
        # save values for LH
        LH['Time'] = np.concatenate((LH['Time'], T[1:]))
        LH['Y'] = np.concatenate((LH['Y'], Y[1:, -8]))
        # save values for FSH 
        FSH['Time'] = np.concatenate((FSH['Time'], T[1:]))
        FSH['Y'] = np.concatenate((FSH['Y'], Y[1:, -10]))
        # save values for FSH Rezeptor
        FSHRez['Time'] = np.concatenate((FSHRez['Time'], T[1:]))
        FSHRez['Y'] = np.concatenate((FSHRez['Y'], Y[1:, -14]))
        # save values for GnRH
        GnRH['Time'] = np.concatenate((GnRH['Time'], T[1:]))
        GnRH['Y'] = np.concatenate((GnRH['Y'], Y[1:, -3]))

        # GnRH Concentration
        GnRHRezA['Time'] = np.concatenate((GnRHRezA['Time'], T[1:]))
        GnRHRezA['Y'] = np.concatenate((GnRHRezA['Y'], Y[1:, -6]))

        # save values for FSH
        FSHmed['Time'] = np.concatenate((FSHmed['Time'], T[1:]))
        FSHmed['Y'] = np.concatenate((FSHmed['Y'], Y[1:, -1]))

        # save solutions
        solutions['Time'] = np.concatenate((solutions['Time'], T[1:]))
        solutions['Y'] = np.concatenate((solutions['Y'], Y[1:, NumFollicles:]))

        # no ovulation (=no event) occurred
        if T[-1] == tspan[1]:
            # initialize new follicle
            # Set initial values for new follicle
            Follicle1 = dict()
            Follicle1['Y'] = np.array([y0Foll])
            if FSHVec.size != 0:
                Follicle1['FSHSensitivity'] = FSHVec[FollCounter]
                FollCounter += 1
            FSHSSave = Follicles.ActiveFSHS
            Follicles.ActiveFSHS = np.concatenate(\
                (Follicles.ActiveFSHS, [Follicle1['FSHSensitivity']]))

            # Test if Follicle(s) could survive
            # (slope of growth-function positive or negative)
            testyvalues = LastYValues[:len(LastYValues) - para[1]]
            testyvalues = np.concatenate(\
                (testyvalues, Follicle1['Y'], LastYValues[-para[1]:]))
            para[0] = 1
            testyslope = FollicleFunction(T[-1], testyvalues, Tovu,\
                                          Follicles, para, parafoll,\
                                          Par, dd1, Stim, LutStim,\
                                          FollStim, DoubStim, firstExtraction)

            # if follicle got chance to survive -> initiate new follicle and update follicles-vector
            if testyslope[-para[1]] > 0:
                Follicle1['Time'] = np.array([T[-1]])
                Follicle1['TimeDecrease'] = 0
                Follicle1['Destiny'] = -1
                Follicles.Number += 1
                Follicle1['Number'] = Follicles.Number
                Follicles.NumActive += 1
                Follicles.Active = np.concatenate(\
                    (Follicles.Active, [Follicle1['Number']]))
                Follicles.Follicle = np.append(Follicles.Follicle, Follicle1)
                NewFollicle.append(T[-1])
                LastYValues = testyvalues
            else:
                # no chance to survive save in NoNewFollicle for statistic
                Follicles.ActiveFSHS = FSHSSave
                NoNewFollicle.append(T[-1])
            t = T[-1]
            TimeCounter += 1
        else:  # ovulation occurred
            t = T[-1]

        # check on every stop of the integrator if status of follicles changed
        # helping variables
        ActiveHelp = []
        # determine actual slope of growth of follicles
        para[0] = 0
        res = FollicleFunction(T[-1], LastYValues, Tovu, Follicles,\
                               para, parafoll, Par, dd1, Stim,\
                               LutStim, FollStim, DoubStim, firstExtraction)
        # reset vector of active FSH sensitivities
        Follicles.ActiveFSHS = []

        count10 = 0
        count14 = 0
        count18 = 0
        count20 = 0
        antralcount = 0
        indexFollGreater8 = []

        # loop over all active follicles to set new destiny   
        for i in range(Follicles.NumActive):
            # Save y-values of i-th (current) follicle
            yCurFoll = LastYValues[i]
            # slope is negative so the follicle is decreasing in size
            if res[i] <= 0: 
                Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] = -2
            
            # follicle is big, but doesn't ovulate yet because there is not enough LH
            if (yCurFoll >= parafoll[6]) and (Y[-1, -8] < parafoll[9] and 
               Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] == -1):
                Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] = 3
                Follicles.Follicle[Follicles.Active[i]-1]['TimeDecrease'] = t

            if (Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] == 3 and 
               (t - Follicles.Follicle[Follicles.Active[i]-1]['TimeDecrease']) >= parafoll[10]):
                Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] = -2
            
            # if Follicles.Follicle[Follicles.Active[i]].Destiny != -2 and ...
            #    (Follicles.Follicle[Follicles.Active[i]].Time[0] - Follicles.Follicle[Follicles.Active[i]].Time[-1]) > 20
            #         Follicles.Follicle[Follicles.Active[i]].Destiny = -2
            
            # if LH high enough dominant follicle rest until ovulation shortly after LH peak 
            if Y[-1, -8] >= parafoll[9]:
                if (yCurFoll >= parafoll[6] and Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] == -1) or \
                   (yCurFoll >= parafoll[6] and Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] == 3):
                    th = t - 0.5
                    idx = np.argmin(np.abs(LH['Time'] - th))
                    if LH.Y[idx] >= parafoll[9]: 
                        Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] = 4
                        Follicles.Follicle[Follicles.Active[i]-1]['TimeDecrease'] = t
            
            # Follicle ovulates
            if (Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] == 4 and 
                Follicles.Follicle[Follicles.Active[i]-1]['TimeDecrease'] + 0.5 <= t):
                Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] = 1
                Tovu = T[-1]
                OvulationNumber = i
                if Stim:
                    if Tovu > Par[70] and Par[63] == 0: 
                        Par[70] = Tovu
            
            if Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] != 1:
                # put the follicle back to the list of actives and its FSH
                ActiveHelp.append(Follicles.Active[i]-1)
                # sensitivity back in the FSH vector...
                Follicles.ActiveFSHS.append(Follicles.Follicle[Follicles.Active[i]-1]['FSHSensitivity'])
            
            if Stim:
                if Par[63] == 1 and t > dosing_events1[0, 0]:
                    # Save y-values of i-th (current) follicle
                    if yCurFoll >= 10: 
                        count10 += 1 
                    if yCurFoll >= 14: 
                        count14 += 1 
                    if yCurFoll >= 18: 
                        count18 += 1 
                    if yCurFoll >= 20 and Follicles.NumActive != OvulationNumber:
                        count20 += 1 
                    if yCurFoll > 8:
                        indexFollGreater8.append(i)
                    if 2 <= yCurFoll <= 8:
                        antralcount += 1

        # Update list of active follicles
        Follicles.Active = ActiveHelp
        # find out how many follicles are active...
        Follicles.NumActive = len(ActiveHelp)
        # determine new initial values for all differential equations
        y0old = []
        for i in range(Follicles.NumActive):
            #print("f:", Follicles.Follicle[Follicles.Active[i]]['Y'])
            y0old.append(Follicles.Follicle[Follicles.Active[i]-1]['Y'][-1])
        y0old = np.array(y0old)#.reshape(-1, 1)
        print("Last Y value of follicles:", y0old)
        print("? not sure:", LastYValues[-para[1]:])
        y0 = np.concatenate((y0old, LastYValues[-para[1]:]))
        #y0 = np.vstack((y0old, LastYValues[-para[1]:]))

        # integration end reached
        t = T[-1]
        if te - t < 0.001:
            t = te

        if LutStim:
            if (Par[63] == 1 and t > Par[71] + 1) or \
               (Par[63] == 1 and count18 >= 3) or \
               (Par[63] == 1 and count20 >= 1):
                break

        if FollStim:
            if (Par[63] == 1 and t > Par[71] + 1) or \
               (Par[63] == 1 and count18 >= 3):
                break

        if DoubStim:
            if not firstExtraction:
                if Par[63] == 1:
                    if Par[71] < t:
                        break
                    if count18 > 0:
                        result[0, 0] = count10
                        result[1, 0] = count14
                        result[2, 0] = count18
                        result[3, 0] = Par[70]
                        result[4, 0] = t
                        # change medicaments
                        Par[70] = int(np.ceil(t)) + 1
                        Par[71] = Par[70] + 20
                        numDoses = Par[71] - Par[70] + 1
                        dosing_events1 = np.array(\
                            [[Par[70] + i for i in range(numDoses)],\
                             range(1, numDoses + 1)])
                        # change follicle size and destination for all follicles >8mm
                        for i in range(indexFollGreater8.shape[1]):
                            currentIndex = indexFollGreater8[0, i]
                            Follicles.Follicle[\
                                Follicles.Active[currentIndex]-1]['Destiny'] = -3
                            Follicles.Follicle[\
                                Follicles.Active[currentIndex]-1]['Y'][-1, 0] = 0
                        if antralcount >= 2:
                            firstExtraction = 1
                        else:
                            break
                else:
                    if Par[63] == 1:
                        if count20 > 0 or count18 >= 3 or Par[71] < t:
                            result[0, 1] = count10
                            result[1, 1] = count14
                            result[2, 1] = count18
                            result[3, 1] = Par[70]
                            result[4, 1] = t
                            break

    # plotting
    if ShowPlots:
        hf = plt.figure(1)
        plt.clf()
        widthofline = 2
        #plt.hold(True)

    # vector to save informations about the ovulating follicle
    FollOvulInfo = []
    # save t_start t_end destiny of all follicles
    FollInfo = []

    for i in range(Follicles.Number):
        # fill follicle information variable...
        help_info = [Follicles.Follicle[i]['Time'][0],\
                     Follicles.Follicle[i]['Time'][-1],
                     Follicles.Follicle[i]['Destiny'],\
                     Follicles.Follicle[i]['FSHSensitivity'], i]
        FollInfo.append(help_info)

        FollInfo2 = np.column_stack((Follicles.Follicle[i]['Time'],\
                                     Follicles.Follicle[i]['Y']))

        # if (SavePlot)
        #    FileName = f'Follicle{i}.csv'
        #    fullFileName = os.path.join(DirStuff, FileName)
        #    np.savetxt(fullFileName, FollInfo2, delimiter=',')

        if Follicles.Follicle[i]['Destiny'] == 1 and Follicles.Follicle[i]['Time'][0] > 20:
            helpFOT = [i, Follicles.Follicle[i]['Time'][0],\
                       Follicles.Follicle[i]['Time'][-1],
                       (Follicles.Follicle[i]['Y'][-1] - Follicles.Follicle[i]['Y'][0]) /
                       (Follicles.Follicle[i]['Time'][-1] - Follicles.Follicle[i]['Time'][0] - 2)]
            FollOvulInfo.append(helpFOT)

        if ShowPlots:
            plt.plot(Follicles.Follicle[i]['Time'], Follicles.Follicle[i]['Y'], color=[0, 0, 0],
                     label='x1', linewidth=widthofline)
    # Cycle length
    FollOvulInfo = ...  # Define or load your FollOvulInfo data
    OvuT = FollOvulInfo[2, :]
    Cyclelength = np.diff(OvuT)
    Cyclelengthmean = np.mean(Cyclelength)
    Cyclelengthstd = np.std(Cyclelength)
    NumCycles = len(Cyclelength)

    FollperCycle = []
    for i in range(NumCycles):
        t1 = OvuT[i]
        t2 = OvuT[i + 1]
        count = 0
        tp = len(FollInfo[0, :])  # Define or load your FollInfo data
        for j in range(tp):
            if FollInfo[0, j] > t1 and FollInfo[0, j] < t2:
                count += 1
        FollperCycle.append(count)

    FollperCyclemean = np.mean(FollperCycle)

    a = sum(FollperCycle)
    rest = n - a  # Define n appropriately

    CycleInfo = np.array([[0] + list(Cyclelength), [rest] + FollperCycle, OvuT])

    if ShowPlots:  # Define ShowPlots appropriately
        # FSH
        hfsh, = plt.plot(FSH['Time'], FSH['Y'], color=[1/2, 1, 1/2],\
                         linewidth=widthofline, label='x1')

        # LH  
        hLH, = plt.plot(LH['Time'], LH['Y'], color=[1, 1/4, 1/2],\
                        linewidth=widthofline, label='x1')

        # P4
        hp4, = plt.plot(P4['Time'], P4['Y'], color=[1, 0, 1],\
                        linewidth=widthofline, label='x1')

        # Threshold when you can measure the follicle size
        hTwo = plt.plot(plt.xlim(), [4, 4], color='r')

        # Plot for the follicle size, amount of FSH and amount of P4 
        h = [hfsh, hTwo, hp4, hLH]
        plt.xlabel('time in d', fontsize=15)
        plt.ylabel('follicle diameter in mm', fontsize=15)
        plt.ylim([0, 50])
        ax = plt.gca()
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=15)
        plt.legend(h, ['follicle growth', 'FSH', 'measurable', 'P4', 'LH'],\
                   fontsize=15, loc='northeast')

        plt.figure(2)
        plt.plot(P4['Time'], P4['Y'], LH['Time'], LH['Y'], linewidth=2)
        plt.gca().tick_params(labelsize=24)
        plt.legend(['P4', 'FSH'], fontsize=24, loc='northeastoutside')

        plt.figure(3)
        plt.plot(E2['Time'], E2['Y'], LH['Time'], LH['Y'], linewidth=2)
        plt.gca().tick_params(labelsize=24)
        plt.legend(['E2', 'LH'], fontsize=24, loc='northeastoutside')

        plt.figure(4)
        plt.plot(GnRH['Time'], GnRH['Y'], linewidth=2)
        plt.gca().tick_params(labelsize=24)
        plt.legend(['GnRH'], fontsize=24, loc='northeastoutside')

        # Indexes of solutions are number from Model28_ODE + 1
        solutions = np.column_stack((solutions.Time, solutions.Y))

        file = '/Users/sophie/Documents/GynCycleModel_Pub2021/NonVec_Model/pfizer_normal.txt'
        Data = np.loadtxt(file, delimiter='\t', skiprows=0)
        ID = np.unique(Data[:, -1])

        plt.figure(5)
        #plt.hold(True)
        for i in range(len(ID)):
            Data_LH = Data[Data[:, -1] == ID[i]]
            plt.scatter(Data_LH[:, 0], Data_LH[:, 1], marker='x')
            #plt.hold(True)

        for i in range(len(FollOvulInfo[-1, :])):
            Tovu = FollOvulInfo[2, i]
            t1 = Tovu - 14
            t2 = Tovu + 14
            idx1 = np.argmin(np.abs(LH['Time'] - t1))
            idx2 = np.argmin(np.abs(LH['Time'] - t2))
            Timenew = LH['Time'][idx1:idx2] - t1
            plt.plot(Timenew, LH['Y'][idx1:idx2], 'k--')
            #plt.hold(True)

        plt.figure(6)
        #plt.hold(True)
        for i in range(len(ID)):
            H = Data[Data[:, -1] == ID[i]]
            plt.scatter(H[:, 0], H[:, 2], marker='x')
            #plt.hold(True)

        for i in range(len(FollOvulInfo[-1, :])):
            Tovu = FollOvulInfo[2, i]
            t1 = Tovu - 14
            t2 = Tovu + 14
            idx1 = np.argmin(np.abs(FSH['Time'] - t1))
            idx2 = np.argmin(np.abs(FSH['Time'] - t2))
            Timenew = FSH['Time'][idx1:idx2] - t1
            plt.plot(Timenew, FSH['Y'][idx1:idx2], 'k--')
            #plt.hold(True)

        plt.figure(7)
        #plt.hold(True)
        for i in range(len(ID)):
            H = Data[Data[:, -1] == ID[i]]
            plt.scatter(H[:, 0], H[:, 3], marker='x')
            #plt.hold(True)
        for i in range(len(FollOvulInfo[-1])):
            Tovu = FollOvulInfo[2, i]
            t1 = Tovu - 14
            t2 = Tovu + 14
            idx1 = np.argmin(np.abs(E2.Time - t1))
            idx2 = np.argmin(np.abs(E2.Time - t2))
            Timenew = E2['Time'][idx1:idx2 + 1] - t1
            plt.plot(Timenew, E2['Y'][idx1:idx2 + 1], 'k--')
            #plt.hold(True)

        plt.figure(8)
        #plt.hold(True)
        for i in range(len(ID)):
            H = []
            for j in range(len(Data[:, -1])):
                if Data[j, -1] == ID[i]:
                    H.append(Data[j, :])
            H = np.array(H)
            plt.scatter(H[:, 0], H[:, 4], marker='x')
            #plt.hold(True)

        for i in range(len(FollOvulInfo[-1])):
            Tovu = FollOvulInfo[2, i]
            t1 = Tovu - 14
            t2 = Tovu + 14
            idx1 = np.argmin(np.abs(P4.Time - t1))
            idx2 = np.argmin(np.abs(P4.Time - t2))
            Timenew = P4['Time'][idx1:idx2 + 1] - t1
            plt.plot(Timenew, P4['Y'][idx1:idx2 + 1], 'k--')
            #plt.hold(True)

        freq = []
        for i in range(len(E2.Y)):
            yGfreq = Par[0] / (1 + (P4['Y'][i] / Par[1]) ** Par[2]) *\
                     (1 + E2['Y'][i] ** Par[3] / (Par[4] ** Par[3] +\
                                               E2['Y'][i] ** Par[3]))
            freq.append(yGfreq)

        plt.figure(9)
        plt.plot(E2['Time'], freq)

        plt.figure(10)
        plt.plot(FSHRez['Time'], FSHRez.Y)
        plt.show()

    if Par[64] == 1:
        val, idx = min((abs(E2['Time'] - (Par[71] - 1)), i)\
                       for i in range(len(E2['Time'])))
        E2dm1 = E2['Y'][idx]
        val, idx = min((abs(E2['Time'] - (Par[71] + 1)), i)\
                       for i in range(len(E2['Time'])))
        E2d1 = E2['Y'][idx]
        val, idx = min((abs(E2['Time'] - (Par[71] + 5)), i)\
                       for i in range(len(E2['Time'])))
        E2d6 = E2['Y'][idx]
        E2dend = E2['Y'][-1]

        val, idx = min((abs(P4['Time'] - (Par[71] - 1)), i)\
                       for i in range(len(P4['Time'])))
        P4dm1 = P4['Y'][idx]
        val, idx = min((abs(P4['Time'] - (Par[71] + 1)), i)\
                       for i in range(len(P4['Time'])))
        P4d1 = P4.Y[idx]
        val, idx = min((abs(P4.Time - (Par[71] + 5)), i)\
                       for i in range(len(P4.Time)))
        P4d6 = P4['Y'][idx]
        P4dend = P4['Y'][-1]

        val, idx = min((abs(LH['Time'] - (Par[71] - 1)), i)\
                       for i in range(len(LH['Time'])))
        LHdm1 = LH['Y'][idx]
        val, idx = min((abs(LH['Time'] - (Par[71] + 1)), i)\
                       for i in range(len(LH['Time'])))
        LHd1 = LH['Y'][idx]
        val, idx = min((abs(LH['Time'] - (Par[71] + 5)), i)\
                       for i in range(len(LH['Time'])))
        LHd6 = LH['Y'][idx]
        LHdend = LH['Y'][-1]

        sumFSH = FSH['Y'] + FSHmed['Y']

        val, idx = min((abs(FSH['Time'] - (Par[71] - 1)), i)\
                       for i in range(len(FSH['Time'])))
        FSHdm1 = sumFSH[idx]
        val, idx = min((abs(FSH['Time'] - (Par[71] + 1)), i)\
                       for i in range(len(FSH['Time'])))
        FSHd1 = sumFSH[idx]
        val, idx = min((abs(FSH['Time'] - (Par[71] + 5)), i)\
                       for i in range(len(FSH['Time'])))
        FSHd6 = sumFSH[idx]
        FSHdend = sumFSH[-1]

        MedInfo = [count10, count14, E2dm1, E2d1, E2d6, E2dend, P4dm1, P4d1, P4d6, P4dend,
                   LHdm1, LHd1, LHd6, LHdend, FSHdm1, FSHd1, FSHd6, FSHdend,
                   dosing_events1[0, 0], t, t - dosing_events1[0, 0]]

        count10 - count14
        count14
        count18
        count20
        dosing_events1
        t

    if Foll_ModelPop or Horm_ModelPop:
        totalcheck = 0
        if FollOvulInfo:
            for i in range(1, len(FollOvulInfo[-1])):
                Tovu = FollOvulInfo[2, i]
                t1 = Tovu - 5
                t2 = Tovu - 2
                t3 = Tovu + 0.05
                t4 = Tovu + 7
                val, idx1 = min((abs(FSH['Time'] - t1), i)\
                                for i in range(len(FSH['Time'])))
                val, idx2 = min((abs(FSH['Time'] - t2), i)\
                                for i in range(len(FSH['Time'])))
                val, idx3 = min((abs(FSH['Time'] - t3), i)\
                                for i in range(len(FSH['Time'])))
                val, idx4 = min((abs(FSH['Time'] - t4), i)\
                                for i in range(len(FSH['Time'])))

                check = 0

                if FSH['Y'][idx2] - FSH['Y'][idx1] < 0:
                    check += 1

                if FSH['Y'][idx3] - FSH['Y'][idx2] > 0:
                    check += 1

                if FSH['Y'][idx4] - FSH['Y'][idx3] < 0:
                    check += 1

                if check == 3:
                    totalcheck += 1

            check = 0

            if 21 < Cyclelengthmean < 40:
                check += 1

            if Cyclelengthstd < 4:
                check += 1

            if totalcheck >= (len(FollOvulInfo[-1]) - 1) * 0.75 and check == 2:
                Par[77] = 1

            if FollOvulInfo:
                H = np.array([Par] + [paraPoi] + [parafoll])
                ModelPop_Params = np.hstack((ModelPop_Params, H))

                F = np.array([Cyclelengthmean, Cyclelengthstd,\
                              FollperCyclemean, Par[77]])
                ModelPop_CycleInfo = np.hstack((ModelPop_CycleInfo, F))

    if SavePlotStuff:
        FileName = f'E2_{runind}.csv'
        fullFileName = os.path.join(DirStuff, FileName)
        np.savetxt(fullFileName, E2['Y'], delimiter=',')

        FileName = f'FSH_{runind}.csv'
        fullFileName = os.path.join(DirStuff, FileName)
        np.savetxt(fullFileName, FSH['Y'], delimiter=',')

        FileName = f'LH_{runind}.csv'
        fullFileName = os.path.join(DirStuff, FileName)
        np.savetxt(fullFileName, LH['Y'], delimiter=',')

        FileName = f'P4_{runind}.csv'
        fullFileName = os.path.join(DirStuff, FileName)
        np.savetxt(fullFileName, P4['Y'], delimiter=',')

        FileName = f'Time_{runind}.csv'
        fullFileName = os.path.join(DirStuff, FileName)
        np.savetxt(fullFileName, E2['Time'], delimiter=',')

        FileName = f'OvulationInfo_{runind}.csv'
        fullFileName = os.path.join(DirStuff, FileName)
        np.savetxt(fullFileName, FollOvulInfo, delimiter=',')

        FileName = 'Solutions.csv'
        fullFileName = os.path.join(DirStuff, FileName)
        np.savetxt(fullFileName, solutions['Y'], delimiter=',')

    if SaveSim:
        FileName = f'DomFolGrowth_{runind}.csv'
        fullFileName = os.path.join(DirStuff, FileName)
        np.savetxt(fullFileName, FollOvulInfo[3, :], delimiter=',')
        
        FileName = f'Cyclelength_{runind}.csv'
        fullFileName = os.path.join(DirStuff, FileName)
        np.savetxt(fullFileName, CycleInfo[0, :], delimiter=',')
        
        FileName = f'CycleFollCount_{runind}.csv'
        fullFileName = os.path.join(DirStuff, FileName)
        np.savetxt(fullFileName, CycleInfo[1, :], delimiter=',')
        
        if (Par[63] == 1 and count18 >= 3) or (Par[63] == 1 and count20 >= 1):
            FileName = 'MedInfo.csv'
            fullFileName = os.path.join(DirStuff, FileName)
            M = np.loadtxt(fullFileName, delimiter=',')
            M = np.vstack((M, MedInfo))
            np.savetxt(fullFileName, M, delimiter=',')
