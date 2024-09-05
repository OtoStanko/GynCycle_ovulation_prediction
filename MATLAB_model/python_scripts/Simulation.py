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


def Simulation(para, paraPoi, parafoll, Par, tb, te,
               StartValues, StartTimes, FSHVec, Stim, runind, settings):
    
    # Integration period
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

    # remove the values of E2 and P4 from the initial vector
    # they will be computed every step in the Follicle function
    # this allows us to use ODE solver, as DAE is not in the scipy
    e2p4_lvls = [[StartValues[1]], [StartValues[2]]]
    y0 = np.array(StartValues)
    y0 = np.delete(y0, [1, 2])

    # Values for tracking the follicles 
    FollCounter = 1

    # Class to save follicles and their properties                                          
    Follicles = FollicleClass(y0Foll, FSHVec[FollCounter - 1], t)  
    FollCounter += 1

    # Values needed for the first integrations
    TimeCounter = 0
    TimeFol = np.array([t])

    # Arrays to save times when new follicles emerge or when they can't emerge
    NewFollicle = []
    NoNewFollicle = []



    # Global variables
    global ModelPop_Params
    global ModelPop_CycleInfo

    # Tracking the concentrations of important hormone species
    #E2 = {'Time': np.array([t]), 'Y': np.array([y0[-17]])}
    #P4 = {'Time': np.array([t]), 'Y': np.array([y0[-16]])}
    FSH = {'Time': np.array([t]), 'Y': np.array([y0[-11]])}
    FSHRez = {'Time': np.array([t]), 'Y': np.array([y0[-15]])}
    LH = {'Time': np.array([t]), 'Y': np.array([y0[-9]])}
    GnRH = {'Time': np.array([t]), 'Y': np.array([y0[-3]])}
    GnRHRezA = {'Time': np.array([t]), 'Y': np.array([y0[-6]])}
    FSHmed = {'Time': np.array([t]), 'Y': np.array([y0[-1]])}
    solutions = {'Time': np.array([t]), 'Y': np.array([y0[1:]])}

    # Main loop
    while t < te:
        print("t:", round(t, 3))

        # Follicles recruitment depending on FSH concentration in the system
        fshAll = y0[-11] + y0[-1]
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
        # This is not needed, E2 and P4 no longer in y0
        #M[NumFollicles, NumFollicles] = 0     # alg. eq. for E2
        #M[NumFollicles + 1, NumFollicles + 1] = 0     # alg. eq. for P4
        M[NumFollicles + 13, NumFollicles + 13] = 0     # alg. eq. for LH med
        M[NumFollicles + 14, NumFollicles + 14] = 0     # alg. eq. for FSH med

        # Event function stops the integration when an ovulation takes place within the interval tspan
        # solve differential equations
        para[0] = 0
        Y = np.array([])
        T = np.array([])

        # Main simulation part with the ODE solver
        #print("para 160", para)
        #print("y0 supposedly y", y0)
        event_function = lambda t_ev, y_ev: EvaluateFollicle(t_ev, y_ev, para, parafoll, LH)
        event_function.terminal = True
        sol = solve_ivp(lambda t, y: FollicleFunction(
                            t, y, Tovu, Follicles, para,
                            parafoll, Par, Stim,
                            settings),
                        tspan, y0, method='LSODA',
                        events=event_function)
        T = sol.t
        Y = sol.y.T
        #print("Events:", sol.t_events)

        for i in range(Follicles.NumActive):
            # saves all times of the foll that was active during last run
            Follicles.Follicle[Follicles.Active[i]-1]['Time'] = \
                    np.concatenate((Follicles.Follicle[Follicles.Active[i]-1]['Time'], T[1:]))
            # saves all sizes of the foll that was active during last run
            Follicles.Follicle[Follicles.Active[i]-1]['Y'] = \
                    np.concatenate((Follicles.Follicle[Follicles.Active[i]-1]['Y'], Y[1:, i]))

        # saves the measuring times of the active foll.
        TimeFol = np.concatenate((TimeFol, T[1:]))
        # saves last Y values of the follicles
        LastYValues = Y[-1, :] # What is assigned here? Are the values correct?

        # E2 production
        # calculate E2 concentration
        # recalculate the x in every time step
        solution_time_points = T[1:]
        num_t = len(solution_time_points)
        for tidx in range(len(solution_time_points)):
            #x = y0[:NumFollicles]
            x = np.array([])
            for i in range(NumFollicles):
                if NumFollicles > 0 and para[0] == 0 and Follicles.Follicle[Follicles.Active[i] - 1]['Destiny'] == 4:
                    x = np.append(x,0)
                else:
                    x = np.append(x, Follicles.Follicle[Follicles.Active[i] - 1]['Y'][-num_t+tidx])
            t = solution_time_points[tidx]
            SF = np.pi * np.sum((x ** Par[56]) / (x ** Par[56] + Par[57] ** Par[56]) * (x ** 2))
            e2p4_lvls[0].append(Par[74] + (Par[58] + Par[59] * SF) + Par[60] * np.exp(-Par[61] * (t - (Tovu + 7)) ** 2))
        # save values for P4
        solution_time_points = T[1:]
        for t in solution_time_points:
            e2p4_lvls[1].append(Par[75] + Par[62] * np.exp(-Par[61] * (t - (Tovu + 7)) ** 2))
        # save values for LH
        LH['Time'] = np.concatenate((LH['Time'], T[1:]))
        LH['Y'] = np.concatenate((LH['Y'], Y[1:, -9]))
        # save values for FSH 
        FSH['Time'] = np.concatenate((FSH['Time'], T[1:]))
        FSH['Y'] = np.concatenate((FSH['Y'], Y[1:, -11]))
        # save values for FSH Rezeptor
        FSHRez['Time'] = np.concatenate((FSHRez['Time'], T[1:]))
        FSHRez['Y'] = np.concatenate((FSHRez['Y'], Y[1:, -15]))
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
                Follicle1['FSHSensitivity'] = FSHVec[FollCounter-1]
                FollCounter += 1
            FSHSSave = Follicles.ActiveFSHS
            Follicles.ActiveFSHS = np.concatenate(
                (Follicles.ActiveFSHS, [Follicle1['FSHSensitivity']]))

            # Test if Follicle(s) could survive
            # (slope of growth-function positive or negative)
            testyvalues = LastYValues[:-para[1]-1]
            testyvalues = np.concatenate(
                (testyvalues, Follicle1['Y'], LastYValues[-para[1]:]))
            para[0] = 1
            testyslope = FollicleFunction(T[-1], testyvalues, Tovu,
                                          Follicles, para, parafoll,
                                          Par, Stim, settings)

            # if follicle got chance to survive -> initiate new follicle and update follicles-vector
            if testyslope[-para[1]-1] > 0:
                Follicle1['Time'] = np.array([T[-1]])
                Follicle1['TimeDecrease'] = 0
                Follicle1['Destiny'] = -1
                Follicles.Number += 1
                Follicle1['Number'] = Follicles.Number
                Follicles.NumActive += 1
                Follicles.Active = np.concatenate(
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
        res = FollicleFunction(T[-1], LastYValues, Tovu, Follicles,
                               para, parafoll, Par, Stim,
                               settings)
        # reset vector of active FSH sensitivities
        Follicles.ActiveFSHS = []

        # loop over all active follicles to set new destiny   
        for i in range(Follicles.NumActive):
            # Save y-values of i-th (current) follicle
            yCurFoll = LastYValues[i]
            # slope is negative so the follicle is decreasing in size
            if res[i] <= 0: 
                Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] = -2
            
            # follicle is big, but doesn't ovulate yet because there is not enough LH
            if (yCurFoll >= parafoll[6]) and (Y[-1, -9] < parafoll[9] and
               Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] == -1):
                Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] = 3
                Follicles.Follicle[Follicles.Active[i]-1]['TimeDecrease'] = t

            if (Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] == 3 and 
               (t - Follicles.Follicle[Follicles.Active[i]-1]['TimeDecrease']) >= parafoll[10]):
                Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] = -2
            
            # if LH high enough dominant follicle rest until ovulation shortly after LH peak 
            if Y[-1, -9] >= parafoll[9]:
                if (yCurFoll >= parafoll[6] and Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] == -1) or \
                   (yCurFoll >= parafoll[6] and Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] == 3):
                    th = t - 0.5
                    idx = np.argmin(np.abs(LH['Time'] - th))
                    if LH['Y'][idx] >= parafoll[9]:
                        Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] = 4
                        Follicles.Follicle[Follicles.Active[i]-1]['TimeDecrease'] = t
            
            # Follicle ovulates
            if (Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] == 4 and 
                Follicles.Follicle[Follicles.Active[i]-1]['TimeDecrease'] <= t):
                Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] = 1
                Tovu = T[-1]
                OvulationNumber = i
                if Stim:
                    if Tovu > Par[70] and Par[63] == 0: 
                        Par[70] = Tovu

            # Follicles that ovulated are no longer active
            # Follicles that are dead and has been active for more than 20 days are also not considered
            #   active to optimize the simulation
            if Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] != 1 and \
                    (Follicles.Follicle[Follicles.Active[i]-1]['Time'][0] + 20 > t or
                     Follicles.Follicle[Follicles.Active[i]-1]['Y'][-1] != 0):
                # put the follicle back to the list of actives and its FSH
                ActiveHelp.append(Follicles.Active[i])
                # sensitivity back in the FSH vector...
                Follicles.ActiveFSHS.append(Follicles.Follicle[Follicles.Active[i]-1]['FSHSensitivity'])


        # Update list of active follicles
        Follicles.Active = ActiveHelp
        # find out how many follicles are active...
        Follicles.NumActive = len(ActiveHelp)
        # determine new initial values for all differential equations
        y0old = []
        for i in range(Follicles.NumActive):
            y0old.append(Follicles.Follicle[Follicles.Active[i]-1]['Y'][-1])
        y0old = np.array(y0old)
        y0 = np.concatenate((y0old, LastYValues[-para[1]:]))

        # integration end reached
        t = T[-1]
        if te - t < 0.001:
            t = te

    # plotting
    if settings.showPlots:
        hf = plt.figure(1)
        plt.clf()
        widthofline = 2

    # vector to save informations about the ovulating follicle
    FollOvulInfo = []
    # save t_start t_end destiny of all follicles
    FollInfo = []

    for i in range(Follicles.Number):
        # fill follicle information variable...
        help_info = [Follicles.Follicle[i]['Time'][0],
                     Follicles.Follicle[i]['Time'][-1],
                     Follicles.Follicle[i]['Destiny'],
                     Follicles.Follicle[i]['FSHSensitivity'], i]
        FollInfo.append(help_info)

        FollInfo2 = np.column_stack((Follicles.Follicle[i]['Time'],
                                     Follicles.Follicle[i]['Y']))

        if Follicles.Follicle[i]['Destiny'] == 1 and Follicles.Follicle[i]['Time'][0] > 20:
            helpFOT = [i, Follicles.Follicle[i]['Time'][0],
                       Follicles.Follicle[i]['Time'][-1],
                       (Follicles.Follicle[i]['Y'][-1] - Follicles.Follicle[i]['Y'][0]) /
                       (Follicles.Follicle[i]['Time'][-1] - Follicles.Follicle[i]['Time'][0] - 2)]
            FollOvulInfo.append(helpFOT)

        if settings.showPlots:
            plt.plot(Follicles.Follicle[i]['Time'], Follicles.Follicle[i]['Y'], color=[0, 0, 0],
                     label='x1', linewidth=widthofline)

    # Cycle length
    FollOvulInfo = np.array(FollOvulInfo)
    OvuT = FollOvulInfo[:, 2]
    Cyclelength = np.diff(OvuT)
    Cyclelengthmean = np.mean(Cyclelength)
    Cyclelengthstd = np.std(Cyclelength)
    NumCycles = len(Cyclelength)
    print("Number of cycles in {} days: {}".format(te, NumCycles))
    print("Mean cycle length: {}+-{} days".format(Cyclelengthmean, Cyclelengthstd))


    FollperCycle = []
    for i in range(NumCycles):
        t1 = OvuT[i]
        t2 = OvuT[i + 1]
        count = 0
        tp = len(FollInfo[0])  # Define or load your FollInfo data
        for j in range(tp):
            if t1 < FollInfo[j][0] < t2:
                count += 1
        FollperCycle.append(count)
    FollperCyclemean = np.mean(FollperCycle)

    #a = sum(FollperCycle)
    #rest = n - a  # Define n appropriately

    #CycleInfo = np.array([[0] + list(Cyclelength), [rest] + FollperCycle, OvuT])

    if settings.showPlots:  # Define ShowPlots appropriately
        # Threshold when you can measure the follicle size
        plt.plot(plt.xlim(), [4, 4], color='b')
        plt.xlabel('time in d', fontsize=15)
        plt.ylabel('follicle diameter in mm', fontsize=15)
        plt.ylim([-30, 70])
        ax = plt.gca()
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=15)
        plt.legend(['Follicle size'],
                   fontsize=15, loc='upper left')

        # FSH
        plt.figure(2)
        hfsh, = plt.plot(FSH['Time'], FSH['Y'], color=[1 / 2, 1, 1 / 2],
                 linewidth=widthofline, label='FSH')

        # P4
        p4, = plt.plot(LH['Time'], e2p4_lvls[1], linewidth=2, label='P4')

        # Plot for the FSH and P4
        h = [hfsh, p4]
        plt.xlabel('time in d', fontsize=15)
        plt.ylabel('FSH and P4 c', fontsize=15)
        ax = plt.gca()
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=15)
        plt.legend(h, ['FSH',  'P4'],
                   fontsize=15, loc='upper left')

        # GnRH
        plt.figure(4)
        plt.plot(GnRH['Time'], GnRH['Y'], linewidth=2)
        plt.gca().tick_params(labelsize=24)
        plt.legend(['GnRH'], fontsize=24, loc='upper left')

        # E2 and LH
        plt.figure(7)
        e2, = plt.plot(LH['Time'], e2p4_lvls[0], linewidth=2, label='E2')

        # LH
        hLH, = plt.plot(LH['Time'], LH['Y'], color=[1, 1 / 4, 1 / 2],
                        linewidth=widthofline, label='LH')
        h = [e2, hLH]
        plt.gca().set_box_aspect(1)
        plt.gca().tick_params(labelsize=24)
        plt.legend(h, ['E2', 'LH'], fontsize=24, loc='upper left')


        # Indexes of solutions are number from Model28_ODE + 1
        solutions = np.column_stack((solutions['Time'], solutions['Y']))

        plt.figure(10)
        plt.plot(FSHRez['Time'], FSHRez['Y'])
        plt.legend(['FSHRez'], fontsize=24, loc='upper left')
        plt.show()

    """if Par[64] == 1:
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
        P4d1 = P4['Y'][idx]
        val, idx = min((abs(P4['Time'] - (Par[71] + 5)), i)\
                       for i in range(len(P4['Time'])))
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
        t"""

    if settings.savePlotsStuff:
        FileName = f'E2_{runind}.csv'
        fullFileName = os.path.join(settings.outputDir, FileName)
        np.savetxt(fullFileName, e2p4_lvls[0], delimiter=',')

        FileName = f'FSH_{runind}.csv'
        fullFileName = os.path.join(settings.outputDir, FileName)
        np.savetxt(fullFileName, FSH['Y'], delimiter=',')

        FileName = f'LH_{runind}.csv'
        fullFileName = os.path.join(settings.outputDir, FileName)
        np.savetxt(fullFileName, LH['Y'], delimiter=',')

        FileName = f'P4_{runind}.csv'
        fullFileName = os.path.join(settings.outputDir, FileName)
        np.savetxt(fullFileName, e2p4_lvls[1], delimiter=',')

        FileName = f'Time_{runind}.csv'
        fullFileName = os.path.join(settings.outputDir, FileName)
        np.savetxt(fullFileName, LH['Time'], delimiter=',')

        FileName = f'OvulationInfo_{runind}.csv'
        fullFileName = os.path.join(settings.outputDir, FileName)
        np.savetxt(fullFileName, FollOvulInfo, delimiter=',')

        """FileName = 'Solutions.csv'
        fullFileName = os.path.join(DirStuff, FileName)
        np.savetxt(fullFileName, solutions['Y'], delimiter=',')"""

    """if settings.saveSim:
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
            np.savetxt(fullFileName, M, delimiter=',')"""
