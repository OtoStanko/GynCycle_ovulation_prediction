import numpy as np

from HormoneModel import ODE_Model_NormalCycle

def FollicleFunction(t, y, Tovu, Follicles, para, parafoll, Par, dd1, Stim, LutStim, FollStim, DoubStim, firstExtraction, e2p4_lvls):
    # determine number of active follicles
    NumFollicles = y.shape[0] - para[1] # does not work when a new follicle is added
    #NumFollicles = len(Follicles.Active)

    if NumFollicles > 0:
        x = y[:NumFollicles]
    else:
        x = np.array([0])

    if NumFollicles > 0 and para[0] == 0:
        #print("Follicles number", Follicles.Number)
        #print("Supposedly num of follicles", NumFollicles)
        #print(Follicles.Follicle)
        for i in range(NumFollicles):
            if Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] in [-2, -3]:
                x[i] = 0

    print("Folls +..., ", y)
    print("Num follicles:", NumFollicles, "Current P4 lvl: ", e2p4_lvls[1][-1])

    # solve differential equations
    dy = ODE_Model_NormalCycle(t, y, Par, e2p4_lvls[0][-1], e2p4_lvls[1][-1]) # E2 and p4 as  params
    f = dy.copy()

    r = len(y)
    fshrezcomp = y[r-15]
    #p4all = y[r-16]
    p4all = e2p4_lvls[1][-1]
    SumV = np.sum(x ** parafoll[0])

    for i in range(NumFollicles):
        # FSH sensitivity of the follicles
        fFSH = Follicles.ActiveFSHS[i]
        fsize = y[i]

        # growth rate
        gamma = parafoll[1] * ((1 / (1 + (p4all / 3) ** 3)) + (fshrezcomp ** 5) / (0.95 ** 5 + fshrezcomp ** 5))

        # negative Hill function for FSH with kappa(proportion of self harm)
        kappa = parafoll[4] * (0.55 ** 10 / (0.55 ** 10 + fshrezcomp ** 10))

        xi = parafoll[2]
        ffsh = (fshrezcomp ** 4) / (fshrezcomp ** 4 + (fFSH) ** 4)

        # follicles growth equation
        X = ffsh * (xi - y[i]) * y[i] * (gamma - (kappa * (SumV - (parafoll[3] * (y[i] ** parafoll[0])))))

        if para[0] == 1:
            if X <= 0:
                NoFoll = X

        if para[0] == 0:
            if (X <= parafoll[11] or
                Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] == -2 or
                (X <= parafoll[12] and (t - Follicles.Follicle[Follicles.Active[i]-1]['Time'][0]) >= parafoll[14] and Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] == 3) or
                (X <= parafoll[12] and (t - Follicles.Follicle[Follicles.Active[i]-1]['Time'][0]) >= parafoll[13] and Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] == -1) or
                (Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] == 3 and (t - Follicles.Follicle[Follicles.Active[i]-1]['TimeDecrease']) >= parafoll[10]) or
                (Follicles.Follicle[Follicles.Active[i]-1]['Time'][0] - Follicles.Follicle[Follicles.Active[i]-1]['Time'][-1] > parafoll[14])):
                # set time the follicle starts to decrease & set destiny to decrease
                #print(Follicles.Follicle)
                #print(Follicles.Active)
                if Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] != -2:
                    Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] = -2
                    Follicles.Follicle[Follicles.Active[i]-1]['TimeDecrease'] = t
                # to decrease the size of the follicle faster
                f[i] = -0.05 * y[i] * (t - Follicles.Follicle[Follicles.Active[i]-1]['TimeDecrease'])
            elif Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] == -3:
                f[i] = -1000 * y[i]
            else:
                # if not dying use normal equation
                f[i] = X
        else:
            # if called to test use normal equation
            f[i] = X

    # E2 production
    if NumFollicles > 0 and para[0] == 0:
        for i in range(NumFollicles):
            if Follicles.Follicle[Follicles.Active[i]-1]['Destiny'] == 4:
                x[i] = 0

    #SF = np.pi * np.sum((x ** Par[56]) / (x ** Par[56] + Par[57] ** Par[56]) * (x ** 2))

    # calculate E2 concentration
    SF = np.pi * np.sum((x ** Par[56]) / (x ** Par[56] + Par[57] ** Par[56]) * (x ** 2))
    e2p4_lvls[0].append(Par[74] + (Par[58] + Par[59] * SF) - Par[60] * np.exp(-Par[61] * (t - (Tovu + 7)) ** 2))

    # Calculation of P4 values
    e2p4_lvls[1].append(Par[75] + Par[62] * np.exp(-Par[61] * (t - (Tovu + 7)) ** 2))
    print("Num follicles:", NumFollicles, "Current P4 lvl: ", e2p4_lvls[1][-1])

    """# calculate E2 concentration
    f[NumFollicles] = y[NumFollicles] - Par[74] - (Par[58] + Par[59] * SF) -\
                      Par[60] * np.exp(-Par[61] * (t - (Tovu + 7)) ** 2)

    # Calculation of P4 values
    #print("P4: ", Par[75] + Par[62] * np.exp(-Par[61] * (t - (Tovu + 7)) ** 2))
    f[NumFollicles + 1] = y[NumFollicles + 1] - Par[75] - Par[62] *\
                          np.exp(-Par[61] * (t - (Tovu + 7)) ** 2)"""

    # Calculation of FSHAnaC
    if Stim == 0:
        f[NumFollicles + 14] = y[NumFollicles + 14] - 0
        f[NumFollicles + 13] = y[NumFollicles + 13] - 0

    return f
