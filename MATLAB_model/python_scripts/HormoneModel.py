import numpy as np

def ODE_Model_NormalCycle(t, y, Par):
    #print("t", t)
    #print("y", y)
    #print("Par", Par)
    #r, c = y.shape
    #dy = np.zeros((r, c))
    r = len(y)
    dy = np.zeros(r)
    
    i_FSH_med  = r - 1
    i_LH_med   = r - 2
    i_GnRH     = r - 3
    i_RecGa    = r - 4
    i_RecGi    = r - 5
    i_GReca    = r - 6
    i_GReci    = r - 7
    i_RP_LH    = r - 8
    i_LH       = r - 9
    i_RP_FSH   = r - 10
    i_FSH      = r - 11
    i_FSHfoll  = r - 12
    i_RFSH     = r - 13
    i_RFSH_des = r - 14
    i_FSHR     = r - 15
    i_P4       = r - 16
    i_E2       = r - 17

    yGfreq = Par[0] / (1 + (y[i_P4] / Par[1]) ** Par[2]) * (1 + y[i_E2] ** Par[3] / (Par[4] ** Par[3] + y[i_E2] ** Par[3]))
    yGmass = Par[5] * (y[i_E2] ** Par[6] / (Par[7] ** Par[6] + y[i_E2] ** Par[6]) + Par[8] ** Par[9] / (Par[8] ** Par[9] + y[i_E2] ** Par[9]))

    dy[i_GnRH] = yGmass * yGfreq - Par[10] * y[i_GnRH] * y[i_RecGa] + Par[11] * y[i_GReca] - Par[12] * y[i_GnRH]
    dy[i_RecGa] = Par[11] * y[i_GReca] - Par[10] * y[i_GnRH] * y[i_RecGa] - Par[13] * y[i_RecGa] + Par[14] * y[i_RecGi]
    dy[i_RecGi] = Par[15] + Par[13] * y[i_RecGa] - Par[14] * y[i_RecGi] + Par[16] * y[i_GReci] - Par[17] * y[i_RecGi]
    dy[i_GReca] = Par[10] * y[i_GnRH] * y[i_RecGa] - Par[11] * y[i_GReca] - Par[18] * y[i_GReca] + Par[19] * y[i_GReci]
    dy[i_GReci] = Par[18] * y[i_GReca] - Par[19] * y[i_GReci] - Par[16] * y[i_GReci] - Par[20] * y[i_GReci]

    hp_e2 = 1.5 * (y[i_E2] / Par[21]) ** Par[22] / (1 + (y[i_E2] / Par[21]) ** Par[22])
    hp_freq = ((yGfreq / Par[23]) ** Par[24]) / (1 + (yGfreq / Par[23]) ** Par[24])
    f_LH_prod1 = Par[25] + Par[26] * hp_e2
    hm_p4 = 1 + Par[29] * (y[i_P4] / Par[27]) ** Par[28]
    #print(hm_p4, y[i_P4])

    f_LH_prod = (f_LH_prod1 / hm_p4) * (1 + hp_freq)
    f_LH_rel = (Par[33] + Par[34] * (y[i_GReca] / Par[35]) ** Par[36] / (1 + (y[i_GReca] / Par[35]) ** Par[36]))

    dy[i_RP_LH] = f_LH_prod - f_LH_rel * y[i_RP_LH]
    dy[i_LH] = (1 / Par[37]) * f_LH_rel * y[i_RP_LH] - Par[38] * y[i_LH]
    f_FSH_prod1 = Par[39]
    f_FSH_prod2 = 1 + (y[i_P4] / Par[40]) ** Par[41]
    hm_freq = 1 / (1 + (yGfreq / Par[42]) ** Par[43])

    f_FSH_prod = (f_FSH_prod1 / f_FSH_prod2) * hm_freq
    f_FSH_rel = Par[44] + Par[45] * (Par[30] * ((y[i_GReca] / Par[46]) ** Par[47] / (1 + (y[i_GReca] / Par[46]) ** Par[47]))) * 1 / (1 + (y[i_E2] / Par[72]) ** Par[73])

    dy[i_RP_FSH] = f_FSH_prod - f_FSH_rel * y[i_RP_FSH]
    dy[i_FSH] = (1 / Par[37]) * f_FSH_rel * y[i_RP_FSH] - Par[49] * y[i_FSH] - Par[50] * y[i_FSH]
    dy[i_FSHfoll] = Par[49] * (y[i_FSH] + y[i_FSH_med]) * Par[51] - Par[52] * y[i_FSHfoll] * y[i_RFSH] - Par[53] * y[i_FSHfoll]

    dy[i_RFSH] = Par[54] * y[i_RFSH_des] - Par[52] * y[i_FSHfoll] * y[i_RFSH]
    dy[i_FSHR] = Par[52] * y[i_FSHfoll] * y[i_RFSH] - Par[55] * y[i_FSHR]
    dy[i_RFSH_des] = Par[55] * y[i_FSHR] - Par[54] * y[i_RFSH_des]

    return dy

