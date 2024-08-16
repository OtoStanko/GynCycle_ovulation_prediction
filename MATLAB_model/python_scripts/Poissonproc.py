import numpy as np

def poissonproc(lambda_, tspan):
    tb = tspan[0]  # start time
    te = tspan[1]  # end time

    # alternative method:
    N = np.random.poisson(lambda_ * (te - tb))
    U = np.random.rand(N)
    Usort = np.sort(U)
    S = tb + (te - tb) * Usort
    # length(S)

    return S

