import numpy as np

def evaluate_follicle(t, y, para, parafoll, LH):
    m = parafoll[6]
    th = t - 0.5
    idx = np.argmin(np.abs(LH['Time'] - th))
    y_lh = LH['Y'][idx]

    # number of current follicle(s)
    NumFoll = y.shape[0] - para[1]
    # size(s) of current follicle(s)
    FollSize = y[:NumFoll]

    if np.max(FollSize) >= (m - 0.001) and y_lh >= parafoll[9]:
        # a follicle might ovulate
        lookfor = (m - 0.001) - np.max(FollSize)
        stop = 1
        # locates zeros where the event function is decreasing
        direction = -1
    else:
        lookfor = 0
        stop = 0
        direction = -1

    return lookfor, stop, direction

