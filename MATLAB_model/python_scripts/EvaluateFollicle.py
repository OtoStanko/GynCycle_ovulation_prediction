import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def EvaluateFollicle(t, y, para, parafoll, LH):
    #print("para evalfol", np.array(para))
    m = parafoll[6]
    th = t - 0.5
    idx = np.argmin(np.abs(LH['Time'] - th))
    y_lh = LH['Y'][idx]
    #print("LH in time:", th, LH['Y'][idx])

    # number of current follicle(s)
    #print(y)
    #print(y.shape[0])
    #print(para)
    NumFoll = y.shape[0] - para[1]
    # size(s) of current follicle(s)
    FollSize = y[:NumFoll]

    #return (m + 0.001) - np.max(FollSize) + parafoll[9] - min(y_lh, parafoll[9])
    return (m + 0.001) - max(np.max(FollSize), m) + parafoll[9] - min(y_lh, parafoll[9])

    if np.max(FollSize) >= (m - 0.001) and y_lh >= parafoll[9]:
        # a follicle might ovulate
        lookfor = (m - 0.001) - np.max(FollSize)
        stop = 1
        # locates zeros where the event function is decreasing
        direction = -1
        #return 0
        return (m - 0.001) - np.max(FollSize)
    else:
        lookfor = 0
        stop = 0
        direction = -1
        return y_lh
        return (m - 0.001) - np.max(FollSize) + y_lh
        return ((m - 0.001) - np.max(FollSize)) * (parafoll[9] - y_lh)


    return lookfor, stop, direction

