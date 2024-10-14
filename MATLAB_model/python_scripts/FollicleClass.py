import numpy as np

class FollicleClass:
    def __init__(self, y0Follicle, FSHSensitivity, t):
        self.Number = 1  # number of active foll
        self.Follicle = np.array([])  # number of created foll
        self.Active = np.array([1])  # number of currently active foll
        self.NumActive = 1  # all foll structures
        self.ActiveFSHS = np.array([])  # all foll FSH sensitivities

        Foll = {
            'Number': 1,
            'Time': np.array([t]),
            'Y': np.array([y0Follicle]),
            'Destiny': -1,  # the foll destinies: -1: not clear yet, 0:died, 1:ovulated, -2:decreasing in size but did not yet died
            'FSHSensitivity': FSHSensitivity,
            'TimeDecrease': 0  # time the size of the foll started to decrease
        }

        self.Follicle = np.append(self.Follicle, Foll)
        self.ActiveFSHS = np.append(self.ActiveFSHS, Foll['FSHSensitivity'])

