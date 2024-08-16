class FollicleClass:
    def __init__(self, y0Follicle, FSHSensitivity, t):
        self.Number = 1  # number of active foll
        self.Follicle = []  # number of created foll
        self.Active = [1]  # number of currently active foll
        self.NumActive = 1  # all foll structures
        self.ActiveFSHS = []  # all foll FSH sensitivities

        Foll = {
            'Number': 1,
            'Time': [t],
            'Y': y0Follicle,
            'Destiny': -1,  # the foll destinys: -1: not clear yet, 0:died, 1:ovulated, -2:decreasing in size but did not yet died
            'FSHSensitivity': FSHSensitivity,
            'TimeDecrease': 0  # time the size of the foll started to decrease
        }

        self.Follicle.append(Foll)
        self.ActiveFSHS.append(Foll['FSHSensitivity'])

