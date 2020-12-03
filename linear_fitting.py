# coding: utf-8
# Author : Edmond Chaussidon (CEA)

class LeastSquares:  # override the class with a better one
    def __init__(self, model, x, y, yerr):
        self.model = model  # model predicts y for given x
        self.x = np.array(x)
        self.y = np.array(y)
        self.yerr = np.array(yerr)
        self.func_code = make_func_code(describe(self.model)[1:])

    def __call__(self, *par):  # par are a variable number of model parameters
        ym = self.model(self.x, *par)
        chi2 = sum((self.y - ym)**2/(self.yerr)**2)
        return chi2

##TO DOOOo
