import numpy as np
class Diff:
    def __init__(self, x,dx,dy):
        self.value = [x,dx,dy]

    def __getitem__(self, key):
        return self.value[key]
        
    def __mul__(self, other):
        return Diff(self[0]*other[0], self[0]*other[1]+self[1]*other[0], self[0]*other[2]+self[2]*other[0])

    def __add__(self, other):
        return Diff(self[0] + other[0], self[1] + other[1], self[2] + other[2])

    def __div__(self, other):
        return Diff(self[0] / other[0], (self[1] * other[0] - self[0] * other[1]) / other[0]**2, (self[2] * other[0] - self[0] * other[2]) / other[0]**2)

x = np.arange(-1,1.1,0.1)
y = np.arange(-1,1.1,0.1)
X,Y = np.meshgrid(x,y)
a = np.array([Diff(x,1,0) for x in X.flatten()]).reshape(X.shape)
b = np.array([Diff(y,0,1) for y in Y.flatten()]).reshape(Y.shape)

from pyfemsolver.solverlib.element import H1Fel
from pyfemsolver.solverlib import integrationrules
a,b,c = integrationrules.get_integration_rule_trig(3)

fel = H1Fel(order = 2)

fel.shape_functions()