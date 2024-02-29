import unittest
import numpy as np
from Dynamics import update
import Plotter
from Quaternion import Quaternion

class TestSpacecraftDynamics(unittest.TestCase):

  def setUp(self):
    self.mass = 750
    self.I = np.diag([900, 800, 600])
    self.qInit = Quaternion(1,np.array([ 0.0, 0.0, 0.0]))  # Initial orientation
    self.omegaInit = np.array([0.0, 0.0, 0.0])
    self.dt = 0.01
    self.simTime = 100

  def tesConstantTorque(self): # In one axis
    T = np.array([2.0, 0.0, 0.0])

    quatHistory = self.simulate(T)
    Plotter.plot3D(quatHistory, self.mass, self.I)

  def testTwoAxisConstantTorque(self):
    T = np.array([2.0, 2.0, 0.0])
    quatHistory = self.simulate(T)
    Plotter.plot3D(quatHistory, self.mass, self.I)

  def simulate(self, T):
    omega = self.omegaInit
    quatHistory = [self.qInit]

    for t in np.arange(self.dt, self.simTime, self.dt):
      omega, q = update(quatHistory[-1], T, omega, self.I, self.dt)
      quatHistory.append(q)
    return np.array(quatHistory)


if __name__ == '__main__':
    unittest.main()
