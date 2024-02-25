import unittest
import numpy as np
from Dynamics import angularAcceleration
import Plotter
from Quaternion import calculateCurrentOrientation

class TestSpacecraftDynamics(unittest.TestCase):

  def setUp(self):
    self.I = np.diag([900, 800, 600])
    self.qInit = np.array([1.0, 0.0, 0.0, 0.0])  # Initial orientation
    self.omegaInit = np.array([0.0, 0.0, 0.0])
    self.dt = 0.01
    self.simTime = 100

  def tesConstantTorque(self): # In one axis
    T = np.array([2.0, 0.0, 0.0])

    quatHistory = self.simulate(T)
    Plotter.plot3D(quatHistory)

  def testTwoAxisConstantTorque(self):
    T = np.array([2.0, 2.0, 0.0])
    quatHistory = self.simulate(T)
    Plotter.plot3D(quatHistory)

  def simulate(self, T):
    """
    Simulates spacecraft dynamics.
    """
    omega = self.omegaInit
    quatHistory = [self.qInit]

    for t in np.arange(self.dt, self.simTime, self.dt):
      omegadot = angularAcceleration(T, omega, self.I)

      omega += omegadot * self.dt

      q = calculateCurrentOrientation(quatHistory[-1], omega, self.dt)

      quatHistory.append(q)

    return np.array(quatHistory)


if __name__ == '__main__':
    unittest.main()
