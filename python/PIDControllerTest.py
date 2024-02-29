import unittest
import numpy as np
from PIDController import PIDController
from Quaternion import calculateCurrentOrientation, eulerToQuaternion, Quaternion
import Plotter
from SensorAndActuatorModel import Gyroscope, ReactionWheel

class TestPIDController(unittest.TestCase):

  def setUp(self):
      self.mass = 750
      self.I = np.diag([900, 800, 600])
      self.torqueInit = np.array([0.0, 0.0, 0.0])  # No External Torque
      self.dt = 0.01
      self.simTime = 100
      self.rwMass = 1 #kg
      self.rwRadius = 0.5 #m

  def test1(self):
    O1 = np.array([60, -30, -10])

    desiredOrientation = np.array([O1])
    desiredQ = np.array([eulerToQuaternion(*o) for o in desiredOrientation])

    omegaInit = np.array([0.0, 0.0, 0.0])

    gyroNoiseStd = 0.005
    gyroscope = Gyroscope(self.I, gyroNoiseStd, omegaInit, self.torqueInit, self.dt)

    Kp = 20 * (self.I.diagonal()/np.max(self.I))
    Ki = 0.4 * (self.I.diagonal()/np.max(self.I))
    Kd = 400 * (self.I.diagonal()/np.max(self.I))
    controller = PIDController(Kp, Ki, Kd, self.dt)

    rwMaxRPM = 2000
    rwInitialSpeed = np.zeros(3)

    reactionWheel = ReactionWheel(rwMaxRPM, self.rwMass, self.rwRadius, rwInitialSpeed, self.dt)
    realOrientation = self.simulate(desiredQ, gyroscope, controller, reactionWheel)
    Plotter.plotOmega(np.array(gyroscope.omegaList), title='Omega of the spacecraft')
    np.save('realOrientation', realOrientation)
    Plotter.plot3D(realOrientation, self.mass, self.I)

  def simulate(self, desiredQ, gyro, controller, reactionwheel):
    realOrientation = [Quaternion(1)]
    nD = len(desiredQ)

    for t in np.arange(self.dt, self.simTime*nD, self.dt):
      qd = desiredQ[int(t // self.simTime)]
      omegaReading = gyro()
      controlTorque = controller(qd, omegaReading)
      actualTorque = reactionwheel(controlTorque)

      gyro.simulateRotation(actualTorque)

      realOrientation.append(calculateCurrentOrientation(realOrientation[-1], gyro.omegaList[-1], gyro.dt))

    return np.array(realOrientation)

if __name__ == '__main__':
  unittest.main()
