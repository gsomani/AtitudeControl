import unittest
import numpy as np
from PIDController import PIDController
from Quaternion import calculateCurrentOrientation, eulerToQuaternion, Quaternion
import Plotter
from PlotterPygame import plotPyGame
from SensorAndActuatorModel import Gyroscope, ReactionWheel

class TestPIDController(unittest.TestCase):

  def setUp(self):
      self.I = np.diag([900, 800, 600])
      self.torqueInit = np.array([0.0, 0.0, 0.0])  # No External Torque
      self.dt = 0.01
      self.simTime = 100
      self.rwMass = 1 #kg
      self.rwRadius = 0.5 #m

  def test1(self):
    """
    This test will change the attitude to a particular set point
    """

    # Change these for different set points
    O1 = np.array([60, -30, -10])
    O2 = np.array([30, 30, 30])
    O3 = np.array([140, -80, 210])

    desiredOrientation = np.array([O1, O2, O3])
    desiredQ = np.array([eulerToQuaternion(*o) for o in desiredOrientation])

    qInit = np.array([1.0, 0.0, 0.0, 0.0])  # Initial orientation
    omegaInit = np.array([0.0, 0.0, 0.0])

    gyroNoiseStd = 0.005
    gyroscope = Gyroscope(self.I, gyroNoiseStd, omegaInit, self.torqueInit, self.dt)

    Kp = 20 * (self.I.diagonal()/np.max(self.I))
    Ki = 0.4 * (self.I.diagonal()/np.max(self.I))
    Kd = 400 * (self.I.diagonal()/np.max(self.I))
    controller = PIDController(Kp, Ki, Kd, qInit, self.dt)

    rwMaxRPM = 2000
    rwInitialSpeed = np.zeros(3)
    reactionWheel = ReactionWheel(rwMaxRPM, self.rwMass, self.rwRadius, rwInitialSpeed, self.dt)

    realOrientation = self.simulate(desiredQ, qInit, gyroscope, controller, reactionWheel)

    Plotter.plotOmega(np.array(gyroscope.omegaList), title='Omega of the spacecraft')
    Plotter.plotOmega(np.array(gyroscope.omegaNoisyList), title='GyroScope Data (With Gyro Noise)')
    np.save('realOrientation', realOrientation)
    framesPerAngle = self.simTime/self.dt
    plotPyGame(realOrientation, np.insert(desiredOrientation, 0, np.zeros(3), axis=0), framesPerAngle)
    Plotter.plot3D(realOrientation)

  def simulate(self, desiredQ, initialQ, gyro, controller, reactionwheel):
    """
    Simulates spacecraft dynamics and pid control.
    """
    realOrientation = [Quaternion(initialQ[0],initialQ[1:])]
    nD = len(desiredQ)

    for t in np.arange(self.dt, self.simTime*nD, self.dt):
      qd = desiredQ[int(t // self.simTime)]
      omegaReading = gyro()
      controlTorque = controller(qd, omegaReading)
      actualTorque = reactionwheel(controlTorque)

      gyro.simulateRotation(actualTorque)

      realOrientation.append(calculateCurrentOrientation(realOrientation[-1], gyro.omegaList[-1], gyro.dt))

    controller.plot()
    return np.array(realOrientation)

if __name__ == '__main__':
  unittest.main()
