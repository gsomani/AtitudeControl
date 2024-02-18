import unittest
import numpy as np
from Dynamics import spacecraftDynamics
from PIDController import PIDController
from Quaternion import calculateCurrentOrientation
import Plotter
from SensorAndActuatorModel import *

class TestPIDController(unittest.TestCase):

    def setUp(self):
        self.I = np.diag([900, 800, 600])
        self.torqueInit = np.array([0.0, 0.0, 0.0])  # No External Torque
        self.dt = 0.01
        self.simTime = 500
        self.rwMass = 1 #kg
        self.rwRadius = 0.5 #m

    def test1(self):
        """
        This test will change the attitude to a particular set point
        """

        # Desired quaternion
        # qd = np.array([0.9961947, 0.0871557, 0, 0])
        # qd = np.array([0.9437144, 0.1276794, 0.1448781, 0.2685358])
        desiredQ = np.array([0.4821467, 0.2258942, 0.7239915, -0.4385689 ])

        qInit = np.array([1.0, 0.0, 0.0, 0.0])  # Initial orientation
        omegaInit = np.array([0.0, 0.0, 0.0])

        gyroNoiseStd = 0.0
        gyroscope = Gyroscope(self.I, gyroNoiseStd, omegaInit, self.torqueInit, self.dt)

        Kp = 20 * (self.I.diagonal()/np.max(self.I))
        Ki = 0.1 * (self.I.diagonal()/np.max(self.I))
        Kd = 400 * (self.I.diagonal()/np.max(self.I))
        controller = PIDController(Kp, Ki, Kd, qInit, self.dt)

        rwMaxRPM = 2000
        rwInitialSpeed = np.zeros(3)
        reactionWheel = ReactionWheel(rwMaxRPM, self.rwMass, self.rwRadius, rwInitialSpeed, self.dt)

        realOrientation = self.simulate(desiredQ, qInit, gyroscope, controller, reactionWheel)

        Plotter.plot(np.array(gyroscope.omegaList), realOrientation, title='PID')

    def simulate(self, desiredQ, initialQ, gyro, controller, reactionwheel):
        """
        Simulates spacecraft dynamics and pid control.
        """
        realOrientation = [initialQ]

        for t in np.arange(self.dt, self.simTime, self.dt):
            omegaReading = gyro()
            controlTorque = controller(desiredQ, omegaReading)
            actualTorque = reactionwheel(controlTorque)

            gyro.simulateRotation(actualTorque)

            realOrientation.append(calculateCurrentOrientation(realOrientation[-1], gyro.omegaList[-1], gyro.dt))

        controller.plot()
        return np.array(realOrientation)

if __name__ == '__main__':
    unittest.main()
