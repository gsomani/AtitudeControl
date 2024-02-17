import unittest
import numpy as np
from Dynamics import spacecraftDynamics
from PIDController import PIDController

class TestPIDController(unittest.TestCase):

    def setUp(self):
        self.I = np.diag([900, 800, 600])
        self.qInit = np.array([1.0, 0.0, 0.0, 0.0])  # Initial orientation
        self.omegaInit = np.array([0.0, 0.0, 0.0])
        self.dt = 0.01
        self.simTime = 100

    def test1(self):
        """
        This test will change the attitude to a particular set point
        """

        # Desired quaternion
        qd = np.array([0.8, 0.0, 0.3, 0.5])

        # PID Controller Gains
        Kp = np.array([0.5, 0.5, 0.5])
        Kd = np.array([0.2, 0.2, 0.2])
        Ki = np.array([0.0, 0.0, 0.0])


    def simulate(self, qd, Kp, Ki, Kd):
        """
        Simulates spacecraft dynamics and pid control.
        """
        omegaHistory = [self.omegaInit]
        quatHistory = [self.qInit]
        controller = PIDController(Kp, Ki, Kd)

        for t in np.arange(self.dt, self.simTime, self.dt):

            controlTorque = controller(qd, quatHistory[-1], omegaHistory[-1])
            omegadot, qdot = spacecraftDynamics(controlTorque, quatHistory[-1], omegaHistory[-1], self.I)

            omega = omegaHistory[-1] + omegadot * self.dt
            q = quatHistory[-1] + qdot * self.dt  # (Assume simple Euler integration for now)

            # Normalize quaternion
            q = q / np.linalg.norm(q)

            omegaHistory.append(omega)
            quatHistory.append(q)

        return np.array(omegaHistory), np.array(quatHistory)
