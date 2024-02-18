import unittest
import numpy as np
from Dynamics import spacecraftDynamics
from PIDController import PIDController
import Plotter

class TestPIDController(unittest.TestCase):

    def setUp(self):
        self.I = np.diag([900, 800, 600])
        self.qInit = np.array([1.0, 0.0, 0.0, 0.0])  # Initial orientation
        self.omegaInit = np.array([0.0, 0.0, 0.0])
        self.dt = 0.01
        self.simTime = 500

    def test1(self):
        """
        This test will change the attitude to a particular set point
        """

        # Desired quaternion
        # qd = np.array([0.9961947, 0.0871557, 0, 0])
        # qd = np.array([0.9437144, 0.1276794, 0.1448781, 0.2685358])
        qd = np.array([0.4821467, 0.2258942, 0.7239915, -0.4385689 ])

        # PID Controller Gains
        # Kp = np.array([0.02, 0.02, 0.02])
        # Kd = np.array([0.001, 0.001, 0.001])
        # Ki = np.array([0.01, 0.01, 0.01])

        # Kp = 40 * (self.I.diagonal()/np.max(self.I))
        # Ki = 1 * (self.I.diagonal()/np.max(self.I))
        # Kd = 5000 * (self.I.diagonal()/np.max(self.I))

        Kp = 20 * (self.I.diagonal()/np.max(self.I))
        Ki = 0.1 * (self.I.diagonal()/np.max(self.I))
        Kd = 400 * (self.I.diagonal()/np.max(self.I))


        omegaHistory, quatHistory = self.simulate(qd, Kp, Ki, Kd)

        Plotter.plot(omegaHistory, quatHistory, title='PID')

    def simulate(self, qd, Kp, Ki, Kd):
        """
        Simulates spacecraft dynamics and pid control.
        """
        omegaHistory = [self.omegaInit]
        quatHistory = [self.qInit]
        controller = PIDController(Kp, Ki, Kd)

        for t in np.arange(self.dt, self.simTime, self.dt):

            controlTorque = controller(qd, quatHistory[-1], self.dt)
            omegadot, qdot = spacecraftDynamics(controlTorque, quatHistory[-1], omegaHistory[-1], self.I)

            omega = omegaHistory[-1] + omegadot * self.dt
            q = quatHistory[-1] + qdot * self.dt  # (Assume simple Euler integration for now)

            # Normalize quaternion
            q = q / np.linalg.norm(q)

            omegaHistory.append(omega)
            quatHistory.append(q)

        controller.plot()
        return np.array(omegaHistory), np.array(quatHistory)

if __name__ == '__main__':
    unittest.main()
