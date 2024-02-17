import unittest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Dynamics import spacecraftDynamics
from Quaternion import quaternionToRotationMatrix

class TestSpacecraftDynamics(unittest.TestCase):

    def setUp(self):
        self.I = np.diag([900, 800, 600])
        self.qInit = np.array([1.0, 0.0, 0.0, 0.0])  # Initial orientation
        self.omegaInit = np.array([0.1, -0.05, 0.0])
        self.dt = 0.01
        self.simTime = 5.0

    def testZeroTorque(self):
        T = np.array([0.0, 0.0, 0.0])

        omegaHistory, quatHistory = self.simulate(T)
        self.plot(omegaHistory, quatHistory, 'Zero-Torque')

        # Assertions
        # maxDeviation = 0.01  # Adjust tolerance as needed
        # self.assertTrue(np.all(np.abs(omegaInit - omegaHistory[-1]) < maxDeviation))

    def testConstantTorque(self): # In one axis
        T = np.array([2.0, 0.0, 0.0])

        omegaHistory, quatHistory = self.simulate(T)
        self.plot(omegaHistory, quatHistory, 'Constant-Torque')

        # Assertions
        # maxDeviation = 0.01  # Adjust tolerance as needed
        # self.assertTrue(np.all(np.abs(omegaInit - omegaHistory[-1]) < maxDeviation))

    def plot(self, omegaHistory, quatHistory, title=''):
        time = np.arange(0, self.simTime, self.dt)
        plt.figure()
        plt.plot(time, omegaHistory[:, 0], label='Roll')  # Plot roll rate
        plt.plot(time, omegaHistory[:, 1], label='Pitch')  # Plot pitch rate
        plt.plot(time, omegaHistory[:, 2], label='Yaw')  # Plot yaw rate
        plt.xlabel('Time (s)')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.legend()
        plt.title(title)
        plt.savefig(f'{title}-Omega')

        # 3D Orientation Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Define vertices of a simple shape (e.g., a cube)
        vertices = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])

        # Initial plot setup
        rotatedVertices = vertices.T  # Assume no initial rotation for simplicity
        lines = ax.plot(rotatedVertices[0,:], rotatedVertices[1,:], rotatedVertices[2,:])
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])

        # Update vertices at a few simulation instants
        steps = 5
        print(quatHistory)
        for i in range(0, len(time), len(time) // steps):
            rotationMatrix = quaternionToRotationMatrix(quatHistory[i])
            # ... (Calculate rotation matrix based on attitude at timestep i)
            rotatedVertices = rotationMatrix @ vertices.T

            # Update 3D plot
            lines[0].set_data(rotatedVertices[0,:], rotatedVertices[1,:])
            lines[0].set_3d_properties(rotatedVertices[2,:])
            plt.pause(0.1)

        plt.show()

    def simulate(self, T):
        """
        Simulates spacecraft dynamics.
        """
        omegaHistory = [self.omegaInit]
        quatHistory = [self.qInit]

        for t in np.arange(self.dt, self.simTime, self.dt):
            omegadot, qdot = spacecraftDynamics(T, quatHistory[-1], omegaHistory[-1], self.I)

            omega = omegaHistory[-1] + omegadot * self.dt
            q = quatHistory[-1] + qdot * self.dt  # (Assume simple Euler integration for now)

            # Normalize quaternion
            q = q / np.linalg.norm(q)

            omegaHistory.append(omega)
            quatHistory.append(q)

        return np.array(omegaHistory), np.array(quatHistory)


if __name__ == '__main__':
    unittest.main()