import unittest
import numpy as np
import matplotlib.pyplot as plt
from Dynamics import spacecraftDynamics
from Quaternion import quaternionToRotationMatrix

class TestSpacecraftDynamics(unittest.TestCase):

    def setUp(self):
        self.I = np.diag([900, 800, 600])
        self.qInit = np.array([1.0, 0.0, 0.0, 0.0])  # Initial orientation
        self.omegaInit = np.array([0.0, 0.0, 0.0])
        self.dt = 0.01
        self.simTime = 100

    def tesConstantTorque(self): # In one axis
        T = np.array([2.0, 0.0, 0.0])

        omegaHistory, quatHistory = self.simulate(T)
        self.plot(omegaHistory, quatHistory, 'Constant-Torque')

    def testTwoAxisConstantTorque(self):
        T = np.array([2.0, 2.0, 0.0])
        omegaHistory, quatHistory = self.simulate(T)
        self.plot(omegaHistory, quatHistory, 'Constant-Torque2Axis')


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

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        vertices = np.array([[i,j,k] for i in [-1,1] for j in [-1,1] for k in [-1,1]])
        
        edges = []

        for i,v in enumerate(vertices):
          for j,u in enumerate(vertices[i+1:]):
            if np.linalg.norm(u-v) == 2:
              edges.append([i,i+1+j])

        rotatedVertices = vertices.T
        lines = []
        for e in edges:
          l = np.array([ rotatedVertices[:,e[i]] for i in range(2) ]).T
          lines.append(ax.fill(*l))

        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])

        for i in range(0, len(time),50):
            rotationMatrix = quaternionToRotationMatrix(quatHistory[i])
            rotatedVertices = rotationMatrix @ vertices.T

            for e,l in zip(edges,lines):
                rl = np.array([ rotatedVertices[:,e[i]] for i in range(2) ]).T
                l[0].set_data(*rl[:2])
                l[0].set_3d_properties(rl[2])

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
