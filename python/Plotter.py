import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Quaternion import quaternionToRotationMatrix

def plotOmega(omegaHistory, title=''):
    time = range(len(omegaHistory))
    plt.figure()
    plt.plot(time, omegaHistory[:, 0], label='Roll')  # Plot roll rate
    plt.plot(time, omegaHistory[:, 1], label='Pitch')  # Plot pitch rate
    plt.plot(time, omegaHistory[:, 2], label='Yaw')  # Plot yaw rate
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()
    plt.title(title)
    plt.savefig(f'{title}-Omega')
    plt.show()

def plot3D(quatHistory, title=''):
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
      lines.append(ax.plot(*l))

    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-4, 4])


    q1 = np.array([3, 0, 0])
    q2 = np.array([0, 3, 0])
    q3 = np.array([0, 0, 3])
    center = np.zeros(3)
    quivers = []
    for q,c in zip([q1, q2, q3], ['red', 'blue', 'green']):
      quivers.append(ax.quiver(*center, *q, color=c))

    for i in range(0, len(quatHistory), 25):
        rotationMatrix = quaternionToRotationMatrix(quatHistory[i])
        rotatedVertices = rotationMatrix @ vertices.T

        # for q, point in zip(quivers, [q1, q2, q3]):
        #   q.set_segments([list(center), list((rotationMatrix @ point.T).T)])

        for e,l in zip(edges,lines):
            rl = np.array([ rotatedVertices[:,e[i]] for i in range(2) ]).T
            l[0].set_data(*rl[:2])
            l[0].set_3d_properties(rl[2])

        plt.pause(0.1)

    plt.show()
