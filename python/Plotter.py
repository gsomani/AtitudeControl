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
    plt.savefig(f'{title}')
    plt.show()

def shapeFromInertia(mass, I):
    return (6/mass)*(sum(I) - 2*I)

def gray(n, width):
    g = n ^ (n >> 1)
    return [(g >> i) & 1 for i in range(width)]

def hypercubeTraversal(n):
    return [gray(i, n) for i in range(1 << n)]

def hyperCuboidTraversal(box):
    return np.array([[box[i][c] for i, c in enumerate(b)] for b in hypercubeTraversal(len(box))])

def cuboid(mass, I):
    dim = shapeFromInertia(mass, I)

    cubeVertices = hypercubeTraversal(3)
    vertices = hyperCuboidTraversal([[-d/2,d/2] for d in dim])

    edges = [ [i,(i+1)%8] for i in range(8)]
    edges += [[0, 3], [1, 6], [2, 5], [4, 7]]

    faces = [[[] for i in range(2)] for j in range(3)]

    for i,v in enumerate(cubeVertices):
      for j in range(3):
        faces[j][v[j]].append(i)

    return dim, vertices, edges, faces

def plot3D(quatHistory, title=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    dim, vertices, edges, faces = cuboid(750, np.array([900,800,600]))

    rotatedVertices = vertices.T
    lines = []

    for e in edges:
      l = np.array([ rotatedVertices[:,e[i]] for i in range(2) ]).T
      lines.append(ax.plot(*l))

    d = np.linalg.norm(dim)

    for a in 'xyz':
      getattr(ax, f'set_{a}lim')([-d/2, d/2])

    ax.set_box_aspect([1,1,1])

    qv = 3*np.eye(3)
    center = np.zeros(3)
    quivers = []

    for q,c in zip(qv, ['red', 'blue', 'green']):
      ax.quiver(*center, *q, color=c)

    for q,c in zip(qv, ['red', 'blue', 'green']):
      quivers.append(ax.quiver(*center, *q, color=c))

    for i in range(0, len(quatHistory), 25):
      rotationMatrix = quaternionToRotationMatrix(quatHistory[i])
      rotatedVertices = rotationMatrix @ vertices.T

      for q, point in zip(quivers, qv):
         q.set_segments([[center, (rotationMatrix @ point.T)]])

      for e,l in zip(edges,lines):
          rl = np.array([ rotatedVertices[:,e[i]] for i in range(2) ]).T
          l[0].set_data(*rl[:2])
          l[0].set_3d_properties(rl[2])

      plt.pause(0.1)

    plt.show()
