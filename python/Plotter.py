def plot(time, omegaHistory, quatHistory, title=''):
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

    print(quatHistory)
    for i in range(0, len(time),50):
        rotationMatrix = quaternionToRotationMatrix(quatHistory[i])
        rotatedVertices = rotationMatrix @ vertices.T

        for e,l in zip(edges,lines):
            rl = np.array([ rotatedVertices[:,e[i]] for i in range(2) ]).T
            l[0].set_data(*rl[:2])
            l[0].set_3d_properties(rl[2])

        plt.pause(0.1)

    plt.show()
