import pygame
import numpy as np
from math import *
from Quaternion import quaternionToRotationMatrix
from Plotter import cube
import os

WINDOW_SIZE =  800
ROTATE_SPEED = 0.02
scale = 50
FOLDER='plots'
fps = 30
# pygame.init()

os.makedirs(FOLDER, exist_ok=True)

def drawReferenceAxis(window):
  axes = 10*np.eye(3)
  axes = (axes * scale) + WINDOW_SIZE/2
  center = (np.zeros(3) * scale) + WINDOW_SIZE/2
  for a in axes:
    pygame.draw.aaline(window, (255, 255, 255), center[:2], a[:2])

def plotPyGame(data, angles, framesPerAngle):
  window = pygame.display.set_mode( (WINDOW_SIZE, WINDOW_SIZE) )
  clock = pygame.time.Clock()

  dim, vertices, edges, faces = cube(750, np.array([900,800,600]))

  quivers = 5*np.eye(3)

  for i,point in enumerate(data[::30]):
    # clock.tick(60)
    window.fill((0,0,0))

    drawReferenceAxis(window)

    # Drawing Cuboid
    r = quaternionToRotationMatrix(point)
    rotatedVertices = (r @ vertices.T).T
    rotatedVertices = (rotatedVertices * scale) + WINDOW_SIZE/2

    for p in rotatedVertices:
      pygame.draw.circle(window, (255, 0, 0), p[:2], 5)

    for e in edges:
      pygame.draw.aaline(window, (255, 255, 255), rotatedVertices[e[0], :2], rotatedVertices[e[1], :2])

    pygame.draw.polygon(window, (0, 0, 255), [rotatedVertices[i, :2] for i in faces[0]])


    # drawing quivers
    rotatedQuivers = (r @ quivers.T).T
    rotatedQuivers = (rotatedQuivers * scale) + WINDOW_SIZE/2
    center = (np.zeros(3) * scale) + WINDOW_SIZE/2
    for q in rotatedQuivers:
      pygame.draw.aaline(window, (0, 255, 0), center[:2], q[:2])

    # displaying strings

    # saving images
    pygame.image.save( window, f"plots/screen_{i:08d}.png")

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
        return

    pygame.display.update()

  pygame.quit()

if __name__ == "__main__":
  dataRaw = np.load('realOrientation.npy')
  extraPoints = 500
  data = np.zeros([len(dataRaw) + extraPoints, 4])
  data[:extraPoints] = np.array([1, 0, 0, 0])
  data[extraPoints:] = dataRaw
  plotPyGame(data, np.array([0,0,0], [20, 20, 20]), len(data))
