import pygame
import numpy as np
from math import *
from Quaternion import Quaternion
from Plotter import cuboid
import os

WINDOW_SIZE =  800
ROTATE_SPEED = 0.02
scale = 50
FOLDER='plots'
fps = 30
DEGREE = u"\u00b0"
# pygame.init()

pygame.font.init()
font = pygame.font.SysFont('Comic Sans MS', 30)

os.makedirs(FOLDER, exist_ok=True)

def plotPyGame(data, angles='', framesPerAngle=''):
  window = pygame.display.set_mode( (WINDOW_SIZE, WINDOW_SIZE) )

  dim, vertices, edges, faces = cuboid(750, np.array([900,800,600]))

  quivers = 6*np.eye(3)
  colours = 255*np.eye(3)

  for i,point in enumerate(data[::30]):
    # clock.tick(60)
    window.fill((0,0,0))

    # Drawing Cuboid
    q = Quaternion(point[0], point[1:])
    r = q.rotationMatrix
    rotatedVertices = (r @ vertices.T).T
    rotatedVertices = (rotatedVertices * scale) + WINDOW_SIZE/2

    for p in rotatedVertices:
      pygame.draw.circle(window, (255, 0, 0), p[:2], 5)

    for e in edges:
      pygame.draw.aaline(window, (255, 255, 255), rotatedVertices[e[0], :2], rotatedVertices[e[1], :2])

    # drawing quivers
    rotatedQuivers = (r @ quivers.T).T
    rotatedQuivers = (rotatedQuivers * scale) + WINDOW_SIZE/2
    center = (np.zeros(3) * scale) + WINDOW_SIZE/2
    for q, colour in zip(rotatedQuivers, colours):
      pygame.draw.aaline(window, colour, center[:2], q[:2])

    # displaying strings
    index = int(i // (framesPerAngle/fps))
    text = font.render(f'Rotating from {angles[index]}{DEGREE} to {angles[index+1]}{DEGREE}', True, (255, 255, 255))
    window.blit(text, (0,0))

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
  plotPyGame(data, np.array([[0,0,0], [20, 20, 20]]), len(data))
