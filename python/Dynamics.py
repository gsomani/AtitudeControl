import numpy as np

def spacecraftDynamics(T, omega, I):
  return np.linalg.inv(I) @ (T - np.cross(omega,(I @ omega)))
