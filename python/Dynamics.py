import numpy as np

def angularAcceleration(T, omega, I):
  return np.linalg.inv(I) @ (T - np.cross(omega,(I @ omega)))
