import numpy as np
from Quaternion import calculateCurrentOrientation

def update(q, T, omega, I, dt):
  omegadot = angularAcceleration(T, omega, I)
  omega   += omegadot * dt
  return omega, calculateCurrentOrientation(q, omega, dt)

def angularAcceleration(T, omega, I):
  return np.linalg.inv(I) @ (T - np.cross(omega,(I @ omega)))
