import numpy as np
from Quaternion import quaternionKinematics

def spacecraftDynamics(T, omega, I):
  """
  Calculates angular acceleration based on Euler's equations.

  Returns:
      array: Angular acceleration vector.
  """
  omegaX = np.array([[0, -omega[2], omega[1]],
                      [omega[2], 0, -omega[0]],
                      [-omega[1], omega[0], 0]])

  omegadot = np.linalg.inv(I) @ (T - omegaX @ (I @ omega))
  return omegadot
