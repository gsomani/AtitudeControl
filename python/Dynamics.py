import numpy as np
from Quaternion import quaternionKinematics

def spacecraftDynamics(T, q, omega, I):
  """
  Calculates angular acceleration based on Euler's equations.

  Args:
      T (array): Control torque vector (roll, pitch, yaw).
      omega (array): Angular velocity vector (roll, pitch, yaw rates).
      I (array): Inertia matrix (assumed diagonal).

  Returns:
      array: Angular acceleration vector.
  """
  qdot = quaternionKinematics(q, omega)

  omegaX = np.array([[0, -omega[2], omega[1]],
                      [omega[2], 0, -omega[0]],
                      [-omega[1], omega[0], 0]])

  omegadot = np.linalg.inv(I) @ (T - omegaX @ (I @ omega))
  return omegadot, qdot
