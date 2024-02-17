import numpy as np

def quaternionToRotationMatrix(q):
  """
  Calculates rotation matrix from a quaternion.
  """
  qw, qx, qy, qz = q

  R = np.array([
      [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
      [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
      [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
  ])
  return R

def quaternionKinematics(q, omega):
  """
  Calculates quaternion derivative (qdot) based on angular velocity.
  """
  qw, qx, qy, qz = q
  wx, wy, wz = omega

  qdot = np.zeros_like(q)  # Initialize qdot array
  qdot[0] = 0.5 * (-qx * wx - qy * wy - qz * wz)  # Scalar part
  qdot[1] = 0.5 * (qw * wx + qy * wz - qz * wy)  # Vector part
  qdot[2] = 0.5 * (qw * wy - qx * wz + qz * wx)
  qdot[3] = 0.5 * (qw * wz + qx * wy - qy * wx)

  return qdot