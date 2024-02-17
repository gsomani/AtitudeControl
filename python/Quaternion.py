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

def quaternionInverse(q):
  """Calculates the inverse of a quaternion."""
  norm_squared = np.sum(q**2)
  return np.array([q[0], -q[1], -q[2], -q[3]]) / norm_squared

def quaternionMultiply(q1, q2):
  """Performs quaternion multiplication."""
  w1, x1, y1, z1 = q1
  w2, x2, y2, z2 = q2

  w = w1*w2 - x1*x2 - y1*y2 - z1*z2
  x = w1*x2 + x1*w2 + y1*z2 - z1*y2
  y = w1*y2 + y1*w2 + z1*x2 - x1*z2
  z = w1*z2 + z1*w2 + x1*y2 - y1*x2

  return np.array([w, x, y, z])

def calculateQuatError(desiredQ, currentQ):
  """Calculates error quaternion (qe = desiredQ^-1 * currentQ)."""
  return quaternionMultiply(quaternionInverse(desiredQ), currentQ)

def extractErrorVector(qe):
    """Takes an error quaternion (qe) and extracts the vector part."""
    return qe[1:]