import numpy as np

class Quaternion:
  def __init__(self, w, v):
    self.w = w
    self.v = np.array(v)

  def __add__(self, q):
    w = self.w + q.w
    v = self.v + q.v
    return Quaternion(w,v)

  def __mul__(self, q):
    if type(q) in [float, int, np.float64] :
      w,v = self.w*q, self.v*q
    else:
      w = self.w*q.w - np.dot(self.v,q.v)
      v = self.w*q.v + q.w*self.v + np.cross(self.v,q.v)
    return Quaternion(w,v)

  def __sub__(self, q):
    return self + (-q)

  def __neg__(self):
    return Quaternion(-self.w, -self.v)

  def __eq__(self, other):
    if not isinstance(other, Quaternion):
        return NotImplemented

    return (other.w == self.w and other.v == self.v)

  def __repr__(self):
    return "%f + %f i + %f j + %f k" %(self.w,*self.v)

  def derivative(self, omega):
    return self*Quaternion(0,omega/2)

  @property
  def rotationMatrix(self):
    R = np.empty([3,3])

    for i in range(3):
      R[i,i] = 2*(self.w*self.w + self.v[i]*self.v[i]) - 1
      j,k = [(i+m) % 3 for m in range(1,3)]
      R[i,j] = 2*(self.v[j]*self.v[i] - self.w*self.v[k])
      R[i,k] = 2*(self.v[k]*self.v[i] + self.w*self.v[j])

    return R

  @property
  def norm(self):
    return np.linalg.norm([self.w, *self.v])

  def normalise(self):
    self = self*(1/self.norm)

  def update(self, omega, dt):
    qdot = self.derivative(omega)
    q = self + qdot*dt
    q.normalise()
    return q

def quaternionToRotationMatrix(qq):
  q = Quaternion(qq[0], np.array(qq[1:]))
  return q.rotationMatrix

def eulerToQuaternion(roll, pitch, yaw):
  """
  Converts ZYX Euler angles (in degrees) to a quaternion.
  """
  r = np.deg2rad(roll)
  p = np.deg2rad(pitch)
  y = np.deg2rad(yaw)

  # Calculate trigonometric components
  c1 = np.cos(r/2)
  c2 = np.cos(p/2)
  c3 = np.cos(y/2)
  s1 = np.sin(r/2)
  s2 = np.sin(p/2)
  s3 = np.sin(y/2)

  # Construct the quaternion (scalar-last convention)
  qw = c1*c2*c3 - s1*s2*s3
  qx = s1*s2*c3 + c1*c2*s3
  qy = s1*c2*c3 + c1*s2*s3
  qz = c1*s2*c3 - s1*c2*s3

  return np.array([qw, qx, qy, qz])

def quaternionToEuler(q):
  """
  Converts a quaternion to ZYX Euler angles (in radians).
  """

  qw, qx, qy, qz = q

  # Avoid singularities near pitch +/- 90 degrees (Gimbal lock)
  test = 2.0 * (qw*qy + qz*qx)
  if test >= 1:
    pitch = np.pi / 2
  elif test <= -1:
    pitch = -np.pi / 2
  else:
    pitch = np.arcsin(test)

  roll = np.arctan2(2.0*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
  yaw = np.arctan2(2.0*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))

  return roll, pitch, yaw

def quaternionInverse(q):
  """Calculates the inverse of a quaternion."""
  norm_squared = np.sum(q**2)
  return np.array([q[0], -q[1], -q[2], -q[3]]) / norm_squared

def quaternionMultiply(q1, q2):
  """Performs quaternion multiplication."""
  q = [ Quaternion(a[0],np.array(a[1:])) for a in [q1,q2]]

  p = q[0]*q[1]

  return np.array([p.w, *p.v])

def calculateCurrentOrientation(previousQ, omega, dt):
  Q = Quaternion(previousQ[0], previousQ[1:])
  q = Q.update(omega, dt)
  return np.array([q.w,*q.v])

def calculateQuatError(desiredQ, currentQ):
  """Calculates error quaternion (qe = desiredQ^-1 * currentQ)."""
  return quaternionMultiply(quaternionInverse(desiredQ), currentQ)

def extractErrorVector(qe):
  """Takes an error quaternion (qe) and extracts the vector part."""
  return qe[1:]
