import numpy as np

class Quaternion:
  def __init__(self, w, v = np.array([0,0,0])):
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

  @property
  def conj(self):
    return Quaternion(self.w, -self.v)

  @property
  def axis(self):
    return self.v / self.linalg.norm(self.v)

  @property
  def angle(self):
    return 2*np.arctan2(self.linalg.norm(self.v), self.w)

  @property
  def inverse(self):
    return (self.conj)*(1/(self.norm**2))

def fromAxisAngle(angle, axis):
  z = np.exp(1j*np.deg2rad(angle/2))
  axis /= np.linalg.norm(axis)
  return Quaternion(z.real, z.imag*axis)

def eulerToQuaternion(roll, pitch, yaw):
  q = [fromAxisAngle(angle, axis) for angle,axis in zip([roll,pitch,yaw],np.eye(3))]
  return q[0]*q[1]*q[2]

def calculateCurrentOrientation(previousQ, omega, dt):
  return previousQ.update(omega, dt)
