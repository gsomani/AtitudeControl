import numpy as np
from Quaternion import calculateCurrentOrientation, Quaternion

class PIDController(object):
    def __init__(self, kp, ki, kd, dt=0.01):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = np.zeros(3)
        self.prevErrorVector = np.zeros(3)
        self.quatList = [Quaternion(1)]
        self.dt = dt

    def __call__(self, desiredQ, omega):
        currentQ = calculateCurrentOrientation(self.quatList[-1], omega, self.dt)
        self.quatList.append(currentQ)
        qe = desiredQ.inverse * currentQ

        errorVector = qe.v
        errorDot = (errorVector - self.prevErrorVector) /self.dt
        self.integral += errorVector * self.dt

        output = (self.kp * errorVector) + (self.ki * self.integral) + (self.kd * errorDot)

        self.prevErrorVector = errorVector

        return -output
