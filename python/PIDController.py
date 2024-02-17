import numpy as np
from Quaternion import calculateQuatError, extractErrorVector


class PIDController(object):
    def __init__(self, kp, ki, kd): # holdCount=80, initialShift=10, dropCount=8):
        self.kp, self.ki, self.kd = kp, ki, kd  # self.calculate_gains(loopBandwidth, damping)
        self.integral = 0
        # self.n = 0
        # self.m = 0
        # self.holdCount = holdCount
        # self.factor = (1 << initialShift)
        # self.dropCount = dropCount

    def __call__(self, desiredQ, currentQ, omega):
        # f = self.factor if self.factor >= 1 else 1
        # self.integral += error*f
        qe = calculateQuatError(desiredQ, currentQ)
        errorVector = extractErrorVector(qe)

        output = (self.kp * errorVector) + (self.ki * self.integral) + (self.kd * omega)

        # self.n += 1
        # if self.n == self.holdCount and self.dropCount > 0:
        #     self.factor //= 2
        #     self.n = 0
        #     self.dropCount -= 1

        return -output
