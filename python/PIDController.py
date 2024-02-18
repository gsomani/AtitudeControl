import numpy as np
from Quaternion import calculateQuatError, extractErrorVector
import matplotlib.pyplot as plt


class PIDController(object):
    def __init__(self, kp, ki, kd): # holdCount=80, initialShift=10, dropCount=8):
        self.kp, self.ki, self.kd = kp, ki, kd  # self.calculate_gains(loopBandwidth, damping)
        self.integral = np.zeros(3)
        self.prevErrorVector = np.zeros(3)
        self.errorVectorList = []
        # self.n = 0
        # self.m = 0
        # self.holdCount = holdCount
        # self.factor = (1 << initialShift)
        # self.dropCount = dropCount

    def __call__(self, desiredQ, currentQ, dt):
        # f = self.factor if self.factor >= 1 else 1
        # self.integral += error*f
        qe = calculateQuatError(desiredQ, currentQ)
        # print(f'{desiredQ}, {currentQ}')
        errorVector = extractErrorVector(qe)
        errorDot = (errorVector - self.prevErrorVector) /dt
        self.integral += errorVector * dt
        self.prevErrorVector = errorVector

        output = (self.kp * errorVector) + (self.ki * self.integral) + (self.kd * errorDot)
        self.errorVectorList.append(errorVector)

        # self.n += 1
        # if self.n == self.holdCount and self.dropCount > 0:
        #     self.factor //= 2
        #     self.n = 0
        #     self.dropCount -= 1

        return -output

    def plot(self):
        self.errorVectorList = np.array(self.errorVectorList)
        plt.plot(self.errorVectorList[:, 0], label='Roll')  # Plot roll rate
        plt.plot(self.errorVectorList[:, 1], label='Pitch')  # Plot pitch rate
        plt.plot(self.errorVectorList[:, 2], label='Yaw')  # Plot yaw rate
        plt.title('error')
