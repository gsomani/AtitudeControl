import numpy as np
from Quaternion import calculateQuatError, extractErrorVector, calculateCurrentOrientation
import matplotlib.pyplot as plt


class PIDController(object):
    def __init__(self, kp, ki, kd, initialOrientation, dt=0.01): # holdCount=80, initialShift=10, dropCount=8):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = np.zeros(3)
        self.prevErrorVector = np.zeros(3)
        self.errorVectorList = []
        self.quatList = [initialOrientation]
        self.dt = dt
        # self.n = 0
        # self.m = 0
        # self.holdCount = holdCount
        # self.factor = (1 << initialShift)
        # self.dropCount = dropCount


    def __call__(self, desiredQ, omega):
        currentQ = calculateCurrentOrientation(self.quatList[-1], omega, self.dt)
        self.quatList.append(currentQ)
        qe = calculateQuatError(desiredQ, currentQ)

        errorVector = extractErrorVector(qe)
        errorDot = (errorVector - self.prevErrorVector) /self.dt
        self.integral += errorVector * self.dt
        # f = self.factor if self.factor >= 1 else 1
        # self.integral += error*f

        output = (self.kp * errorVector) + (self.ki * self.integral) + (self.kd * errorDot)

        self.errorVectorList.append(errorVector)
        self.prevErrorVector = errorVector

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
        plt.title('PID Error')
        plt.ylabel('Quaternion Error Component')
        plt.xlabel('Time (s)')
        plt.legend()
        plt.savefig(f'PID_Error')
