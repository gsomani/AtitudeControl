import numpy as np
from Quaternion import calculateCurrentOrientation, Quaternion
import matplotlib.pyplot as plt

class PIDController(object):
    def __init__(self, kp, ki, kd, initialOrientation, dt=0.01):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = np.zeros(3)
        self.prevErrorVector = np.zeros(3)
        self.errorVectorList = []
        self.quatList = [Quaternion(initialOrientation[0],initialOrientation[1:])]
        self.dt = dt

    def __call__(self, desiredQ, omega):
        currentQ = calculateCurrentOrientation(self.quatList[-1], omega, self.dt)
        self.quatList.append(currentQ)
        qe = desiredQ.inverse * currentQ

        errorVector = qe.v
        errorDot = (errorVector - self.prevErrorVector) /self.dt
        self.integral += errorVector * self.dt

        output = (self.kp * errorVector) + (self.ki * self.integral) + (self.kd * errorDot)

        self.errorVectorList.append(errorVector)
        self.prevErrorVector = errorVector

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
        plt.savefig('PID_Error')
