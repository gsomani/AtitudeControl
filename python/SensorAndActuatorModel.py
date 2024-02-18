import numpy as np
from Dynamics import spacecraftDynamics


class Gyroscope(object):
    def __init__(self, I, noiseStd=0.01, initialOmega=np.zeros(3), initialTorque=0, dt=0.01): # holdCount=80, initialShift=10, dropCount=8):
        self.I = I
        self.initialOmega = initialOmega
        self.torque = initialTorque
        self.noiseStd=noiseStd
        self.omegaList = [self.initialOmega]
        self.dt = dt

    def simulateRotation(self, torque):
        self.torque = torque

    def __call__(self):
        omegadot = spacecraftDynamics(self.torque, self.omegaList[-1], self.I)

        omega = self.omegaList[-1] + omegadot * self.dt
        omegaNoisy = omega + np.random.randn(3) * self.noiseStd
        self.omegaList.append(omega)
        return omegaNoisy


class ReactionWheel(object):
    def __init__(self, maxRPM, mass, radius, initialSpeed=np.zeros(3), dt=0.01):
        self.maxSpeed = (maxRPM*2*np.pi) / 60
        self.Irw = 0.5 * mass * (radius ** 2)
        self.currentSpeed = initialSpeed
        self.dt = dt

    def __call__(self, requestedTorque):
        c = self.currentSpeed + ((requestedTorque / self.Irw) * self.dt)
        previousSpeed = self.currentSpeed
        self.currentSpeed = np.clip(c, -self.maxSpeed, self.maxSpeed)
        return ((self.currentSpeed - previousSpeed) / self.dt) * self.Irw

