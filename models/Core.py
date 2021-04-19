# -*- coding:utf-8 -*-
# Author : ZhengMX
# Data : 2021/4/17 16:37
# Project : ArrarySynthesis
# FileName : Core
# Cooperation : 265

import math
from enum import Enum

import numpy as np


class Source(Enum):
    Equiv = 1
    Chebyshev = 2
    Taylor = 3


class ChebyshevPlaneSyn:

    def __init__(self, sidelobe: float, scan: np.ndarray, omega: np.ndarray):
        self.omega = omega
        self.scan = scan
        self.sidelobe = np.abs(sidelobe)

        self.R0 = 10 ** (self.sidelobe / 20.0)
        lobeFTemp = (2 / self.R0) * np.cosh(np.sqrt(np.arccosh(self.R0) * np.arccosh(self.R0) - np.pi * np.pi))
        self.lobeF = 1 + 0.636 * lobeFTemp * lobeFTemp
        self.gradlobeinverval = 1.0 / (1 + np.abs(np.sin(self.scan)))
        degree = self.omega * 180 / np.pi
        factor = 51 * self.lobeF / degree
        numberUV = np.array(2 * np.ceil(factor / self.gradlobeinverval / 2.0), dtype=int)

        self.__adjust_number__(numberUV)

    def __baberier__(self, x0, M, odd=True):
        """
            Babiere function to calculate the current of chebyshev in line
            :param x0: the zero point of main lobe
            :param M: the total number of elements 1-D array
            :return: the current of every element
        """

        if odd:
            current = np.zeros(M + 1, dtype=float)
            for zmc in range(M + 1):
                n = zmc + 1
                temp = 0.
                for q in np.arange(n, M + 2):
                    term = math.factorial(q + M - 2) / (
                            math.factorial(q - n) * math.factorial(q + n - 2) * math.factorial(M - q + 1))
                    coef = x0 ** (2 * q - 2) * (-1) ** (M - q + 1)
                    temp = temp + term * coef
                current[zmc] = (2 * M - 1) * temp
            current[0] = 2 * current[0]
        else:
            current = np.zeros(M, dtype=float)
            for zmc in range(M):
                n = zmc + 1
                temp = 0.
                for q in np.arange(n, M + 1):
                    term = math.factorial(q + M - 2) / (
                            math.factorial(q - n) * math.factorial(q + n - 1) * math.factorial(M - q))
                    coef = x0 ** (2 * q - 1) * (-1) ** (M - q)
                    temp = temp + term * coef
                current[zmc] = (2 * M - 1) * temp

        current = current / np.max(current)

        return current

    def __adjust_number__(self, num):
        self.numberUV = num
        self.oddU = (np.mod(num[0], 2) == 1)
        self.oddV = (np.mod(num[1], 2) == 1)

        degree = self.omega * 180 / np.pi
        self.interval = 51 * self.lobeF / (degree * self.numberUV)

        self.x0 = np.cosh(np.arccosh(self.R0) / (self.numberUV - 1))

        direct = 2 * self.R0 * self.R0 / (1 + (self.R0 * self.R0 - 1) * degree / 51)

        self.direct = np.pi * direct[0] * direct[1]

    def synthesis(self, number: np.ndarray = np.array([1, 1], int)):
        """
        synthesis of Chebyshev
        :param number: the size of new array
        """

        self.numberUV[0] = np.maximum(self.numberUV[0], number[0])
        self.numberUV[1] = np.maximum(self.numberUV[1], number[1])
        self.__adjust_number__(self.numberUV)

        Mu = int(self.numberUV[0] / 2.)
        Mv = int(self.numberUV[1] / 2.)

        current_u = np.reshape(self.__baberier__(self.x0[0], Mu, odd=self.oddU), (Mu, 1))
        current_v = np.reshape(self.__baberier__(self.x0[1], Mv, odd=self.oddV), (1, Mv))

        self.current = np.matmul(current_u, current_v)

    def get_current(self):
        """
        :return: the first quadrant of current of array
        """
        return self.current

    def get_size(self):
        return self.numberUV

    def array_factor(self, theta_scope, phi_scope):
        """
            calculate the S
            :param theta_scope: the scan angle of theta
            :param phi_scope: the scan angle of phi
            :return: the array_factor S
        """
        k = 2 * np.pi
        nu = int(self.numberUV[0] / 2.)
        nv = int(self.numberUV[1] / 2.)
        AF = np.zeros([len(theta_scope), len(phi_scope)], dtype=np.complex128)
        du = self.interval[0]
        dv = self.interval[1]

        temp_nu = np.broadcast_to(np.arange(nu), (nu, nu))
        temp_nv = np.broadcast_to(np.arange(nv), (nv, nv)).T

        for zmcT in np.arange(theta_scope.shape[0]):
            theta = theta_scope[zmcT]
            for zmcP in np.arange(phi_scope.shape[0]):
                phi = phi_scope[zmcP]
                # wave_diff = (temp_nu*du*np.cos(phi)+temp_nv*dv*np.sin(phi))*np.sin(theta)
                # To Draw yz plane
                wave_diff = temp_nu * du * np.cos(theta) + temp_nv * dv * np.sin(phi) * np.sin(theta)
                phase = np.exp(1j * k * wave_diff)
                temp = self.current * phase
                AF[zmcP, zmcT] = np.einsum('ij->', temp)

        return 4 * AF

    def show(self):
        Num = self.get_size()
        print("ArraySize: " + str(Num[0]) + "*" + str(Num[1]))
        print("Sidelobe: " + str(self.sidelobe))
        degreeScan = self.scan * 180 / np.pi
        degreeOmega = self.omega * 180 / np.pi
        print("Scan Angle: " + "Theta: " + str(degreeScan[0]) + "\tPhi: " + str(degreeScan[1]))
        print("BWhalf : " + "Theta: " + str(degreeOmega[0]) + "\tPhi: " + str(degreeOmega[1]))
        print("Interval : " + "U: " + str(self.interval[0]) + "\tV: " + str(self.interval[1]))
        print("Direct: " + str(10 * np.log10(self.direct)) + "dB")

    sidelobe = 0
    scan = None
    omega = None
    interval = None
    x0 = None
    direct = 1
    current = None
    gradlobeinverval = None
    lobeF = 1
    R0 = 0
    numberUV = None
    oddU = True
    oddV = True


if __name__ == '__main__':
    sidelobe = 30
    scan = np.array([np.pi / 3, np.pi / 3])
    omega = np.array([5, 5]) / 180 * np.pi
    number = np.array([70, 70], dtype=int)

    # test = ChebyshevPlaneSyn(sidelobe, scan, omega)
    # info = test.ParseIndex(number)
    # info.show()
