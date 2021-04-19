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
        self.__parse__()

    def __baberier__(self, x0, N, odd=True):
        """
            Babiere function to calculate the current of chebyshev in line
            :param x0: the zero point of main lobe
            :param N: the total number of elements 1-D array
            :return: the current of every element
        """
        current = np.zeros(N, dtype=float)
        if odd:
            N = N - 1
            for zmc in range(N + 1):
                n = zmc + 1
                temp = 0.
                for q in np.arange(n, N + 2):
                    term = math.factorial(q + N - 2) / (
                            math.factorial(q - n) * math.factorial(q + n - 2) * math.factorial(N - q + 1))
                    coef = x0 ** (2 * q - 2) * (-1) ** (N - q + 1)
                    temp = temp + term * coef
                current[zmc] = (2 * N - 1) * temp
            current[0] = 2 * current[0]
        else:
            for zmc in range(N):
                n = zmc + 1
                temp = 0.
                for q in np.arange(n, N + 1):
                    term = math.factorial(q + N - 2) / (
                            math.factorial(q - n) * math.factorial(q + n - 1) * math.factorial(N - q))
                    coef = x0 ** (2 * q - 1) * (-1) ** (N - q)
                    temp = temp + term * coef
                current[zmc] = (2 * N - 1) * temp

        current = current / np.max(current)

        return current

    def __parse__(self):
        """
        parse the index
        :return:
        """
        self.R0 = 10 ** (self.sidelobe / 20.0)

        lobeFTemp = (2 / self.R0) * np.cosh(np.sqrt(np.arccosh(self.R0) * np.arccosh(self.R0) - np.pi * np.pi))
        self.lobeF = 1 + 0.636 * lobeFTemp * lobeFTemp

        self.gradlobeinverval = 1.0 / (1 + np.abs(np.sin(self.scan)))

        degree = self.omega * 180 / np.pi
        factor = 51 * self.lobeF / degree

        self.numberUV = np.array(2 * np.ceil(factor / self.gradlobeinverval / 2.0), dtype=int)

        self.interval = factor / self.numberUV

        self.x0 = np.cosh(np.arccosh(self.R0) / (self.numberUV - 1))

        self.direct = np.pi * 4 * 51 * 51 / (degree[0] * degree[1])

    def syntheis(self, number: np.ndarray = np.array([1, 1], int)):

        if self.numberUV[0] * 2 - 1 < number[0]:
            self.oddU = (np.mod(number[0], 2) == 1)
            self.numberUV[0] = int(number[0] / 2.)

        if self.numberUV[1] * 2 - 1 < number[1]:
            self.oddV = (np.mod(number[1], 2) == 1)
            self.numberUV[1] = int(number[1] / 2.)

        Nu = self.numberUV[0]
        Nv = self.numberUV[1]

        self.interval = 51 * self.lobeF * np.pi / (self.omega * 180 * self.numberUV)

        self.x0 = np.cosh(np.arccosh(self.R0) / (self.numberUV - 1))

        current_u = np.reshape(self.__baberier__(self.x0[0], Nu, odd=self.oddU), (Nu, 1))
        current_v = np.reshape(self.__baberier__(self.x0[1], Nv, odd=self.oddV), (1, Nv))

        self.current = np.matmul(current_u, current_v)

    def get_current(self):
        return self.current

    def get_size(self):
        res = 2 * self.numberUV
        if self.oddU:
            res[0] = res[0] - 1
        if self.oddV:
            res[1] = res[1] - 1

        return res

    def array_factor(self, theta_scope, phi_scope):
        """
            calculate the S
            :param current: the current of all elements
            :param theta_scope: the scan angle of theta
            :param phi_scope: the scan angle of phi
            :return: the array_factor S
        """
        k = 2 * np.pi
        nu = self.numberUV[0]
        nv = self.numberUV[1]
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
