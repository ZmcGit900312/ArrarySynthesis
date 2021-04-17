# -*- coding:utf-8 -*-
# Author : ZhengMX
# Data : 2021/4/17 16:37
# Project : ArrarySynthesis
# FileName : Core
# Cooperation : 265

from enum import Enum

import numpy as np

from models.ArrayMask import Mask


class Source(Enum):
    Equiv = 1
    Chebyshev = 2
    Taylor = 3


class ArrayIndex:
    def __init__(self, source: Source):
        self.sour = source

    def show(self):
        print("ArraySize: " + str(self.distribution.Nu) + "*" + str(self.distribution.Nu))
        print("Sidelobe: " + str(self.sidelobe))
        degreeScan = self.scan * 180 / np.pi
        degreeOmega = self.omega * 180 / np.pi
        print("Scan Angle: " + "Theta: " + str(degreeScan[0]) + "\tPhi: " + str(degreeScan[1]))
        print("BWhalf : " + "Theta: " + str(degreeOmega[0]) + "\tPhi: " + str(degreeOmega[1]))
        print("Interval : " + "U: " + str(self.interval[0]) + "\tV: " + str(self.interval[1]))
        print("Direct: " + str(self.direct) + "dB")

    distribution = None
    interval = None
    sidelobe = -13.5
    direct = 1
    gain = 1
    omega = None
    sour = Source.Chebyshev
    gradlobe = [0.5, 0.5]
    BW0 = 0
    scan = [np.pi / 4, np.pi / 3]


class ChebyshevPlaneSyn:

    def __init__(self, sidelobe: float, scan: np.ndarray, omega: np.ndarray):
        self.omega = omega
        self.scan = scan
        self.sidelobe = np.abs(sidelobe)

    def ParseIndex(self, numberUV: np.ndarray = np.array([1, 1], dtype=int)):

        arrayInfo = ArrayIndex(Source.Chebyshev)

        R0 = 10 ** (self.sidelobe / 20.0)

        lobeFTemp = (2 / R0) * np.cosh(np.sqrt(np.arccosh(R0) * np.arccosh(R0) - np.pi * np.pi))
        lobeF = 1 + 0.636 * lobeFTemp * lobeFTemp

        gradlobe = 1.0 / (1 + np.abs(np.sin(scan)))

        degree = omega * 180 / np.pi
        factor = 51 * lobeF / degree

        number = np.array(2 * np.ceil(factor / gradlobe / 2.0), dtype=int)

        if number[0] > numberUV[0]:
            numberUV[0] = number[0]

        if number[1] > numberUV[1]:
            numberUV[1] = number[1]
        interval = factor / numberUV

        direct = np.pi * 4 * 51 * 51 / (degree[0] * degree[1])

        arrayInfo.sidelobe = self.sidelobe
        arrayInfo.scan = self.scan
        arrayInfo.omega = self.omega
        arrayInfo.gradlobe = gradlobe
        arrayInfo.interval = interval
        arrayInfo.direct = 10 * np.log10(direct)
        arrayInfo.distribution = Mask(numberUV[0], numberUV[1])
        return arrayInfo

    sidelobe = 0
    scan = None
    omega = None


if __name__ == '__main__':
    sidelobe = 30
    scan = np.array([np.pi / 3, np.pi / 3])
    omega = np.array([5, 5]) / 180 * np.pi
    number = np.array([40, 40], dtype=int)

    test = ChebyshevPlaneSyn(sidelobe, scan, omega)
    info = test.ParseIndex(number)
    info.show()
