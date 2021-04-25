# -*- coding:utf-8 -*-
# Author : ZhengMX
# Data : 2021/4/21 20:29
# Project : ArrarySynthesis
# FileName : IFTSynthesis
# Cooperation : 265

import time
from enum import Enum

import numpy as np
from numpy.fft import ifft2, ifftshift
from progress_bar import InitBar


# shape enumerate
class ArrayShape(Enum):
    circle = 1
    rectangle = 2


class IFTSynthesis:
    def __init__(self, sidelobe, interval, aperture, array_shape=ArrayShape.circle):
        self.sidelobe = sidelobe
        self.array_shape = array_shape
        self.aperture = aperture
        self.numberUV = np.array(np.ceil(self.aperture / interval), dtype=np.int32) + 1
        self.interval = self.aperture / (self.numberUV - 1)

        if self.array_shape == ArrayShape.circle:
            num_half = int(np.ceil(self.aperture[0] / 2 / interval[0]))
            self.numberUV[0] = (2 * num_half + 1)
            self.numberUV[1] = self.numberUV[0]
            self.aperture[1] = self.aperture[0]
            self.interval = self.aperture / (self.numberUV - 1)

        self.__generate_mask__()

    def synthesis(self):
        """
        synthesis
        """
        number_FFT = np.array([self.numFFT, self.numFFT], dtype=np.int0)
        m = 0
        print("执行阵列综合开始，祈祷不报错".center(25, "-"))

        for trial in np.arange(1, self.max_trial + 1):
            msg_1 = "随机生成阵列激励第" + str(trial) + "次"
            # initial current space
            self.current = np.array(self.array_mask, dtype=float)
            # time bar
            pbar = InitBar(title=msg_1)
            for iteration in np.arange(1, self.max_iteration + 1):
                # Project
                m = m + 1
                af_space = ifftshift(ifft2(self.current, number_FFT))
                af_abs = np.abs(af_space)
                max_index = np.unravel_index(af_abs.argmax(), af_abs.shape)
                maxAF = af_abs.max()

                time.sleep(0.1)
                pbar(iteration / self.max_iteration * 100)

            del pbar

        print("执行结束，万幸".center(25, "-"))

    def array_factor(self):
        pass

    def show(self):
        pass

    def wrirte(self, filename: str):
        pass

    def __generate_mask__(self):
        """
        Generate the array mask with 0,1
        :return:
        """

        if self.array_shape == ArrayShape.circle:
            half = int(self.numberUV[0] / 2)
            mask_line = np.abs(np.arange(-half, half + 1, 1)) * self.interval[0]
            temp_mask = np.einsum("i,j->ij", np.ones([self.numberUV[0]], dtype=float), mask_line * mask_line)
            dis_mask = temp_mask + temp_mask.T - self.aperture[0] * self.aperture[0] / 4.

            self.array_mask = np.array(np.where(dis_mask > 0, 0, 1), dtype=np.int8)
            self.current = np.array(self.array_mask, dtype=float)
        else:
            self.array_mask = np.ones(self.numberUV, dtype=np.int8)
            self.current = np.array(self.array_mask, dtype=float)

    # antenna parameter
    array_shape = ArrayShape.circle
    sidelobe = 0

    aperture = np.array([8, 8], dtype=float)
    gradlobeinterval = np.array([0.5, 0.5], dtype=float)
    numberUV = np.array([1, 1], dtype=int)
    interval = np.array([0.5, 0.5], dtype=float)

    direct = 1

    array_mask = None
    current = None
    # constrained
    scan = np.array([90, 90], dtype=float)
    omega = np.array([1, 1], dtype=float)

    # others
    numFFT = 2048
    max_trial = 1
    max_iteration = 100
    null_depth = -165
    excitation_ratio = 20


if __name__ == '__main__':
    sidelobe = 30
    scan = np.array([np.pi / 3, np.pi / 3])
    omega = np.array([3, 3]) / 180 * np.pi
    interval = np.array([0.45, 0.45], dtype=float)
    aperture = np.array([8, 8], dtype=float)

    sample = IFTSynthesis(sidelobe, interval, aperture)
    sample.synthesis()
    pass
