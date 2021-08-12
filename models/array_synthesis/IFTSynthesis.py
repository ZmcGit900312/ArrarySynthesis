# -*- coding:utf-8 -*-
# Author : ZhengMX
# Data : 2021/4/21 20:29
# Project : ArrarySynthesis
# FileName : IFTSynthesis
# Cooperation : 265

from enum import Enum

import matplotlib.axes._axes as axes
import matplotlib.figure as figure
import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import ifft2, ifftshift, fft2
from progress_bar import InitBar
from scipy import integrate as si


# shape enumerate
class ArrayShape(Enum):
    circle = 1
    rectangle = 2


class IFTSynthesis:
    def __init__(self, sidelobe, interval, aperture, array_shape=ArrayShape.circle):
        self.sidelobe = -np.abs(sidelobe)
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
        INF = np.inf * np.ones([self.numFFT, 1])
        min_current = 10 ** (-np.abs(self.excitation_ratio) / 20.)
        m = 0
        print("执行阵列综合开始，祈祷不报错".center(25, "-"))

        for trial in np.arange(1, self.max_trial + 1):
            msg_1 = "随机生成阵列激励第" + str(trial) + "次"
            # initial current space
            self.current = np.random.random(size=self.numberUV) * self.array_mask
            # time bar
            pbar = InitBar(title=msg_1)
            for iteration in np.arange(1, self.max_iteration + 1):
                # Normalized the AF
                af_space = ifftshift(ifft2(self.current, number_FFT))
                af_abs = np.abs(af_space)
                max_ind = np.unravel_index(af_abs.argmax(), af_abs.shape)
                maxAF = af_abs.max()
                af_abs = af_abs / maxAF
                af_space = af_space / maxAF

                # Find all FF nulls
                min_val = np.sign(np.diff(np.hstack([INF, af_abs, INF])))
                min_ind = np.where(np.diff(min_val + (min_val == 0)) == 2)

                # Find all FF peaks
                peak_ind = np.diff(min_val) < 0
                indP = np.argsort(af_abs[peak_ind])[::-1]

                # Find indices all SLL directions

                # Adapt AF to SLL constrains

                current = fft2(ifftshift(af_space))
                # Truncate current
                self.current = abs(current[0:self.numberUV[0], 0:self.numberUV[1]]) * self.array_mask
                self.current = self.current / np.max(self.current)
                select = self.current > 0
                self.current[self.current(select) < min_current] = min_current

                pbar(iteration / self.max_iteration * 100)

            del pbar

        print("执行结束，万幸".center(25, "-"))

    def array_factor(self, theta_scope, phi_scope):
        k = 2 * np.pi
        du = self.interval[0]
        dv = self.interval[1]

        zmc = np.arange(0, self.current.shape[0])
        local_current = np.copy(self.current)

        u = 0.5 * k * du * np.cos(theta_scope)
        v = 0.5 * k * dv * np.einsum("i,j->ij", np.sin(phi_scope), np.sin(theta_scope))

        cos_mu = np.cos(np.einsum("i,j->ij", u, zmc))
        cos_mv = np.cos(np.einsum("ij,k->ijk", v, zmc))
        AF = np.einsum("ptn,tm,mn->pt", cos_mv, cos_mu, local_current)

        power, error = si.dblquad(self.__direct__, 0, np.pi / 2, 0, 2 * np.pi)

        mainAF = np.max(np.abs(AF))

        self.max_gain = 4 * np.pi * mainAF * mainAF / power
        return AF

    def show(self):
        print("Interval : " + "U: " + str(np.round(self.interval[0], 2)) + "\tV: " + str(np.round(self.interval[1], 2)))
        print("ArraySize: " + str(self.numberUV[0]) + "*" + str(self.numberUV[1]))
        print("Radius: " + str(self.aperture[0]) + " lambda")
        print("Gain: " + str(10 * np.log10(self.max_gain)) + " dB")
        print("Total Number: " + str(self.total_number))

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

            self.array_mask = np.array(np.where(dis_mask > 0, 0, 1), dtype=np.int0)
            self.total_number = np.einsum("ij->", self.array_mask)
            self.current = np.array(self.array_mask, dtype=float)
        else:
            self.array_mask = np.ones(self.numberUV, dtype=np.int8)
            self.current = np.array(self.array_mask, dtype=float)

    def __direct__(self, phi, theta):
        """
        \int_{0}^{2\pi}{d\phi} \int_{0}^{\pi/2}{d\theta} |S(\theta, \phi)|^2 \sin\theta
        :param phi:
        :param theta:
        :return: denomator of D
        """

        k = 2 * np.pi
        du = self.interval[0]
        dv = self.interval[1]

        zmc = np.arange(0, self.current.shape[0])

        u = 0.5 * k * du * np.sin(theta) * np.cos(phi)
        v = 0.5 * k * dv * np.sin(theta) * np.sin(phi)
        cos_mu = np.cos(u * zmc)
        cos_mv = np.cos(v * zmc)

        temp_res = np.einsum("ij,i,j->", self.current, cos_mu, cos_mv)
        return np.sin(theta) * temp_res * temp_res

    # antenna parameter
    array_shape = ArrayShape.circle
    sidelobe = 0

    aperture = np.array([8, 8], dtype=float)
    gradlobeinterval = np.array([0.5, 0.5], dtype=float)
    numberUV = np.array([1, 1], dtype=int)
    interval = np.array([0.5, 0.5], dtype=float)
    max_gain = 1
    total_number = 1

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


class CircularArray:
    def __init__(self, aperture: np.float64, number: np.array, radius: np.array, centred=True):
        self.set(aperture, number, radius, centred)

    def set(self, ap: np.float64, num: np.array, radius: np.array, centre: bool):
        self._aperture = ap
        self._numbers = num
        self._radius = radius
        self._loops = len(num)
        self._centring = centre
        self.__distribution__()

    def __distribution__(self):
        total_number = self.get_total_number()
        loop_index = 1 if self._centring else 0

        detal_phi = 2 * np.pi / self._numbers
        self._locations = np.zeros([2, total_number], dtype=np.float64)

        message = "Begin to distribute circular array \n" + "Total number:\t" + str(total_number) + "\nLoops:\t" + str(
            self._loops)
        print(message)

        for zmc, angle in enumerate(detal_phi):
            phi = np.arange(0, 2 * np.pi, angle)
            tail = loop_index + len(phi)
            self._locations[0, loop_index:tail] = self._radius[zmc] * np.cos(phi)
            self._locations[1, loop_index:tail] = self._radius[zmc] * np.sin(phi)
            loop_index = tail

    def get_total_number(self):
        return np.einsum("i->", self._numbers) + (1 if self._centring else 0)

    def synthesis(self):
        self._current = np.ones([self.get_total_number()], dtype=np.float64)

    def array_factor(self, frequency, theta_scope, phi_scope):
        omega = 2 * np.pi * frequency
        temp = omega / 3.0e8

        local_current = np.copy(self._current)

        cos_mu = np.cos(0.5 * temp * np.einsum("l,t->lt", self._locations[0], np.cos(theta_scope)))
        cos_mv = np.cos(
            0.5 * temp * np.einsum("l,p,t->lpt", self._locations[1], np.sin(phi_scope), np.sin(theta_scope)))

        AF = np.einsum("lpt,lt,l->pt", cos_mv, cos_mu, local_current)
        return AF

    def __direct__(self, phi, theta):
        """
        \int_{0}^{2\pi}{d\phi} \int_{0}^{\pi/2}{d\theta} |S(\theta, \phi)|^2 \sin\theta
        :param phi:
        :param theta:
        :return: denomator of D
        """

        k = 2 * np.pi

        zmc = np.arange(0, self.current.shape[0])

        u = 0.5 * k * du * np.sin(theta) * np.cos(phi)
        v = 0.5 * k * dv * np.sin(theta) * np.sin(phi)
        cos_mu = np.cos(u * zmc)
        cos_mv = np.cos(v * zmc)

        temp_res = np.einsum("ij,i,j->", self.current, cos_mu, cos_mv)
        return np.sin(theta) * temp_res * temp_res

    def show_mask(self, ax: axes.Axes, title="", fontfamily="times new roman", fontsize=15, scatter=20):

        phi = np.linspace(0, 2 * np.pi, 360 * 4)
        circle = ax.fill(self._aperture * np.cos(phi), self._aperture * np.sin(phi), color="lightblue", zorder=0)
        elements = ax.scatter(x=self._locations[0], y=self._locations[1], s=scatter, c="gray", zorder=2)

        if title == "":
            title = "Total number: " + str(self.get_total_number())

        # loop line
        for radius in self._radius:
            ax.plot(radius * np.cos(phi), radius * np.sin(phi), color="orange", zorder=1)

        ax.set_aspect("equal", 'box')
        ax.set_axis_off()
        ax.grid(True)
        ax.set_title(title, fontsize=fontsize, fontfamily=fontfamily)
        return ax

    _aperture = 1
    _numbers = np.array([1], dtype=np.int64)
    _radius = np.array([1], dtype=np.float64)
    _loops = 1
    _centring = False
    _locations = np.array([1])
    _current = np.array([1], dtype=np.float64)
    _directivity = 0


if __name__ == '__main__':
    loop = 21
    loop_number = np.arange(1, loop) * 8
    loop_radius = np.arange(1, loop) * 12e-3

    sample = CircularArray(aperture=np.float64(.25), number=loop_number, radius=loop_radius, centred=False)
    sample.synthesis()

    total_number = sample.get_total_number()

    # array factor
    theta = np.arange(30, 150, 1)
    phi = np.arange(-60, 60, 1)

    AF = sample.array_factor(12e9, theta * np.pi / 180, phi * np.pi / 180)
    AF_abs = np.abs(AF)
    AFnormal = 20 * np.log10(AF_abs / np.max(AF_abs))

    # draw1
    fig = plt.figure(num=1, figsize=(8, 8), dpi=300)  # type:figure.Figure
    ax = fig.add_subplot(111)
    ax = sample.show_mask(ax)

    # draw2
    fig2 = plt.figure(num=2, figsize=(8, 8), dpi=300)  # type:figure.Figure
    ax = fig2.add_subplot(111, projection="3d")
    Theta, Phi = np.meshgrid(theta, phi)
    AFnormal[AFnormal < -90] = -90
    surf = ax.plot_surface(Theta, Phi, AFnormal, cmap='coolwarm')
    # ax.contour(Theta, Phi, AFnormal, zdir='z', levels=8, offset=20, cmap="coolwarm")

    ax.set_ylabel('Theta(degree)', family='times new roman')
    ax.set_xlabel('Phi(degree)', family='times new roman')
    ax.set_zlabel('/dB', family='times new roman')
    ax.set_zlim([-90, 0])

    fig2.colorbar(surf, shrink=0.7, pad=0.15)
    plt.show()
