import unittest

import matplotlib.axes._axes as axes
import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np

import models.array_synthesis.draw as dw
from models.array_synthesis.IFTSynthesis import IFTSynthesis as IFT


class IFTTestCase(unittest.TestCase):
    def test_circular_equal_exciation(self):
        sidelobe = 30
        scan = np.array([np.pi / 3, np.pi / 3])
        omega = np.array([3, 3]) / 180 * np.pi
        physical_aperture = 700e-3
        freq = 12e9
        c0 = 3e8
        lam = c0 / freq

        interval = np.array([13.3e-3 / lam, 13.3e-3 / lam], dtype=float)
        aperture = np.array([physical_aperture / lam, physical_aperture / lam], dtype=float)
        theta = np.arange(30, 150, 1)
        phi = np.arange(-60, 60, 1)

        # calculate factor
        sample = IFT(sidelobe, interval, aperture)

        AF = sample.array_factor(theta * np.pi / 180, phi * np.pi / 180)
        AF_abs = np.abs(AF)
        AFnormal = 20 * np.log10(AF_abs / np.max(AF_abs))

        sample.show()

        # Draw pictures
        fig = plt.figure(num=1, figsize=(10, 8), dpi=300)

        # subfigure1

        dw.IFT_MaskArray(radius=sample.aperture[0] / 2, number=sample.numberUV[0], interval=sample.interval[0],
                         ax=fig.add_subplot(221))

        # subfigure2

        ax = fig.add_subplot(222, projection='3d')
        Theta, Phi = np.meshgrid(theta, phi)
        AFnormal[AFnormal < -90] = -90
        surf = ax.plot_surface(Theta, Phi, AFnormal, cmap='coolwarm')
        # ax.contour(Theta, Phi, AFnormal, zdir='z', levels=8, offset=20, cmap="coolwarm")
        picture_title = "(Directivity: " + str(np.round(10 * np.log10(sample.max_gain), 2)) + " dB )"
        ax.set_title(picture_title, family='times new roman', fontsize=15)
        ax.set_ylabel('Theta(degree)', family='times new roman')
        ax.set_xlabel('Phi(degree)', family='times new roman')
        ax.set_zlabel('/dB', family='times new roman')
        ax.set_zlim([-90, 0])

        fig.colorbar(surf, shrink=0.7, pad=0.15)

        # subfigure3
        ax = fig.add_subplot(223)
        y = AF_abs[:, int(AF_abs.shape[1] / 2)]
        y = 20 * np.log10(y / y.max())

        ax.plot(phi, y)

        ax.set_title(r"$\phi=0$", family="times new roman", style='italic')

        ax.set_xticks([-60, -45, -30, -15, -3, 3, 15, 30, 45, 60])
        ax.set_xlabel(r'$\theta$ (degree)', family="times new roman", style='italic', fontsize=15)
        ax.set_xlim([-60, 60])

        ax.set_ylabel('normalized pattern/dB', family="times new roman", fontsize=15)
        ax.set_ylim([-70, 0])
        ax.set_yticks([-70, -60, -50, -40, -30, -20, -10, -3, 0])

        ax.grid(True)

        # subfigure4
        ax = fig.add_subplot(224)
        y = AF_abs[int(len(AF_abs) / 2)]
        y = 20 * np.log10(y / y.max())

        ax.plot(phi, y)

        ax.set_title(r"$\phi=\pi/2$", family="times new roman", style='italic')

        ax.set_xticks([-60, -45, -30, -15, -3, 3, 15, 30, 45, 60])
        ax.set_xlabel(r'$\theta$ (degree)', family="times new roman", style='italic', fontsize=15)
        ax.set_xlim([-60, 60])

        ax.set_ylabel('normalized pattern/dB', family="times new roman", fontsize=15)
        ax.set_ylim([-70, 0])
        ax.set_yticks([-70, -60, -50, -40, -30, -20, -10, -3, 0])

        ax.grid(True)

        # fig.suptitle(str(sample.aperture[0]) + r" $\lambda$ circular aperture array", family="times new roman",fontsize=15)
        fig.suptitle(str(physical_aperture * 1000) + r"(mm) circular aperture array at " + str(freq / 1e9) + " GHz",
                     family="times new roman",
                     fontsize=15)

        plt.show()

    def test_circular_distribution(self):
        physical_aperture = 600
        physical_interval = 13.3
        physical_number = 2 * int(np.ceil(physical_aperture / 2 / physical_interval)) + 1

        fig = plt.figure(num=1, figsize=(6, 6), dpi=200)  # type:figure.Figure

        ax = fig.add_subplot(1, 1, 1)  # type:axes.Axes
        dw.Circular_Distribution(physical_aperture, physical_number, physical_interval, ax)
        plt.show()

        print("Interval : " + "U: " + str(physical_interval) + "\tV: " + str(physical_interval))
        print("ArraySize: " + str(physical_number) + "*" + str(physical_number))
        print("Radius: " + str(physical_aperture) + "(mm)")

    def test_circular_distribution9(self):
        aperture = np.array([500, 600, 700], dtype=np.float64)
        interval = np.array([13.3, 14.5, 16], dtype=np.float64)

        fig = plt.figure(num=1, figsize=(6, 6), dpi=200)

        for zmc_ap in np.arange(aperture.size):
            for zmc_in in np.arange(interval.size):
                fignum = zmc_ap * interval.size + zmc_in + 1
                physical_aperture = aperture[zmc_ap]
                physical_interval = interval[zmc_in]
                physical_number = 2 * int(np.ceil(physical_aperture / 2 / physical_interval)) + 1
                ax = fig.add_subplot(3, 3, fignum)
                dw.Circular_Distribution(physical_aperture, physical_number, physical_interval, ax)

        plt.show()

    def test_power_density(self):
        # %% initial
        rmin = 5
        rmax = 100
        r = np.arange(rmin, rmax, 0.1) * 1000

        deg = 5
        theta = deg * np.pi / 180
        gain_dB = np.arange(30, 41, 0.01)
        gain = 10 ** (gain_dB / 10)
        power = 4.5e3

        area = 0.25 * (r * theta) ** 2 * np.pi
        power_density = np.einsum("i,j->ji", 1. / area, power * gain * 0.5)
        pdB = 10 * np.log10(power_density)
        Xi, Yi = np.meshgrid(r / 1000, gain_dB)

        # %% draw with code hint
        fontstyle = 'times new roman'
        fsize = 12
        fig = plt.figure(num=1, figsize=(6, 6), dpi=200)
        ax = fig.add_subplot(1, 1, 1)
        assert isinstance(fig, figure.Figure)
        assert isinstance(ax, axes.Axes)

        contour = ax.contour(Xi, Yi, pdB, linewidths=1, colors='k')
        surf = ax.contourf(Xi, Yi, pdB, cmap="coolwarm")
        # adjust labels' locations
        manual_locations = [(95, 30.5), (60, 34), (40, 34), (20, 36), (20, 38), (10, 38), (5, 38)]
        labels = ax.clabel(contour, inline=True, fontsize=fsize, manual=manual_locations)

        cbar = fig.colorbar(surf, ax=ax)
        cbar.set_label(r'dBW/m$^2$', fontfamily=fontstyle, fontsize=fsize)

        ax.set_xlim(xmin=0, xmax=100)
        ax.set_xlabel("Distance(km)", fontfamily=fontstyle, fontsize=fsize)
        ax.set_xticks([0, 5, 20, 40, 60, 80, 100])
        ax.set_ylabel("Radiation Gain(dB)", fontfamily=fontstyle, fontsize=fsize)
        ax.set_title("Power Density at " + str(deg) + " degree", fontfamily=fontstyle, fontsize=fsize + 3)

        plt.show()


if __name__ == '__main__':
    unittest.main()
