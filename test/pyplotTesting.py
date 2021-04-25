import unittest

import matplotlib.pyplot as plt
import numpy as np


class MatplotlibTestCase(unittest.TestCase):
    def test_simple_example(self):
        fig, axs = plt.subplots(1, 2)
        axs[0].plot([1, 2, 3, 4], [1, 4, 2, 3])

        x = np.linspace(0, 2, 100)

        plt.plot(x, x, label="linear")
        plt.plot(x, x * x, label="quadratic")
        plt.plot(x, x ** 3, label="cubic")
        plt.xlabel("t/s")
        plt.ylabel("distance/m")
        plt.title("Movement")
        plt.legend()

        plt.show()

    def test_multiple_plot(self):
        x1 = np.linspace(0, 5)
        x2 = np.linspace(0, 2)

        y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
        y2 = np.sin(2 * np.pi * x2)

        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle("Two plots in one picture")

        ax1.plot(x1, y1, 'o-')
        ax1.set_ylabel("Damped oscillation")

        ax2.plot(x2, y2, '.-')
        ax2.set_xlabel('times/s')
        ax2.set_ylabel('Undamped')

        # plt.subplot(2, 1, 1)
        # plt.plot(x1, y1, 'o-')
        # plt.title('Tow plots in one picture')
        # plt.ylabel("Damped oscillation")
        #
        # plt.subplot(2, 1, 2)
        # plt.plot(x2, y2, '.-')
        # plt.xlabel('time/s')
        # plt.ylabel("Undamped")

        plt.show()

    def test_simple_picture(self):
        delta = 0.025
        x = y = np.arange(-3., 3., delta)
        X, Y = np.meshgrid(x, y)
        Z1 = np.exp(-X ** 2 - Y ** 2)
        Z2 = np.exp(-(X - -1) ** 2 - (Y - -1) ** 2)
        Z = (Z1 - Z2) * 2

        fig, ax = plt.subplots()
        im = ax.imshow(Z, interpolation='bilinear', cmap='rainbow',
                       origin='lower', extent=[-3, 3, -3, 3],
                       vmax=abs(Z).max(), vmin=-abs(Z).max())

        plt.show()

    def test_polar_plot(self):
        r = np.arange(0, 2, 0.001)
        theta = 2 * np.pi * r

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(theta, r)
        ax.set_rmax(2)
        ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
        ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
        ax.grid(True)

        ax.set_title(r'A line plot on a polar axis with $(\theta, \rho)$')
        plt.show()

    def test_zoom_plot(self):
        delta = 0.001
        x = y = np.arange(-3., 3., delta)
        extent = [-3, 3, -3, 3]
        X, Y = np.meshgrid(x, y)
        Z1 = np.exp(-X ** 2 - Y ** 2)
        Z2 = np.exp(-(X - -1) ** 2 - (Y - -1) ** 2)
        Z = (Z1 - Z2) * 2

        fig, ax = plt.subplots()

        im = ax.imshow(Z, interpolation='bilinear', cmap='rainbow',
                       origin='lower', extent=extent,
                       vmax=abs(Z).max(), vmin=-abs(Z).max())

        # inset axes
        axins = ax.inset_axes([0.6, 0.6, 0.4, 0.4])  # location and enlarge factor
        axins.imshow(Z, extent=extent, origin='lower', cmap='rainbow')
        # subregion of the original image
        x1, x2, y1, y2 = -0.75, 0.25, -1.25, -0.25
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticklabels('')
        axins.set_yticklabels('')

        ax.indicate_inset_zoom(axins, edgecolor="black")

        plt.show()


if __name__ == '__main__':
    unittest.main()
