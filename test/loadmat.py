import unittest

import matplotlib.pyplot as plt
from scipy.io import loadmat

import models.array_synthesis.draw as dw


class LoadMatTestCase(unittest.TestCase):
    def test_loadmat(self):
        filename = './number25.mat'
        data = loadmat(filename)

        number = data['number_diameter'].flatten()[0]
        radius = data['radius'].flatten()[0]
        interval = data['dx'].flatten()[0]
        theta = data['theta'][0]
        phi = data['phi'][0]
        AF = data['AF']

        fig = plt.figure(num=1, figsize=(10, 8), dpi=300)

        dw.IFT_MaskArray(radius=radius, number=number, interval=interval, ax=fig.add_subplot(221))
        surf = dw.IFT_3D_surface(theta=theta, phi=phi, AF=AF, ax=fig.add_subplot(222, projection='3d'), title='')
        fig.colorbar(surf, shrink=0.7, pad=0.15)

        dw.IFT_line_plot(x=theta, y=AF[0], ax=fig.add_subplot(223), title=r"$\phi=0$")
        dw.IFT_line_plot(x=theta, y=AF[int(len(AF) / 2)], ax=fig.add_subplot(224), title=r"$\phi=\pi/2$")

        fig.suptitle(str(radius * 2) + r"$\lambda$ circle aperture array", family="times new roman", fontsize=15)

        plt.show()


if __name__ == '__main__':
    unittest.main()
