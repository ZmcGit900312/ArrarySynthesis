import unittest

import numpy as np
import numpy.polynomial.chebyshev as cbs

from models.Core import ChebyshevPlaneSyn
from models.draw import surface_3D


class ChebyshevTestCase(unittest.TestCase):
    def test_chebyshev_arithmetic(self):
        print("chebdomain:\t" + str(cbs.chebdomain))
        print("chebzero:\t" + str(cbs.chebzero))
        print("chebone\t:" + str(cbs.chebone))
        print("chebx\t:" + str(cbs.chebx))

        p1 = cbs.chebval(0.2, 4)
        print("P(0.5) = " + str(p1))

        N = 100
        chebyshevMatrix = np.zeros([N, N], dtype=float)
        for zmc in np.arange(N):
            vec = np.zeros([zmc + 1], dtype=float)
            vec[zmc] = 1.
            chebyshevMatrix[zmc, 0:zmc + 1] = cbs.cheb2poly(vec)

        print("Size of Matrix:")
        print(chebyshevMatrix.shape)

        # for zmc in np.arange(chebyshevMatrix.shape[0]):
        #     print(chebyshevMatrix[zmc])
        np.save("chebyshevCoeficientsMatrix100", chebyshevMatrix)

    def test_chebyshev_load(self):
        chebyshevMatrix = np.load("chebyshevCoeficientsMatrix100.npy")
        print(chebyshevMatrix.shape)

    def test_chebyshev_sample(self):
        sidelobe = 30
        scan = np.array([np.pi / 3, np.pi / 3])
        omega = np.array([3, 3]) / 180 * np.pi
        number = np.array([39, 39], dtype=int)

        theta = np.arange(30, 150, 1)
        phi = np.arange(-60, 60, 1)

        cbs_sample = ChebyshevPlaneSyn(sidelobe, scan, omega)
        cbs_sample.synthesis(number)
        # cbs_sample.synthesis()
        cbs_sample.show()

        theta_degree = theta * np.pi / 180
        phi_degree = phi * np.pi / 180

        AF = cbs_sample.array_factor(theta_degree, phi_degree)
        AF_abs = np.abs(AF)
        AFnormal = 20 * np.log10(AF_abs / np.max(AF_abs))

        size = cbs_sample.get_size()

        gain = np.round(10 * np.log10(cbs_sample.direct), 2)

        picture_title = str(size[0]) + "*" + str(size[1]) + " Array Factor" + "(Gain: " \
                        + str(gain) + "dB )"

        surface_3D(theta, phi, AFnormal, picture_title)

    def test_chebyshev_sample_undepart(self):
        sidelobe = 30
        scan = np.array([np.pi / 3, np.pi / 3])
        omega = np.array([3, 3]) / 180 * np.pi
        number = np.array([39, 39], dtype=int)

        theta = np.arange(30, 150, 1)
        phi = np.arange(-60, 60, 1)

        cbs_sample = ChebyshevPlaneSyn(sidelobe, scan, omega)
        cbs_sample.undepart_synthesis()
        cbs_sample.show()

        theta_degree = theta * np.pi / 180
        phi_degree = phi * np.pi / 180

        AF = cbs_sample.undepart_array_factor(theta_degree, phi_degree)
        AF_abs = np.abs(AF)
        AFnormal = 20 * np.log10(AF_abs / np.max(AF_abs))

        size = cbs_sample.get_size()

        gain = np.round(10 * np.log10(cbs_sample.direct), 2)

        picture_title = str(size[0]) + "*" + str(size[0]) + " Array Factor" + "(Gain: " \
                        + str(gain) + "dB )"

        surface_3D(theta, phi, AFnormal, picture_title)


if __name__ == '__main__':
    unittest.main()
