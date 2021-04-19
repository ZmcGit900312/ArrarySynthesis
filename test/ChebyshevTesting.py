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

    def test_sample(self):
        sidelobe = 30
        scan = np.array([np.pi / 3, np.pi / 3])
        omega = np.array([5, 5]) / 180 * np.pi
        number = np.array([50, 50], dtype=int)

        theta = np.arange(30, 150, 1)
        phi = np.arange(-60, 60, 1)

        cbs_sample = ChebyshevPlaneSyn(sidelobe, scan, omega)
        cbs_sample.syntheis(number)
        cbs_sample.show()

        AF = cbs_sample.array_factor(theta * np.pi / 180, phi * np.pi / 180)
        AF_abs = np.abs(AF)
        AFnormal = 20 * np.log10(AF_abs / np.max(AF_abs))

        size = cbs_sample.get_size()
        surface_3D(theta, phi, AFnormal, str(size[0]) + "*" + str(size[1]) + " Array Factor")


if __name__ == '__main__':
    unittest.main()
