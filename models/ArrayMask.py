# -*- coding:utf-8 -*-
# Author : ZhengMX
# Data : 2021/4/17 16:17
# Project : ArrarySynthesis
# FileName : ArrayMask
# Cooperation : 265
import numpy as np


class Mask:
    def __init__(self, numU=1, numV=1):
        self.Nu = numU
        self.Nv = numV
        self.mask = np.ones([numU, numV], dtype=int)
        self.AF = np.zeros_like(self.mask, dtype=np.complex128)

    Nu = 1
    Nv = 1
    mask = np.ones([1, 1], dtype=int)
    AF = np.ones([1, 1], dtype=np.complex128)
