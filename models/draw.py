# -*- coding:utf-8 -*-
# Author : ZhengMX
# Data : 2021/4/19 15:55
# Project : ArrarySynthesis
# FileName : draw
# Cooperation : 265

import numpy as np
from matplotlib import pyplot as plt


def surface_3D(x_scope, y_scope, z_value, title="3D_surface"):
    """
    Draw 3D pattern with surface
    :param title: title of the 3D picture
    :param X:
    :param Y:
    :param Z:
    :return: handle of fig
    """

    fig = plt.figure(num=1, figsize=(8, 6), dpi=300)  # 参数为图片大小
    ax = plt.axes(projection='3d')  # get current axes，且坐标轴是3d的
    # ax.set_aspect('equal')  # 坐标轴间比例一致
    X, Y = np.meshgrid(x_scope, y_scope)

    surf = ax.plot_surface(X, Y, Z=z_value, cmap='coolwarm')
    ax.contour(X, Y, z_value, zdir='z', offset=10, cmap="coolwarm")  # 生成z方向投影，投到x-y平面

    ax.set_title(title)
    ax.set_xlabel('Theta(degree)')
    ax.set_ylabel('Phi(degree)')
    ax.set_zlabel('dB')

    fcb = fig.colorbar(surf, shrink=0.8, pad=0.1)

    plt.show()

    return fig


if __name__ == '__main__':
    x_s = np.arange(-5, 5, 0.1)
    y_s = np.arange(-5, 5, 0.1)
    xx, yy = np.meshgrid(x_s, y_s)
    z_v = np.sin(np.sqrt(xx ** 2 + yy ** 2))
    surface_3D(x_s, y_s, z_v)
