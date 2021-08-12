# -*- coding:utf-8 -*-
# Author : ZhengMX
# Data : 2021/4/19 15:55
# Project : ArrarySynthesis
# FileName : draw
# Cooperation : 265

import matplotlib.axes._axes as axes
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle


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
    ax.contour(X, Y, z_value, zdir='z', offset=20, cmap="coolwarm")  # 生成z方向投影，投到x-y平面

    ax.set_title(title)
    ax.set_xlabel('Theta(degree)')
    ax.set_ylabel('Phi(degree)')
    ax.set_zlabel('dB')

    # ax.set_zlim([-60, 0])

    fcb = fig.colorbar(surf, shrink=0.8, pad=0.1)

    plt.show()

    return fig


def Circular_Distribution(physical_aperture, physical_number, physical_interval, ax):
    # array mask

    half = int(physical_number / 2)

    radius = physical_aperture / 2
    element_radius = physical_interval * 0.9 / 2

    mask_line = np.arange(-half, half + 1, 1) * physical_interval
    temp_mask = np.einsum("i,j->ij", np.ones([2 * half + 1], dtype=float), mask_line * mask_line)
    dis_mask = temp_mask + temp_mask.T - radius * radius

    array_mask = np.array(np.where(dis_mask > 0, 0, 1), dtype=np.int8)

    idx, idy = np.where(array_mask == 1)

    # draw
    circle = Circle((0, 0), radius=radius, facecolor='lightblue', linewidth=1, alpha=1)
    ax.add_patch(circle)

    # Draw element

    for zmc in np.arange(idx.size):
        element_circle = Circle((mask_line[idx[zmc]], mask_line[idy[zmc]]), radius=element_radius, facecolor='gray',
                                linewidth=1, alpha=1)
        ax.add_patch(element_circle)

    ax.set_xlim([-radius * 1.1, radius * 1.1])
    ax.set_ylim([-radius * 1.1, radius * 1.1])

    ax.set_aspect('equal', 'box')
    ax.set_axis_off()
    ax.grid(True)

    ax.set_title(str(physical_aperture) + " (mm) circular aperture array " + str(len(idx)) + " elements", fontsize=6,
                 family='times new roman')


def IFT_MaskArray(radius, number, interval, ax):
    """
    Draw the circle aperture
    :param radius:
    :param number:
    :param interval:
    :param ax: handle of axe
    :return ax
    """

    phi = np.linspace(0, 2 * np.pi, 360 * 4)

    # array mask

    half = int(number / 2)
    mask_line = np.arange(-half, half + 1, 1) * interval
    temp_mask = np.einsum("i,j->ij", np.ones([2 * half + 1], dtype=float), mask_line * mask_line)
    dis_mask = temp_mask + temp_mask.T - radius * radius

    array_mask = np.array(np.where(dis_mask > 0, 0, 1), dtype=np.int8)

    idx, idy = np.where(array_mask == 1)

    # draw
    circle = ax.fill(radius * np.cos(phi), radius * np.sin(phi), color='lightblue', zorder=0)
    elements = ax.scatter(x=mask_line[idx], y=mask_line[idy], s=10, c='gray', zorder=1)

    # Arrow
    arrow_width = 0.03
    arrow_head_width = 0.3
    arrow_head_length = 0.5
    fontsize = 15

    ax.arrow(x=0, y=0, dx=radius - 0.5, dy=0,
             width=arrow_width, head_width=arrow_head_width, head_length=arrow_head_length, color='k')
    ax.text(x=radius / 2, y=0.3, s="r", size=fontsize, family='times new roman',
            style='italic')

    ax.set_aspect('equal', 'box')
    ax.set_axis_off()
    ax.grid(True)

    ax.set_title("Mask of Array with " + str(len(idx)) + " elements", fontsize=fontsize, family='times new roman')

    return ax


def IFT_LoopMaskArray(aperture, loop_number: np.array, loop_radius: np.array, ax: axes.Axes, centred=False):
    """
    Draw circular array on the circular aperture face
    :param aperture: the radius of the array
    :param loop_number: number of each loop
    :param loop_radius: radius of each loop
    :param ax: handle of subplot
    :param centred: the centre whether has the element for True or False
    :return:
    """
    detal_phi = 2 * np.pi / loop_number
    total_number = np.einsum("i->", loop_number)

    if centred:
        loc = np.zeros([2, total_number + 1], dtype=np.float64)
        loop_index = 1
    else:
        loc = np.zeros([2, total_number], dtype=np.float64)
        loop_index = 0

    for zmc, radius in enumerate(loop_radius):
        current_number = loop_number[zmc]
        phi = np.arange(0, current_number) * detal_phi[zmc]
        loc_x = radius * np.cos(phi)
        loc_y = radius * np.sin(phi)
        loc[0, loop_index:loop_index + current_number] = loc_x
        loc[1, loop_index:loop_index + current_number] = loc_y
        loop_index = loop_index + current_number

    phi = np.linspace(0, 2 * np.pi, 360 * 4)

    # draw
    circle = ax.fill(aperture * np.cos(phi), aperture * np.sin(phi), color='lightblue', zorder=0)
    elements = ax.scatter(x=loc[0], y=loc[1], s=10, c='gray', zorder=1)

    ax.set_aspect('equal', 'box')
    ax.set_axis_off()
    ax.grid(True)

    return ax


def IFT_3D_surface(theta, phi, AF, ax, title=''):
    r = np.linspace(-1, 1, len(theta))

    R, RHO = np.meshgrid(r, phi)

    X, Y = R * np.cos(RHO), R * np.sin(RHO)

    val = 20 * np.log10(AF / AF.max())

    surf = ax.plot_surface(X, Y, Z=val, cmap='coolwarm')
    ax.contour(X, Y, val, zdir='z', levels=8, offset=20, cmap="coolwarm")  # 生成z方向投影，投到x-y平面

    if title == '':
        title = "normalized 3D pattern"

    ax.set_title(title, family="times new roman", fontsize=15)
    # ax.set_xlabel('Theta(degree)')
    # ax.set_ylabel('Phi(degree)')
    ax.set_zlabel('/dB', family="times new roman")
    ax.set_zlim([-60, 0])

    return surf


def IFT_line_plot(x, y, ax, title=''):
    y = 20 * np.log10(y / y.max())

    theta = x * 180 / np.pi

    ax.plot(theta, y)

    ax.set_title(title, family="times new roman", style='italic')

    ax.set_xticks([-90, -60, -30, -5, 5, 30, 60, 90])
    ax.set_xlabel(r'$\theta$ (degree)', family="times new roman", style='italic', fontsize=15)
    ax.set_xlim([-90, 90])

    ax.set_ylabel('normalized pattern/dB', family="times new roman", fontsize=15)
    ax.set_ylim([-80, 0])
    ax.set_yticks([-80, -70, -60, -50, -40, -30, -20, -10, -3, 0])

    ax.grid(True)

    return ax


if __name__ == '__main__':
    radius = 8
    interval = 0.5
    number = 33

    fig = plt.figure(num=1, figsize=(8, 8), dpi=300)
    loop = 20
    loop_number = np.arange(1, loop) * 12
    loop_radius = np.arange(1, loop) * 13

    total_number = np.einsum("i->", loop_number)

    ax = IFT_LoopMaskArray(aperture=250, loop_number=loop_number, loop_radius=loop_radius, ax=fig.add_subplot(111),
                           centred=True)
    ax.set_title("Total number: " + str(total_number), fontsize=15, fontfamily="times new roman")
    # surf = IFT_3D_surface(theta=0, phi=0, AF=0, ax=fig.add_subplot(224, subplot_kw={'projection': '3d'}), title='')
    # fig.colorbar(surf, shrink=0.8, pad=0.1)

    plt.show()
