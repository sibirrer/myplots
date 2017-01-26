
"""
a nice setting for contour plots originally from Lukas Gamper
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def contourPlot(xlabel, ylabel, xMin, xMax, yMin, yMax, x1, y1, label1, x2=None, y2=None, label2=None):
    gridLen = 128
    zLevels = np.array([5, 10, 50, 100])
    xGrid, yGrid = np.meshgrid(np.r_[xMin:xMax:(gridLen * 1j)], np.r_[yMin:yMax:(gridLen * 1j)])
    gridXY = np.append(xGrid.reshape(-1, 1), yGrid.reshape(-1, 1), axis=1)

    maskKdeXY1 = np.where((x1 > xMin) & (x1 < xMax) & (y1 > yMin) & (y1 < yMax))
    kdeXY1 = stats.kde.gaussian_kde(np.array([x1[maskKdeXY1], y1[maskKdeXY1]]))
    z1 = kdeXY1(gridXY.T).reshape(gridLen, gridLen)
    n1 = x1[maskKdeXY1].size
    if x2 is not None:
        maskKdeXY2 = np.where((x2 > xMin) & (x2 < xMax) & (y2 > yMin) & (y2 < yMax))
        kdeXY2 = stats.kde.gaussian_kde(np.array([x2[maskKdeXY2], y2[maskKdeXY2]]))
        z2 = kdeXY2(gridXY.T).reshape(gridLen, gridLen)
        n2 = x2[maskKdeXY2].size

    cs1 = plt.contour(xGrid, yGrid, n1 * z1, levels=zLevels, colors='#FF0000', border='#000000',
                      linewidths=np.linspace(0.3, 2, num=len(zLevels)))
    cs1.ax.autoscale(False)

    cs1.ax.set_xlabel(xlabel, fontsize=15)
    cs1.ax.set_ylabel(ylabel, fontsize=15)
    p1, = plt.plot(x1, y1, 'ro', markersize=0.4, markeredgewidth=0.007, alpha=1., rasterized=True)
    p1, = plt.plot([2 * xMax, 3 * xMax], [2 * yMax, 3 * yMax], 'r', label=label1)
    plt.clabel(cs1, inline=0.5, fontsize=7, fontcolor='#FF0000', fmt='%1.0f')
    if x2 is not None:
        cs2 = plt.contour(xGrid, yGrid, n2 * z2, levels=zLevels, colors='b', border='#000000',
                          linewidths=np.linspace(0.3, 2, num=len(zLevels)))
        cs2.ax.autoscale(False)
        p1, = plt.plot(x2, y2, 'bo', markersize=0.4, markeredgewidth=0.007, alpha=1., rasterized=True)
        p1, = plt.plot([2 * xMax, 3 * xMax], [2 * yMax, 3 * yMax], 'b', label=label2)
        plt.clabel(cs2, inline=0.5, fontsize=7, fontcolor='#b', fmt='%1.0f')
    plt.legend()