#implement equal sampling of map vector
import numpy as np
import math
import matplotlib.pyplot as plt

class Spline:
    """
    Cubic Spline class
    """

    def __init__(self, x, y):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = np.array(x)
        self.y = np.array(y)

        self.eps = np.finfo(float).eps

        self.nx = len(x)  # dimension of x
        h = np.diff(x)

        # calc coefficient c
        self.a = np.array([iy for iy in y])

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)
        #  print(self.c1)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i] + self.eps))
            tb = (self.a[i + 1] - self.a[i]) / (h[i] + self.eps) - h[i] * \
                (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)
        self.b = np.array(self.b)
        self.d = np.array(self.d)

    def calc(self, t):
        """
        Calc position
        if t is outside of the input x, return None
        """
        t = np.asarray(t)
        mask = np.logical_and(t < self.x[0], t > self.x[-1])
        t[mask] = self.x[0]

        i = self.__search_index(t)
        dx = t - self.x[i.astype(int)]
        result = self.a[i] + self.b[i] * dx + \
            self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        result = np.asarray(result)
        result[mask] = None
        return result

    def calcd(self, t):
        """
        Calc first derivative
        if t is outside of the input x, return None
        """
        t = np.asarray(t)
        mask = np.logical_and(t < self.x[0], t > self.x[-1])
        t[mask] = 0

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0

        result = np.asarray(result)
        result[mask] = None
        return result

    def calcdd(self, t):
        """
        Calc second derivative
        """
        t = np.asarray(t)
        mask = np.logical_and(t < self.x[0], t > self.x[-1])
        t[mask] = 0

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx

        result = np.asarray(result)
        result[mask] = None
        return result

    def __search_index(self, x):
        """
        search data segment index
        """
        indices = np.asarray(np.searchsorted(self.x, x, "left") - 1)
        indices[indices <= 0] = 0
        return indices

    def __calc_A(self, h):
        """
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        #  print(A)
        return A

    def __calc_B(self, h):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / (h[i + 1] + self.eps) \
                       - 3.0 * (self.a[i + 1] - self.a[i]) / (h[i] + self.eps)
        return B
class Spline2D:
    """
    2D Cubic Spline class
    """

    def __init__(self, x, y, resolution=0.1):
        self.s = self.__calc_s(x, y)
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)

        self.s_fine = np.arange(0, self.s[-1], resolution)
        xy = np.array([self.calc_global_position_online(s_i) for s_i in self.s_fine])

        self.x_fine = xy[:, 0]
        self.y_fine = xy[:, 1]

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_global_position_online(self, s):
        """
        calc global position of points on the line, s: float
        return: x: float; y: float; the global coordinate of given s on the spline
        """
        x = self.sx.calc(s)
        y = self.sy.calc(s)

        return x, y

    def calc_global_position_offline(self, s, d):
        """
        calc global position of points in the frenet coordinate w.r.t. the line.
        s: float, longitudinal; d: float, lateral;
        return: x, float; y, float;
        """
        s_x = self.sx.calc(s)
        s_y = self.sy.calc(s)

        theta = math.atan2(self.sy.calcd(s), self.sx.calcd(s))
        x = s_x - math.sin(theta) * d
        y = s_y + math.cos(theta) * d
        return x, y

    def calc_frenet_position(self, x, y):
        """
        cal the frenet position of given global coordinate (x, y)
        return s: the longitudinal; d: the lateral
        """
        # find nearst x, y
        diff = np.hypot(self.x_fine - x, self.y_fine - y)
        idx = np.argmin(diff)
        [x_s, y_s] = self.x_fine[idx], self.y_fine[idx]
        s = self.s_fine[idx]

        # compute theta
        theta = math.atan2(self.sy.calcd(s), self.sx.calcd(s))
        d_x, d_y = x - x_s, y - y_s
        cross_rd_nd = math.cos(theta) * d_y - math.sin(theta) * d_x
        d = math.copysign(np.hypot(d_x, d_y), cross_rd_nd)
        return s, d

    def calc_curvature(self, s):
        """
        calc curvature
        """
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
        return k

    def calc_yaw(self, s):
        """
        calc yaw
        """
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        yaw = np.arctan2(dy, dx)
        return yaw
    
def visualize_centerline(centerline) -> None:
    """Visualize the computed centerline.
    Args:
        centerline: Sequence of coordinates forming the centerline
    """
    line_coords = list(zip(*centerline))
    lineX = line_coords[0]
    lineY = line_coords[1]
    plt.plot(lineX, lineY, "--", color="grey", alpha=1, linewidth=1, zorder=0)
    plt.text(lineX[0], lineY[0], "s")
    plt.text(lineX[-1], lineY[-1], "e")
    plt.axis("equal")