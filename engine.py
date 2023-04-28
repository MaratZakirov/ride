import numpy as np

# Fundamental object
class phy_object():
    # to create object please pass
    # 1. Coordinates of masses
    # 2. Masses themselfs
    # 3. Adjacent table
    # 4. K - table
    # 5. g - force
    def __init__(self, xy, ms, A, K, g, iscen=False):
        self.xy = xy
        self.v = np.zeros_like(self.xy)
        self.ms = ms
        self.A = A
        self.K = K
        self.g = g
        self.iscen = iscen
        self.dt = 0.01
        self.l = np.zeros_like(A, dtype=float)
        self.F = np.zeros_like(xy)

        for j in range(self.A.shape[1]):
            p = self.xy - self.xy[self.A[:, j]]
            self.l[:, j] = (p * p).sum(axis=1)

    def step(self):
        # F = -kx
        F = np.zeros_like(self.l)
        for j in range(self.A.shape[1]):
            p = self.xy - self.xy[self.A[:, j]]
            p_hat = p / np.linalg.norm(p, axis=1, keepdims=True)
            F += self.K[:, [j]] * (self.l[:, [j]] - (p * p).sum(axis=1, keepdims=True)) * p_hat

        self.F = F
        self.a = F / self.ms[:, None] + self.g

        # apply for speed
        self.v = self.v + self.a * self.dt

        # apply speed
        self.xy = self.xy + self.v

        # hit floor
        self.xy[:, 1] = np.clip(self.xy[:, 1], a_min=0, a_max=100)

    def show(self, ax, ax_aux):
        l = np.zeros_like(self.l)
        for j in range(self.A.shape[1]):
            p = self.xy - self.xy[self.A[:, j]]
            l[:, j] = (p * p).sum(axis=1)

        color = self.l - l
        color = 1 / (1 + np.exp(-0.6 * color))
        color = color[..., None] * np.array([0, 0, 1]) + (1 - color[..., None]) * np.array([1, 0, 0])

        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                ax.plot([self.xy[i, 0], self.xy[self.A[i, j], 0]], [self.xy[i, 1], self.xy[self.A[i, j], 1]], c=color[i, j])
            ax.plot([self.xy[i, 0], self.xy[i, 0] + self.F[i, 0]], [self.xy[i, 1], self.xy[i, 1] + self.F[i, 1]], c='g')
        #ax.plot(*np.split(self.xy, 2, 1), c='b')
        ax_aux.plot(np.linalg.norm(self.F, axis=1), c='b')
        print(np.linalg.norm(self.F, axis=1))

class wheel(phy_object):
    def __init__(self, num=3, R=5):
        phi = np.linspace(start=.0, stop=2*np.pi, num=num+1)[:-1]
        xy = np.array([0, 12]) + np.stack([R * np.cos(phi), R * np.sin(phi)], axis=1)
        ms = np.ones(num)
        A = np.zeros((num, 2), dtype=int)

        A[:, 0] = np.roll(np.arange(num), 1)
        A[:, 1] = np.roll(np.arange(num), -1)

        K = 0.6 * np.ones((num, 2))
        g = np.array([0, -1])
        super().__init__(xy, ms, A, K, g)

# Brief selfcheck demo
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from celluloid import Camera
    fig, (ax, ax_aux) = plt.subplots(2, 1)
    camera = Camera(fig)

    ax.set_aspect('equal')
    ax.set_ylim(0, 40)
    ax.set_xlim(-20, 20)

    w = wheel()

    for i in range(100):
        w.show(ax, ax_aux)
        w.step()
        camera.snap()

    animation = camera.animate()

    plt.show()