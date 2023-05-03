import numpy as np
import matplotlib.pyplot as plt

"""
Based on Philip Mocz (2021) implementation
"""

# convert subindex (i,j) to linear index (idx)
def sub2ind(array_shape, i, j):
    idx = i * array_shape[1] + j
    return idx


# calculate acceleration on nodes using hooke's law + gravity
def getAcc(pos, vel, ci, cj, springL, spring_coeff, gravity):
    # initialize
    acc = np.zeros(pos.shape)

    # Hooke's law: F = - k * displacement along spring
    sep_vec = pos[ci, :] - pos[cj, :]
    sep = np.linalg.norm(sep_vec, axis=1)
    dL = sep - springL
    ax = - spring_coeff * dL * sep_vec[:, 0] / sep
    ay = - spring_coeff * dL * sep_vec[:, 1] / sep
    np.add.at(acc[:, 0], ci, ax)
    np.add.at(acc[:, 1], ci, ay)
    np.add.at(acc[:, 0], cj, -ax)
    np.add.at(acc[:, 1], cj, -ay)

    # gravity
    acc[:, 1] += gravity

    return acc


# apply box boundary conditions, reverse velocity if outside box
def applyBoundary(pos, vel, boxsize):
    for d in range(0, 2):
        is_out = np.where(pos[:, d] < 0)
        pos[is_out, d] *= -1
        vel[is_out, d] *= -1

        is_out = np.where(pos[:, d] > boxsize)
        pos[is_out, d] = boxsize - (pos[is_out, d] - boxsize)
        vel[is_out, d] *= -1

    return (pos, vel)

class body():
    def __init__(self, pos, ci, cj):
        # Simulation parameters
        self.dt = 0.1  # timestep
        self.spring_coeff = 40  # Hooke's law spring coefficient
        self.gravity = -0.1  # strength of gravity

        # construct spring nodes / initial conditions
        self.pos = pos
        self.vel = np.zeros(self.pos.shape)
        self.acc = np.zeros(self.pos.shape)

        self.ci = ci
        self.cj = cj

        # calculate spring rest-lengths
        self.springL = np.linalg.norm(self.pos[self.cj, :] - self.pos[self.ci, :], axis=1)

    def step(self, ax):
        # (1/2) kick
        self.vel += self.acc * self.dt / 2.0

        # drift
        self.pos += self.vel * self.dt

        # apply boundary conditions
        pos, vel = applyBoundary(self.pos, self.vel, 3)

        # update accelerations
        self.acc = getAcc(self.pos, self.vel, self.cj, self.ci, self.springL, self.spring_coeff, self.gravity)

        # (1/2) kick
        self.vel += self.acc * self.dt / 2.0

        # plot in real time
        if ax != None:
            ax.cla()
            ax.plot(pos[[self.ci, self.cj], 0], pos[[self.ci, self.cj], 1], color='blue')
            ax.scatter(pos[:, 0], pos[:, 1], s=10, color='blue')

    plt.show()

class ngrid_body(body):
    def __init__(self, N):
        # construct spring network connections
        ci = []
        cj = []
        #  o--o
        for r in range(0, N):
            for c in range(0, N - 1):
                idx_i = sub2ind([N, N], r, c)
                idx_j = sub2ind([N, N], r, c + 1)
                ci.append(idx_i)
                cj.append(idx_j)
        # o
        # |
        # o
        for r in range(0, N - 1):
            for c in range(0, N):
                idx_i = sub2ind([N, N], r, c)
                idx_j = sub2ind([N, N], r + 1, c)
                ci.append(idx_i)
                cj.append(idx_j)
        # o
        #   \
        #     o
        for r in range(0, N - 1):
            for c in range(0, N - 1):
                idx_i = sub2ind([N, N], r, c)
                idx_j = sub2ind([N, N], r + 1, c + 1)
                ci.append(idx_i)
                cj.append(idx_j)
        #     o
        #   /
        # o
        for r in range(0, N - 1):
            for c in range(0, N - 1):
                idx_i = sub2ind([N, N], r + 1, c)
                idx_j = sub2ind([N, N], r, c + 1)
                ci.append(idx_i)
                cj.append(idx_j)

        xlin = np.linspace(1, 2, N)

        x, y = np.meshgrid(xlin, xlin)
        pos = np.vstack((x.flatten(), y.flatten())).T

        super().__init__(pos=pos, ci=ci, cj=cj)

class wheel_body(body):
    def __init__(self, N, R):
        a = np.linspace(start=0., stop=2*np.pi, num=N+1)[:-1]
        pos = np.stack([R * np.cos(a), R * np.sin(a)], axis=1)
        pos = np.concatenate([pos, np.array([[0., 0]])])
        pos += R + 0.5

        ci = np.arange(N).tolist()
        cj = np.roll(np.arange(N), 1).tolist()

        ci = ci + np.arange(N).tolist()
        cj = cj + [N] * N

        super().__init__(pos=pos, ci=ci, cj=cj)

def main(Nt=400):
    #b = body(pos=np.array([[1.01, 0.2], [1, 0.4], [1.5, 0.3]]), ci=[0, 1, 2], cj=[1, 2, 0])
    #b = ngrid_body(5)
    b = wheel_body(6, 0.6)

    # prep figure
    fig = plt.figure(figsize=(4, 4), dpi=80)
    ax = fig.add_subplot(111)

    # Simulation Main Loop
    for i in range(Nt):
        b.step(ax)
        ax.set(xlim=(0, 3), ylim=(0, 3))
        ax.set_aspect('equal', 'box')
        ax.set_xticks([0, 1, 2, 3])
        ax.set_yticks([0, 1, 2, 3])
        plt.pause(0.001)

    plt.show()

if __name__ == "__main__":
    main()