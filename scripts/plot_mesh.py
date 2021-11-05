import matplotlib.pyplot as plt
import numpy as np

from fem.mesh import SimplexMesh


def plot_mesh(m):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    colors = ['black', 'red', 'blue']

    for i, e in enumerate(m.nfaces[1]):
        # plot all edges
        plt.plot(m.nfaces[0][e, 0], m.nfaces[0][e, 1], 'k')

    # annotate points
    for i, x in enumerate(m.nfaces[0]):
        ax.annotate('(%s, %s)' % (0, i), xy=x, xytext=(3, 1),
                    textcoords='offset points', color=colors[0])

    # annotate higher dim entites
    for d in range(1, m.dim+1):
        adj = m.adjacency(d, 0)
        for i, e in enumerate(adj):
            x = np.mean(m.nfaces[0][e, :], axis=0)
            ax.annotate('(%s, %s)' % (d, i), xy=x, xytext=(0, 1),
                        textcoords='offset points', color=colors[d])

    ax.axis(np.add(ax.axis(), [-.1, .1, -.1, .1]))

    plt.show()


if __name__ == "__main__":
    n = 3
    m = SimplexMesh.Create_2d_unit_square_structured(n)

    plot_mesh(m)

