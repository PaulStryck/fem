from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from fem.mesh import SimplexMesh


def plot_mesh(m, labels):
    fig = plt.figure()

    projection = 'rectilinear'
    if m.dim == 3:
        projection = '3d'

    ax = fig.add_subplot(projection=projection)

    colors = ['black', 'red', 'blue']

    for i, e in enumerate(m.nfaces[1].arr):
        # plot all edges
        # plt.plot(m.nfaces[0][e, 0], m.nfaces[0][e, 1], m.nfaces[0][e, 2])
        plt.plot(*(m.nfaces[0].arr[e].T), 'k')

    if labels:
        # annotate points
        for i, x in enumerate(m.nfaces[0].arr):
            ax.annotate('(%s, %s)' % (0, i), xy=x, xytext=(3, 1),
                        textcoords='offset points', color=colors[0])

        # annotate higher dim entites
        for d in range(1, m.dim_submanifold+1):
            adj = m.adjacency(d, 0)
            for i, e in enumerate(adj):
                x = np.mean(m.nfaces[0].arr[e, :], axis=0)
                ax.annotate('(%s, %s)' % (d, i), xy=x, xytext=(0, 1),
                            textcoords='offset points', color=colors[d])

    # ax.axis(np.add(ax.axis(), [-.1, .1, -.1, .1]))

    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser(description="""Plot entities in the mesh""")
    parser.add_argument('n', type=int, nargs=1,
                        help="Number of vertices on each edge. N^2 in total")
    parser.add_argument('--labels', dest='labels', action='store_true')
    parser.add_argument('--no-labels', dest='labels', action='store_false')

    parser.set_defaults(labels=True)

    args = parser.parse_args()

    # m = SimplexMesh.Create_2d_unit_square_structured(n)
    m = SimplexMesh.Create_2d_unit_square_unstructured(args.n[0])

    m = SimplexMesh.Create_2d_manifold(args.n[0])
    plot_mesh(m, args.labels)

