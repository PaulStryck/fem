from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from fem.mesh import SimplexMesh


def plot_mesh(ax, m, labels):
    colors = ['black', 'red', 'blue']

    for e in m.nfaces[1].masked_view:
        # plot all edges
        # plt.plot(m.nfaces[0][e, 0], m.nfaces[0][e, 1], m.nfaces[0][e, 2])
        ax.plot(*(m.nfaces[0].arr[e].T), 'k')

    if labels:
        # annotate points
        for i, x in enumerate(m.nfaces[0].masked_view):
            ax.annotate('(%s, %s)' % (0, i), xy=x, xytext=(3, 1),
                        textcoords='offset points', color=colors[0])

        # annotate higher dim entites
        for d in range(1, m.dim_submanifold+1):
            adj = m.adjacency(d, 0)
            for i, e in enumerate(adj):
                x = np.mean(m.nfaces[0].arr[e, :], axis=0)
                ax.annotate('(%s, %s)' % (d, i), xy=x, xytext=(0, 1),
                            textcoords='offset points', color=colors[d])

    ax.axis(np.add(ax.axis(), [-.1, .1, -.1, .1]))
    ax.set_aspect('equal')


if __name__ == "__main__":
    parser = ArgumentParser(description="""Plot entities in the mesh""")
    parser.add_argument('n', type=int, nargs=2,
                        help="Number of vertices on each edge. N^2 in total")
    parser.add_argument('--labels', dest='labels', action='store_true')
    parser.add_argument('--no-labels', dest='labels', action='store_false')

    parser.set_defaults(labels=True)

    args = parser.parse_args()

    m = SimplexMesh.Create_2d_unit_square_structured(args.n[0])
    # m = SimplexMesh.Create_2d_unit_square_unstructured(args.n[0])

    # m = SimplexMesh.Create_2d_manifold(args.n[0])

    fig = plt.figure()

    projection = 'rectilinear'
    if m.dim == 3:
        projection = '3d'

    ax = fig.add_subplot(121, projection=projection)
    ax.set_title("Structured Grid, n = 4")
    ax.set_xticks(np.linspace(0,1,4), ['0', '1/3', '2/3', '1'])
    ax.set_yticks(np.linspace(0,1,4), ['0', '1/3', '2/3', '1'])
    plot_mesh(ax, m, args.labels)

    m = SimplexMesh.Create_2d_unit_square_structured(args.n[1])
    ax = fig.add_subplot(122, projection=projection)
    ax.set_title("Structured Grid, n = 6")
    ax.set_xticks(np.linspace(0,1,6))
    ax.set_yticks(np.linspace(0,1,6))
    plot_mesh(ax, m, args.labels)

    plt.savefig("figs/structured_grids.png", bbox_inches="tight")
