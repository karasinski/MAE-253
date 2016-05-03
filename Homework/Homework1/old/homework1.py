import numpy as np
import numpy.linalg as la
import networkx as nx
import matplotlib.pylab as plt


def problem1():
    f, ax = plt.subplots()
    A = np.array([k**-3 for k in range(1, 1000001)]).sum()
    x = np.linspace(1, 1E6)

    ax.loglog(x, 2 * x**-3, 'k-', label='Incorrect Normalization')
    ax.loglog(x, A * x**-3, 'r', label='Correct Normalization')
    plt.xlabel('$k$')
    plt.ylabel('$p(k)$')
    plt.legend()
    plt.savefig('p1.pdf')
    plt.close()


def problem2():
    A = np.array([[0, 1, 0, 0, 1],
                  [0, 0, 1, 0, 0],
                  [1, 0, 0, 1, 1],
                  [0, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0]], dtype=float)
    print(A)

    #g = nx.DiGraph(M)
    #nx.draw(g)
    #plt.show()

    T = np.copy(A)
    for i in range(len(T)):
        if (np.sum(T[:, i]) > 0):
            T[:, i] = T[:, i]/np.sum(T[:, i])
    print(T)

    eigvals, eigvec = la.eig(T)
    pi = np.real(eigvec[:, 0]/np.sum(eigvec[:, 0]))
    print(pi)

    undirected = ((A + A.T) > 0).astype(float)

    print(undirected.sum(axis=1))
    print(undirected.sum(axis=1))


#The Barabasi and Albert model
#• Note only the old nodes are capable of attaining high degree.

#“Winners don’t take all: Characterizing the competition for links
#on the web”, D M. Pennock, G. W. Flake, S. Lawrence, E. J.
#Glover, C. Lee Giles, PNAS 99 (2002).
#• First mover advantage
#• Second mover advantage
