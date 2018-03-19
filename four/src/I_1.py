# coding=utf-8
'''
Plotting:
make a grid {
    w = (w1, w2),
    w1 ∈ [−6, 6],
    w2 ∈ [−6, 6]
}
and plot the values of Lsimple(w)

Lsimple(w) = Lsimple(w) = [σ(w, [1, 0]) − 1]^2 + [σ(w, [0, 1])]^2 + [σ(w, [1, 1]) − 1]^2

(Matlab, Octave or matplotlib in python should make this easy for you).

By inspecting the graph which value of w is minimizing the loss Lsimple(w)? And what is the minimum of Lsimple(w)?
'''

from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def createPoints():
    w1 = int(np.random.uniform(-6,7))
    w2 = int(np.random.uniform(-6,7))
    w = np.matrix(str(w1) + "," + str(w2))
    print("#"*10 + "\n" +str(w) +"\n" + "#"*10 + "\n" +  str(w.shape))
    return w

def lossFunction(w):
    a = np.power((sigmoidLogisticFunction(w, [1,0]) - 1), 2)
    b = np.power((sigmoidLogisticFunction(w, [0,1])), 2)
    c = np.power((sigmoidLogisticFunction(w, [1,1]) - 1), 2)
    return a + b + c


def sigmoidLogisticFunction(w,x):
    return (1 / (1 + np.exp(x * -w.T)))

def task():
    Xaxis = np.arange(-7, 6, 0.1)
    Yaxis = np.arange(-7, 6, 0.1)

    X, Y = np.meshgrid(Xaxis, Yaxis)

    zs = np.array([lossFunction(np.matrix(str(x) + "," + str(y))) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    minZ = zs.min()
    print("Minimum loss value: ", minZ)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel('w1 Label')
    ax.set_ylabel('w2 Label')
    ax.set_zlabel('Loss Label')

    ax.set_aspect(1)

    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    plt.show()





if __name__ == "__main__":
    task()
