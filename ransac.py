import numpy as np
from matplotlib import pyplot as plt
from common import *

def homography_transform(X, H):
    N = X.shape[0]
    xH = np.hstack([X, np.ones((N,1))])
    yH = xH @ H.T
    Y = yH[:,:2] / yH[:,2:]
    return Y


def fit_homography(XY):
    N = XY.shape[0]
    A = []
    for i in range(N):
        x1, y1, x2, y2 = XY[i]
        A.append([-x1, -y1, -1, 0, 0, 0, x2*x1, x2*y1, x2])
        A.append([0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2])
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A) #U and S aren't used, but they must be extracted
    h = Vt[-1]
    H = h.reshape(3, 3)
    H /= H[2, 2]
    return H


def p1():
    data = np.load('./output/p1/transform.npy')
    X = data[:,:2]
    Y = data[:,2:]

    N = X.shape[0]
    A = np.zeros((2*N, 6))
    b = Y.flatten()
    for i in range(N):
        A[2*i] = [X[i,0], X[i,1], 1, 0, 0, 0]
        A[2*i+1] = [0, 0, 0, X[i,0], X[i,1], 1]
    v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    M = np.array([[v[0], v[1]], [v[3], v[4]]])
    t = np.array([v[2], v[5]])
    print("M:\n", M)
    print("t:\n", t, "\n")

    Y_hat = (X @ M.T) + t
    plt.scatter(X[:,1],X[:,0],c="red", label="Original")
    plt.scatter(Y[:,1],Y[:,0],c="green", label="Observed") #Y
    plt.scatter(Y_hat[:,1],Y_hat[:,0],c="blue", label="Y_hat")
    plt.legend()
    plt.savefig('./output/part1_4.png')
    plt.close()

    case = 8
    for i in range(case):
        XY = np.load('./output/p1/points_case_'+str(i)+'.npy')
        H = fit_homography(XY)
        print(H)
        Y_H = homography_transform(XY[:,:2], H)
        plt.scatter(XY[:,1],XY[:,0],c="red") #X
        plt.scatter(XY[:,3],XY[:,2],c="green") #Y
        plt.scatter(Y_H[:,1],Y_H[:,0],c="blue") #Y_hat
        plt.title('Case '+ str(i))
        plt.savefig('./output/case_'+str(i)+'.png')
        plt.close()



if __name__ == "__main__":
    p1()