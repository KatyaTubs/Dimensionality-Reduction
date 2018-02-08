import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, metrics
import sklearn
from matplotlib import offsetbox

from mpl_toolkits.mplot3d import Axes3D


def plot_embedding_list(data, y, titles):
    fig, ax = plt.subplots(nrows=len(data), ncols=1)
    fig.set_size_inches(7, 21)
    for i in range(len(titles)):
        ax[i].set_title(titles[i])

    for j in range(len(data)):

        X = data[j]

        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)

        for i in range(X.shape[0]):
            ax[j].text(X[i, 0], X[i, 1], str(digits.target[i]),
                       color=plt.cm.Set1(y[i] / 10.),
                       fontdict={'weight': 'bold', 'size': 9})

        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(digits.data.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < 4e-3:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X[i]]]
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                    X[i])
                ax[j].add_artist(imagebox)
        plt.xticks([]), plt.yticks([])


def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    if title is not None:
        plt.title(title)


def plot_faces_with_images(data, images, titles, image_num=25):
    '''
    A plot function for viewing images in their embedded locations. The
    function receives the embedding (X) and the original images (images) and
    plots the images along with the embeddings.

    :param X: Nxd embedding matrix (after dimensionality reduction).
    :param images: NxD original data matrix of images.
    :param title: The title of the plot.
    :param num_to_plot: Number of images to plot along with the scatter plot.
    :return: the figure object.
    '''

    n, pixels = np.shape(images)
    img_size = int(pixels ** 0.5)

    fig, ax = plt.subplots(nrows=1, ncols=len(data))
    fig.set_size_inches(21, 6)
    for i in range(len(titles)):
        ax[i].set_title(titles[i])

    # draw random images and plot them in their relevant place:
    for j in range(len(data)):
        X = data[j]
        # get the size of the embedded images for plotting:
        x_size = (max(X[:, 0]) - min(X[:, 0])) * 0.1
        y_size = (max(X[:, 1]) - min(X[:, 1])) * 0.1

        for i in range(image_num):
            img_num = np.random.choice(n)
            x0, y0 = X[img_num, 0] - x_size / 2., X[img_num, 1] - y_size / 2.
            x1, y1 = X[img_num, 0] + x_size / 2., X[img_num, 1] + y_size / 2.
            img = images[img_num, :].reshape(img_size, img_size)
            ax[j].imshow(img, aspect='auto', cmap=plt.cm.gray, zorder=100000,
                         extent=(x0, x1, y0, y1))

            # draw the scatter plot of the embedded data points:
            ax[j].scatter(X[:, 0], X[:, 1], marker='.', alpha=0.7)

    return fig


def MDS(X, d):
    '''
    Given a NxN pairwise distance matrix and the number of desired dimensions,
    return the dimensionally reduced data points matrix after using MDS.

    :param X: NxN distance matrix.
    :param d: the dimension.
    :return: Nxd reduced data point matrix.
    '''

    X = X ** 2
    n = np.shape(X)[0]
    H = np.matrix((np.eye(n)) - (1 / n) * np.ones(np.shape(X)))
    S = (-0.5) * np.dot(H, np.dot(X, H))  # matrix multiplication
    (eigvals, eigvecs) = np.linalg.eigh(S)

    eigvals_root = np.sqrt(eigvals[(n - d):]).tolist()  # take the square root of last d eigenvalues
    biggest_eigvecs = eigvecs[:, (len(eigvals) - d):]  # take the last d eigenvectors
    eigvals_root_matrix = [[eigvals_root], ] * n  # duplicate the eigvals vector n times
    eigvals_root_matrix = np.reshape(eigvals_root_matrix, (n, d))

    return eigvals, np.array(biggest_eigvecs) * np.array(eigvals_root_matrix)  # element-wise multiplication


def knn(X, k):
    '''
    Claculates k nearest neighbors of each line in given matrix
    :param X: Matrix where each row is datapoint
    :param k: the number of neighbors
    :return: nearest - boolean matrix where nearest[i,j] = 1 iff j is in k nearest neighbors of i
    '''
    N = np.shape(X)[0]

    distances = sklearn.metrics.pairwise.euclidean_distances(X)
    nearest = np.zeros((N, N))
    for i in range(N):
        sorted_indexes = np.argsort(distances[i])
        nearest[i, sorted_indexes[1:k + 1]] = 1
    return nearest


def finding_W(X, nearest, k):
    """
    Performs the second step in LLE algorithm
    :param X: the given data
    :param nearest: boolean matrix of nearest neighbors
    :param k: number of nearest neighbors
    :return:
    """
    N = np.shape(X)[0]
    W = np.zeros((N, N))
    for i in range(N):
        indexes = nearest[i].astype(bool)
        neighbors = X[indexes] - X[i]
        gram = np.dot(neighbors, neighbors.transpose())
        w = np.dot(np.linalg.pinv(gram), np.ones(k))
        w = w / np.sum(w)
        W[i][indexes] = w
    return W


def finding_Y(W, d):
    '''
    Performs the third step in LLE algorithm
    :param W: The result of the second step of the algorithm
    :param d: the dimension to redur=ce the data to
    :return: the output of LLE algoruthm
    '''
    N = np.shape(W)[0]
    M = np.dot((np.eye(N) - W).transpose(), np.eye(N) - W)
    (eigvals, eigvecs) = np.linalg.eigh(M)
    Y = eigvecs[:, 1:d + 1]
    return Y


def LLE(X, d, k):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the LLE algorithm.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param k: the number of neighbors for the weight extraction.
    :return: Nxd reduced data matrix.
    '''

    nearest = knn(X, k)
    W = finding_W(X, nearest, k)
    return finding_Y(W, d)


def DiffusionMap(X, d, sigma, t, k=None):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the Diffusion Map algorithm. The k parameter allows restricting the
    gram matrix to only the k nearest neighbor of each data point.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param sigma: the sigma of the gaussian for the gram matrix transformation.
    :param t: the scale of the diffusion (amount of time steps).
    :param k: the amount of neighbors to take into account when calculating the gram matrix.
    :return: Nxd reduced data matrix.
    '''
    N = np.shape(X)[0]
    K = np.zeros((N, N))

    X_dists = (metrics.pairwise.euclidean_distances(X) ** 2) / (-sigma)
    K = np.exp(X_dists)

    if not (k == None):
        for i in range(N):
            K[i, np.argsort(K[i])[0:N - k]] = 0

    K = K / np.sum(K, axis=1, keepdims=True)

    (eigvals, eigvecs) = np.linalg.eig(K)
    sorted_indexes = np.flip(np.argsort(eigvals), 0)
    eigvecs = eigvecs[:, sorted_indexes][:, 1:d + 1]
    eigvals = eigvals[sorted_indexes][1:d + 1]
    return np.multiply(eigvals ** t, eigvecs)


def plot_swiss_roll(data, titles, color):
    fig = plt.figure(figsize=(18, 5))
    for i in range(len(data)):
        X = data[i]
        ax = fig.add_subplot(1, len(data), i + 1, projection='3d')
        ax.set_title(titles[i])
        ax.scatter(X[:, 0], X[:, 1], c=color, cmap=plt.cm.Spectral)
    return fig


if __name__ == '__main__':
    N = 500
    D = 3
    X = np.random.normal(size=(N, D))
    fig = plt.figure(figsize=(12, 8))

    ax0 = fig.add_subplot(221, projection='3d')
    ax0.set_title('Original Data')
    ax0.scatter(X[:, 0], X[:, 1],X[:,2])

    mds = MDS(sklearn.metrics.pairwise.euclidean_distances(X),2)[1]
    ax1 = fig.add_subplot(222, projection='3d')
    ax1.set_title('MDS')
    ax1.scatter(mds[:, 0], mds[:, 1])


    lle = LLE(X, d = 2, k = 70)
    ax2 = fig.add_subplot(223, projection='3d')
    ax2.set_title('LLE')
    ax2.scatter(lle[:, 0], lle[:, 1])

    dm = np.real(DiffusionMap(X, d = 2, t = 10, sigma  = 100))
    ax3 = fig.add_subplot(224, projection='3d')
    ax3.set_title('DiffusionMap')
    ax3.scatter(dm[:, 0], dm[:, 1])

    plt.show()



