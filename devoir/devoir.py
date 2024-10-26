import json
import cv2
import numpy as np
from sklearn.cluster import KMeans
import math
from os import listdir
from os.path import isfile, join
import random
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import PyQt5
import PyQt5.QtWidgets
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class ScrollableWindow(QtWidgets.QMainWindow):
    def __init__(self, fig):
        self.qapp = QtWidgets.QApplication([])

        QtWidgets.QMainWindow.__init__(self)
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLayout(QtWidgets.QVBoxLayout())
        self.widget.layout().setContentsMargins(0, 0, 0, 0)
        self.widget.layout().setSpacing(0)

        self.fig = fig
        self.canvas = FigureCanvas(self.fig)
        self.canvas.draw()
        self.scroll = QtWidgets.QScrollArea(self.widget)
        self.scroll.setWidget(self.canvas)

        self.nav = NavigationToolbar(self.canvas, self.widget)
        self.widget.layout().addWidget(self.nav)
        self.widget.layout().addWidget(self.scroll)


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    # return the histogram
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    # loop over the percentage of each cluster and the color of
    # each cluster
    for percent, color in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(
            bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1
        )
        startX = endX

    # return the bar chart
    return bar


def chi2_distance(dominant_colors_A, dominant_colors_B, eps=1e-10):
    d = (
        np.sum(
            [
                ((a - b) ** 2) / (a + b + eps)
                for (a, b) in zip(dominant_colors_A, dominant_colors_B)
            ]
        )
    ) / (2 * dominant_colors_A.shape[0])

    return d


def find_dominant_colors(image):
    img = cv2.imread(image, 1)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    img_flat = img_lab.reshape((-1, 3))

    unique_colors, counts = np.unique(img_flat, return_counts=True, axis=0)

    order = np.argsort(counts)[::-1]
    unique_colors = unique_colors[order]
    counts = counts[order]

    pourcenetage = math.floor(0.95 * img_flat.shape[0])
    unique_colors_new = []
    colors_new = []
    sum = 0
    for i in range(len(unique_colors)):
        sum += counts[i]
        if sum > pourcenetage:
            break
        unique_colors_new.append(unique_colors[i])
        colors_new.append(counts[i])

    kmeans = KMeans(n_clusters=16)
    kmeans.fit(unique_colors_new)
    dominant_colors = kmeans.cluster_centers_

    return dominant_colors


def main():
    print("Running main")

    onlyfiles = [
        f for f in listdir("devoir/assets/") if isfile(join("devoir/assets/", f))
    ]

    onlyfiles.sort()

    print("Retrieving dominant colors for", len(onlyfiles), "images")

    dominant_colors = {}

    for image in onlyfiles:
        # string of the list
        dominant_colors[image] = find_dominant_colors("devoir/assets/" + image).tolist()

    # fig, axs = plt.subplots(10, 2, figsize=(20, 30))
    # plt.subplots_adjust(hspace=0.5, wspace=0.5)
    # plt.title("Histograms")
    # plt.xlabel("Bins")
    # plt.ylabel("Number of Pixels")

    # for i, image in enumerate(onlyfiles):
    # img = cv2.imread("devoir/assets/" + image, 1)
    # chans = cv2.split(img)
    # colors = ("b", "g", "r")

    # for chan, color in zip(chans, colors):
    # hist = cv2.calcHist([chan], [0], None, [256], [0, 256])

    # axs[int(i / 2)][i % 2].plot(hist, color=color)
    # axs[int(i / 2)][i % 2].set_xlim([0, 256])
    # axs[int(i / 2)][i % 2].set_title(image)

    print("Calculating distances")

    index = random.randint(0, 4)
    distances = {}

    while index < len(onlyfiles):
        distances[onlyfiles[index]] = {}
        for i in range(len(onlyfiles)):
            if index != i:
                print("Comparing", onlyfiles[index], "and", onlyfiles[i])
                distances[onlyfiles[index]][onlyfiles[i]] = chi2_distance(
                    np.array(dominant_colors[onlyfiles[index]]),
                    np.array(dominant_colors[onlyfiles[i]]),
                )

        index += 5

    final_output = {"dominant_colors": dominant_colors, "distances": distances}

    with open("devoir/output.json", "w") as f:
        json.dump(final_output, f, indent=4)

    print("Done! Output saved to devoir/output.json")

    # TO FIX: segmentation fault (core dumped)  python -m pdb devoir/devoir.py
    # a = ScrollableWindow(fig)
    # a.showMaximized()
    # ret = a.qapp.exec_()
    # exit(ret)


if __name__ == "__main__":
    main()
