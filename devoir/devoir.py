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
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

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

    print("Retrieving dominant colors and historgrams for", len(onlyfiles), "images")

    dominant_colors = {}
    histograms = {}

    color = ("b", "g", "r")
    imHistW = 512
    imHistH = 400
    bord = 10
    binHistW = (imHistW - 2 * bord) / 256
    histImage = np.ones((imHistH, imHistW, 3), dtype=np.uint8) * 255

    for file in onlyfiles:
        image = cv2.imread("devoir/assets/" + file, 1)
        imageH, imageW = image.shape[:2]
        combinedW = imageW + imHistW
        combinedH = max(imageH, imHistH)
        combinedImage = np.ones((combinedH, combinedW, 3), dtype=np.uint8) * 255

        # Place the image on the left
        combinedImage[:imageH, :imageW] = image

        # Create the histogram image
        histImage = np.ones((imHistH, imHistW, 3), dtype=np.uint8) * 255

        cv2.line(
            histImage,
            (bord, imHistH - bord),
            (bord, bord),
            (0, 0, 0),
            2,
        )
        cv2.line(
            histImage,
            (bord, imHistH - bord),
            (imHistW - bord, imHistH - bord),
            (0, 0, 0),
            2,
        )

        histograms[file] = {}

        for i, col in enumerate(color):
            histr = cv2.calcHist([image], [i], None, [256], [0, 256], accumulate=False)

            histograms[file][col] = histr.tolist()
            # Normalize the histogram
            cv2.normalize(
                histr,
                histr,
                alpha=0,
                beta=imHistH - 2 * bord,
                norm_type=cv2.NORM_MINMAX,
            )

            for j in range(1, 256):
                cv2.line(
                    histImage,
                    (
                        int(bord + (j - 1) * binHistW),
                        int(imHistH - bord - histr[j - 1]),
                    ),
                    (int(bord + j * binHistW), int(imHistH - bord - histr[j])),
                    (
                        255 if col == "b" else 0,
                        255 if col == "g" else 0,
                        255 if col == "r" else 0,
                    ),
                    1,
                )

        # Place the histogram on the right
        combinedImage[:imHistH, imageW : imageW + imHistW] = histImage

        window_name = f"Image and Histogram for {file}"
        cv2.imshow(window_name, combinedImage)

        # Move the window to a specific position (e.g., x=100, y=100)
        cv2.moveWindow(window_name, 0, 0)

        dominant_colors[file] = find_dominant_colors(image).tolist()

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("Calculating distances")

    index = random.randint(0, 4)
    distances = {}

    while index < len(onlyfiles):
        distances[onlyfiles[index]] = {}
        base_image = cv2.imread("devoir/assets/" + onlyfiles[index], 1)
        base_lab = cv2.cvtColor(base_image, cv2.COLOR_BGR2LAB)
        base_hist = cv2.calcHist(
            [base_lab], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
        )
        cv2.normalize(base_hist, base_hist, 0, 1, cv2.NORM_MINMAX)

        for i in range(len(onlyfiles)):
            if index != i:
                distances[onlyfiles[index]][onlyfiles[i]] = {}
                distances[onlyfiles[index]][onlyfiles[i]][
                    "dominant_colors_comparison"
                ] = float(
                    chi2_distance(
                        np.array(dominant_colors[onlyfiles[index]]),
                        np.array(dominant_colors[onlyfiles[i]]),
                    )
                )

                image = cv2.imread("devoir/assets/" + onlyfiles[i], 1)
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                hist = cv2.calcHist(
                    [lab], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
                )
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                distances[onlyfiles[index]][onlyfiles[i]]["histogram_comparison"] = (
                    cv2.compareHist(base_hist, hist, cv2.HISTCMP_CHISQR)
                )
                distances[onlyfiles[index]][onlyfiles[i]]["general_comparison"] = float(
                    (
                        distances[onlyfiles[index]][onlyfiles[i]][
                            "dominant_colors_comparison"
                        ]
                        + distances[onlyfiles[index]][onlyfiles[i]][
                            "histogram_comparison"
                        ]
                    )
                    / 2
                )

                base_image_rgb = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
                comparison_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Create a white canvas
                canvas_height = (
                    max(base_image_rgb.shape[0], comparison_image_rgb.shape[0]) + 100
                )
                canvas_width = base_image_rgb.shape[1] + comparison_image_rgb.shape[1]
                canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

                # Place the base image on the canvas
                canvas[: base_image_rgb.shape[0], : base_image_rgb.shape[1]] = (
                    base_image_rgb
                )

                # Place the comparison image on the canvas
                canvas[: comparison_image_rgb.shape[0], base_image_rgb.shape[1] :] = (
                    comparison_image_rgb
                )

                # Add text for the distances
                text = (
                    f"Chi2 Distance: {distances[onlyfiles[index]][onlyfiles[i]]['dominant_colors_comparison']}\n"
                    f"Histogram Distance: {distances[onlyfiles[index]][onlyfiles[i]]['histogram_comparison']}\n"
                    f"General Distance: {distances[onlyfiles[index]][onlyfiles[i]]['general_comparison']}"
                )
                y0, dy = (
                    max(base_image_rgb.shape[0], comparison_image_rgb.shape[0]) + 20,
                    30,
                )
                for i, line in enumerate(text.split("\n")):
                    y = y0 + i * dy
                    cv2.putText(
                        canvas,
                        line,
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

                # Display the canvas
                cv2.imshow("Images and Distances", canvas)
                cv2.moveWindow(
                    "Images and Distances", 100, 100
                )  # Move the window to a specific position
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        index += 5

    final_output = {
        "dominant_colors": dominant_colors,
        "histograms": histograms,
        "distances": distances,
    }

    with open("devoir/output.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print("Done! Output saved to devoir/output.json")


if __name__ == "__main__":
    main()
