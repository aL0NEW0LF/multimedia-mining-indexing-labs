from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np

def chi2_distance(histA, histB, eps = 1e-10):
	d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
	return d

img_1 = cv2.cvtColor(cv2.imread("assets/PRDR31330275638_1.jpeg"), cv2.COLOR_BGR2RGB)
img_2 = cv2.cvtColor(cv2.imread("assets/PRDR3873521911_1.jpeg"), cv2.COLOR_BGR2RGB)

hist_1 = cv2.calcHist([img_1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
hist_2 = cv2.calcHist([img_2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

hist_1 = cv2.normalize(hist_1, hist_1).flatten()
hist_2 = cv2.normalize(hist_2, hist_2).flatten()

d = chi2_distance(hist_1, hist_2)

fig = plt.figure("Images")
images = ("Image 1", img_1), ("Image 2", img_2)

for (i, (name, image)) in enumerate(images):
	ax = fig.add_subplot(1, 2, i + 1)
	plt.imshow(image)
	plt.axis("off")

plt.suptitle(f"Chi2 distance: {d}")
print("Chi2 distance: ", d)
plt.show()