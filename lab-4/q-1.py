# TODO: - Fix the bug: nan values generated in the responses, which translates to the features and similarity

import cv2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def build_gabor_filters(num_orientations=4, num_scales=3, kernel_size=11):
    filters = []
    for theta in np.arange(0, np.pi, np.pi / num_orientations):
        for scale in range(num_scales):
            sigma = 2 * np.pi * scale / num_scales
            lamda = np.pi / 4
            gamma = 0.25
            psi = 0
            kernel = cv2.getGaborKernel(
                (kernel_size, kernel_size), sigma, theta, lamda, gamma, psi
            )
            filters.append(kernel)
    return filters


def apply_gabor_filters(image, filters):
    responses = []
    for kernel in filters:
        response = cv2.filter2D(image, cv2.CV_32F, kernel)
        responses.append(response)
    return responses


def extract_features(responses, x, y, window_size=11):
    features = []
    half_window = window_size // 2
    height, width = responses[0].shape

    # if (
    # x - half_window < 0
    # or x + half_window >= height
    # or y - half_window < 0
    # or y + half_window >= width
    # ):
    # return np.array([np.nan] * 4 * len(responses))
    # return np.array([np.nan] * len(responses))

    for response in responses:
        feature = response[
            x - half_window : x + half_window + 1, y - half_window : y + half_window + 1
        ]

        mean = np.mean(feature)
        # std = np.std(feature)
        # skewness = stats.skew(feature.flatten())
        # kurtosis = stats.kurtosis(feature.flatten())

        # features.extend([mean, std, skewness, kurtosis])
        features.append(mean)

    return np.array(features)


def segment_features(image, selected_roi, window_size=11, threshold=0.01):
    filters = build_gabor_filters()
    responses = apply_gabor_filters(image, filters)
    reference_responses = apply_gabor_filters(selected_roi, filters)
    reference_features = np.zeros(
        (len(selected_roi), len(selected_roi[0]), len(filters))
    )
    print(f"Reference responses: {reference_responses}")
    for x in range(len(selected_roi)):
        for y in range(len(selected_roi[0])):
            reference_features[x, y] = extract_features(
                reference_responses, x, y, window_size
            )
    print(f"Reference features: {reference_features}")
    reference_feature = np.mean(reference_features, axis=(0, 1))

    height, width = image.shape
    mask = np.zeros((height, width))
    i = 0

    for x in range(height):
        for y in range(width):
            current_features = extract_features(responses, x, y, window_size)
            similarity = np.dot(reference_feature, current_features) / (
                np.linalg.norm(reference_feature) * np.linalg.norm(current_features)
            )
            if similarity > threshold:
                mask[x, y] = 1

            i += 1
            # print(f"Processed {i} pixels of {height * width}", end="\r")
    return mask


def main():
    # select a point in the image interactively
    img = cv2.imread("assets/a.jpg", 0)
    cv2.namedWindow("Select Point")
    cv2.imshow("Select Point", img)
    roi = cv2.selectROI("Select Point", img)
    selected_roi = img[
        int(roi[1]) : int(roi[1] + roi[3]), int(roi[0]) : int(roi[0] + roi[2])
    ]
    cv2.destroyAllWindows()

    # segment the image using the selected point
    mask = segment_features(img, selected_roi, threshold=0.2)
    print(f"Segmented image: {mask}")
    cv2.imshow("Segmented Image", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
