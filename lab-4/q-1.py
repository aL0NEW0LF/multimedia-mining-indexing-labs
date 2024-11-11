import cv2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def build_gabor_filters(num_orientations=4, num_scales=3, kernel_size=31):
    filters = []
    for theta in range(num_orientations):
        theta = (theta * np.pi) / num_orientations
        for scale in range(num_scales):
            sigma = 3.0
            lamda = 4.0 * (2**scale)
            gamma = 0.5
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


def extract_features(responses, x, y, window_size=9):
    features = []
    half_window = window_size // 2
    height, width = responses[0].shape

    if (
        x - half_window < 0
        or x + half_window >= height
        or y - half_window < 0
        or y + half_window >= width
    ):
        return np.array([np.nan] * 4 * len(responses))

    for response in responses:
        feature = response[
            x - half_window : x + half_window + 1, y - half_window : y + half_window + 1
        ]

        mean = np.mean(feature)
        std = np.std(feature)
        skewness = stats.skew(feature.flatten())
        kurtosis = stats.kurtosis(feature.flatten())

        features.extend([mean, std, skewness, kurtosis])

    return np.array(features)


def segment_features(image, selected_x, selected_y, window_size=5, threshold=0.5):
    filters = build_gabor_filters()
    responses = apply_gabor_filters(image, filters)
    ref_features = extract_features(responses, selected_x, selected_y, window_size)

    height, width = image.shape
    mask = np.zeros((height, width))
    i = 0

    for x in range(height):
        for y in range(width):
            current_features = extract_features(responses, x, y, window_size)
            similarity = np.corrcoef(ref_features, current_features)[0, 1]

            if similarity > threshold:
                mask[x, y] = 1

            i += 1
            print(f"Processed {i} pixels of {height * width}", end="\r")
    return mask


def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param["x"] = x
        param["y"] = y
        param["selected"] = True


def main():
    # select a point in the image interactively
    img = cv2.imread("assets/a.jpg", 0)
    cv2.namedWindow("Select Point")
    cv2.imshow("Select Point", img)
    params = {"x": -1, "y": -1, "selected": False}
    cv2.setMouseCallback("Select Point", select_point, params)
    while not params["selected"]:
        if cv2.waitKey(20) & 0xFF == 27:
            break
    print(f"Selected point: ({params['x']}, {params['y']})")
    cv2.destroyAllWindows()

    # segment the image using the selected point
    mask = segment_features(img, params["x"], params["y"], window_size=5, threshold=0.8)
    print(f"Segmented image: {mask}")
    cv2.imshow("Segmented Image", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
