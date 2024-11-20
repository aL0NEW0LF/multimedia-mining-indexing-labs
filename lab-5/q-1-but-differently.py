from math import log10, copysign
import cv2
import numpy as np


def compare_hu_moments(
    imageA: np.ndarray, imageB: np.ndarray, method: int = 1
) -> np.float64:
    """Compare the Hu moments of two images

    Args:
        imageA (numpy.ndarray): the first image
        imageB (numpy.ndarray): the second image
        method (int, optional): the method to use. Defaults to 1.

    Raises:
        Exception: False Zero: One of the images has no moments

    Returns:
        numpy.float64: the distance between the two images
    """
    eps = 1e-5
    d = 0
    huMomentsA = cv2.HuMoments(cv2.moments(imageA)).flatten()
    huMomentsB = cv2.HuMoments(cv2.moments(imageB)).flatten()

    if np.any(np.abs(huMomentsA) > 0) != np.any(np.abs(huMomentsB) > 0):
        raise Exception("False Zero: One of the images has no moments")

    if method == 1:
        for i in range(0, 7):
            if abs(huMomentsA[i]) > eps and abs(huMomentsB[i]) > eps:
                d += abs(
                    (1.0 / (np.sign(huMomentsB[i]) * log10(abs(huMomentsB[i]))))
                    - (1.0 / (np.sign(huMomentsA[i]) * log10(abs(huMomentsA[i]))))
                )

    elif method == 2:
        for i in range(0, 7):
            if abs(huMomentsA[i]) > eps and abs(huMomentsB[i]) > eps:
                d += abs(
                    (np.sign(huMomentsB[i]) * log10(abs(huMomentsB[i])))
                    - (np.sign(huMomentsA[i]) * log10(abs(huMomentsA[i])))
                )

    elif method == 3:
        for i in range(0, 7):
            if abs(huMomentsA[i]) > eps and abs(huMomentsB[i]) > eps:
                mmm = abs(
                    (
                        (np.sign(huMomentsA[i]) * log10(abs(huMomentsA[i])))
                        - (np.sign(huMomentsB[i]) * log10(abs(huMomentsB[i])))
                    )
                    / (np.sign(huMomentsA[i]) * log10(abs(huMomentsA[i])))
                )
                if mmm > d:
                    d = mmm

    return d


if __name__ == "__main__":
    imageA = cv2.imread("assets/formes/apple-1.png", cv2.IMREAD_GRAYSCALE)
    _, imA = cv2.threshold(imageA, 128, 255, cv2.THRESH_BINARY)
    imageB = cv2.imread("assets/formes/apple-2.png", cv2.IMREAD_GRAYSCALE)
    _, imB = cv2.threshold(imageB, 128, 255, cv2.THRESH_BINARY)

    print("d1:", compare_hu_moments(imA, imB), end="\n")
    print("d2:", compare_hu_moments(imA, imB, 2), end="\n")
    print("d3:", compare_hu_moments(imA, imB, 3), end="\n")

    """
    cv2.namedWindow("comparison", cv2.WINDOW_NORMAL)
    canvas_height = max(imA.shape[0], imB.shape[0]) + 50
    canvas_width = imA.shape[1] + imB.shape[1]
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    canvas[0 : imA.shape[0], 0 : imA.shape[1]] = imA
    canvas[0 : imB.shape[0], imA.shape[1] :] = imB

    if (
        compare_hu_moments(imA, imB) < 0.01
        and compare_hu_moments(imA, imB, 2) < 0.01
        and compare_hu_moments(imA, imB, 3) < 0.01
    ):
        cv2.putText(
            canvas,
            "The images are similar",
            (10, canvas_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
    else:
        cv2.putText(
            canvas,
            "The images are different",
            (10, canvas_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

    cv2.imshow("comparison", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
