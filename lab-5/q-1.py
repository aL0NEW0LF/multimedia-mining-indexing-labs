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
    anyA = False
    anyB = False
    d = 0
    huMomentsA = cv2.HuMoments(cv2.moments(imageA)).flatten()
    huMomentsB = cv2.HuMoments(cv2.moments(imageB)).flatten()

    if method == 1:
        for i in range(0, 7):
            amA = abs(huMomentsA[i])
            amB = abs(huMomentsB[i])

            if amA > 0:
                anyA = True
            if amB > 0:
                anyB = True

            if huMomentsA[i] > 0:
                smA = 1
            elif huMomentsA[i] < 0:
                smA = -1
            else:
                smA = 0

            if huMomentsB[i] > 0:
                smB = 1
            elif huMomentsB[i] < 0:
                smB = -1
            else:
                smB = 0

            if amA > eps and amB > eps:
                amA = 1.0 / (smA * log10(amA))
                amB = 1.0 / (smB * log10(amB))
                d += abs(amB - amA)

    elif method == 2:
        for i in range(0, 7):
            amA = abs(huMomentsA[i])
            amB = abs(huMomentsB[i])

            if amA > 0:
                anyA = True
            if amB > 0:
                anyB = True

            if huMomentsA[i] > 0:
                smA = 1
            elif huMomentsA[i] < 0:
                smA = -1
            else:
                smA = 0

            if huMomentsB[i] > 0:
                smB = 1
            elif huMomentsB[i] < 0:
                smB = -1
            else:
                smB = 0

            if amA > eps and amB > eps:
                amA = smA * log10(amA)
                amB = smB * log10(amB)
                d += abs(amB - amA)

    elif method == 3:
        for i in range(0, 7):
            amA = abs(huMomentsA[i])
            amB = abs(huMomentsB[i])

            if amA > 0:
                anyA = True
            if amB > 0:
                anyB = True

            if huMomentsA[i] > 0:
                smA = 1
            elif huMomentsA[i] < 0:
                smA = -1
            else:
                smA = 0

            if huMomentsB[i] > 0:
                smB = 1
            elif huMomentsB[i] < 0:
                smB = -1
            else:
                smB = 0

            if amA > eps and amB > eps:
                amA = smA * log10(amA)
                amB = smB * log10(amB)
                mmm = abs((amA - amB) / amA)
                if mmm > d:
                    d = mmm

    if anyA != anyB:
        raise Exception("False Zero")

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
