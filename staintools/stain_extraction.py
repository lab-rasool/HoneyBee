from abc import ABC, abstractmethod
import numpy as np
import cv2 as cv

from preprocessing import is_uint8_image
# import spams


def get_sign(x):
    """
    Returns the sign of x.

    :param x: A scalar x.
    :return: The sign of x.
    """

    if x > 0:
        return +1
    elif x < 0:
        return -1
    elif x == 0:
        return 0


def normalize_matrix_rows(A):
    """
    Normalize the rows of an array.

    :param A: An array.
    :return: Array with rows normalized.
    """
    return A / np.linalg.norm(A, axis=1)[:, None]


def convert_RGB_to_OD(I):
    """
    Convert from RGB to optical density (OD_RGB) space.

    RGB = 255 * exp(-1*OD_RGB).

    :param I: Image RGB uint8.
    :return: Optical denisty RGB image.
    """
    mask = I == 0
    I[mask] = 1
    return np.maximum(-1 * np.log(I / 255), 1e-6)


def convert_OD_to_RGB(OD):
    """
    Convert from optical density (OD_RGB) to RGB.

    RGB = 255 * exp(-1*OD_RGB)

    :param OD: Optical denisty RGB image.
    :return: Image RGB uint8.
    """
    assert OD.min() >= 0, "Negative optical density."
    OD = np.maximum(OD, 1e-6)
    return (255 * np.exp(-1 * OD)).astype(np.uint8)


# def get_concentrations(I, stain_matrix, regularizer=0.01):
#     """
#     Estimate concentration matrix given an image and stain matrix.

#     :param I:
#     :param stain_matrix:
#     :param regularizer:
#     :return:
#     """
#     OD = convert_RGB_to_OD(I).reshape((-1, 3))
#     return spams.lasso(X=OD.T, D=stain_matrix.T, mode=2, lambda1=regularizer, pos=True).toarray().T


class TissueMaskException(Exception):
    pass


class ABCStainExtractor(ABC):
    @staticmethod
    @abstractmethod
    def get_stain_matrix(I):
        """
        Estimate the stain matrix given an image.

        :param I:
        :return:
        """


class ABCTissueLocator(ABC):
    @staticmethod
    @abstractmethod
    def get_tissue_mask(I):
        """
        Get a boolean tissue mask.

        :param I:
        :return:
        """


class LuminosityThresholdTissueLocator(ABCTissueLocator):
    @staticmethod
    def get_tissue_mask(I, luminosity_threshold=0.8):
        """
        Get a binary mask where true denotes pixels with a luminosity less than the specified threshold.
        Typically we use to identify tissue in the image and exclude the bright white background.

        :param I: RGB uint 8 image.
        :param luminosity_threshold: Luminosity threshold.
        :return: Binary mask.
        """
        assert is_uint8_image(I), "Image should be RGB uint8."
        I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
        L = I_LAB[:, :, 0] / 255.0  # Convert to range [0,1].
        mask = L < luminosity_threshold

        # Check it's not empty
        if mask.sum() == 0:
            raise TissueMaskException("Empty tissue mask computed")

        return mask


class MacenkoStainExtractor(ABCStainExtractor):
    @staticmethod
    def get_stain_matrix(I, luminosity_threshold=0.8, angular_percentile=99):
        """
        Stain matrix estimation via method of:
        M. Macenko et al. 'A method for normalizing histology slides for quantitative analysis'

        :param I: Image RGB uint8.
        :param luminosity_threshold:
        :param angular_percentile:
        :return:
        """
        assert is_uint8_image(I), "Image should be RGB uint8."
        # Convert to OD and ignore background
        tissue_mask = LuminosityThresholdTissueLocator.get_tissue_mask(
            I, luminosity_threshold=luminosity_threshold
        ).reshape((-1,))
        OD = convert_RGB_to_OD(I).reshape((-1, 3))
        OD = OD[tissue_mask]

        # Eigenvectors of cov in OD space (orthogonal as cov symmetric)
        _, V = np.linalg.eigh(np.cov(OD, rowvar=False))

        # The two principle eigenvectors
        V = V[:, [2, 1]]

        # Make sure vectors are pointing the right way
        if V[0, 0] < 0:
            V[:, 0] *= -1
        if V[0, 1] < 0:
            V[:, 1] *= -1

        # Project on this basis.
        That = np.dot(OD, V)

        # Angular coordinates with repect to the prinicple, orthogonal eigenvectors
        phi = np.arctan2(That[:, 1], That[:, 0])

        # Min and max angles
        minPhi = np.percentile(phi, 100 - angular_percentile)
        maxPhi = np.percentile(phi, angular_percentile)

        # the two principle colors
        v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
        v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))

        # Order of H and E.
        # H first row.
        if v1[0] > v2[0]:
            HE = np.array([v1, v2])
        else:
            HE = np.array([v2, v1])

        return normalize_matrix_rows(HE)
