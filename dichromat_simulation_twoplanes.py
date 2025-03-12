# MIT License
# Copyright (c) 2025 Paul Maximov <pmaximov@iitp.ru>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

def plane_normal_from_vectors(anchor1: np.ndarray,
                              anchor2: np.ndarray) -> np.ndarray:
    """ Function for calculation of normal unit vector of dichromatic projection
    plane from origin and two anchor points (Vienot et al., 1999) """
    plane_normal = np.empty(3)
    plane_normal[0] = anchor1[1]*anchor2[2] - anchor1[2]*anchor2[1]
    plane_normal[1] = anchor1[2]*anchor2[0] - anchor1[0]*anchor2[2]
    plane_normal[2] = anchor1[0]*anchor2[1] - anchor1[1]*anchor2[0]
    return plane_normal / np.linalg.norm(plane_normal)

def simmatr_from_points(anchor1: np.ndarray, anchor2: np.ndarray,
                        confusionvec: np.ndarray) -> np.ndarray:
    matrix = np.vstack((confusionvec, anchor1, anchor2))
    matrix_inv = np.linalg.inv(matrix)
    initial_values = np.vstack((np.zeros(3), anchor1, anchor2))
    return (matrix_inv @ initial_values).T

def confusionvector_from_hue(hue_angle: np.float64) -> np.ndarray:
    vec = np.array([np.cos(np.pi * hue_angle / 180),
                    np.sin(np.pi * hue_angle / 180),
                    1])
    matrix = np.array([[2, 0, 1],
                       [-1, np.sqrt(3), 1],
                       [-1, -np.sqrt(3), 1]])
    return 1/3 * matrix @ vec

LMS_from_XYZ = np.array([[ 0.15514, 0.54312, -0.03286],
                         [-0.15514, 0.45684,  0.03286],
                         [ 0,       0,        0.01608]])
XYZ_from_RGB = np.array([[0.409568,  0.355041, 0.179167],
                         [0.213389,  0.706743, 0.079868],
                         [0.0186297, 0.114620, 0.912367]])
RGB_from_XYZ = np.linalg.inv(XYZ_from_RGB)

# Let us norm matrices for our convenience. Assume that linearRGB of (1,1,1)
# causes LMS of (1,1,1)
weights = LMS_from_XYZ @ XYZ_from_RGB @ np.ones(3)
LMS_from_XYZ_n = (LMS_from_XYZ.T / weights).T
XYZ_from_LMS_n = np.linalg.inv(LMS_from_XYZ_n)

# Calculate the single transformations RGB<->LMS
LMS_from_RGB_n = LMS_from_XYZ_n @ XYZ_from_RGB
RGB_from_LMS_n = RGB_from_XYZ @ XYZ_from_LMS_n

def get_simulation_matrix_by_anchors(hue_angle: np.float64,
                          anchor1: np.ndarray,
                          anchor2: np.ndarray) -> np.ndarray:
    Anchor_W_ort = np.ones(3) / np.sqrt(3)
    N_1plane_RGB_ort = plane_normal_from_vectors(Anchor_W_ort, anchor1)
    N_2plane_RGB_ort = plane_normal_from_vectors(Anchor_W_ort, anchor2)

    dichvec_LMS_ort = confusionvector_from_hue(hue_angle)
    dichvec_RGB_ort = RGB_from_LMS_n @ dichvec_LMS_ort
    dichvec_RGB_ort = dichvec_RGB_ort / np.linalg.norm(dichvec_RGB_ort)

    sinfi_RGB_1plane = abs(dichvec_RGB_ort @ N_1plane_RGB_ort)
    sinfi_RGB_2plane = abs(dichvec_RGB_ort @ N_2plane_RGB_ort)

    if sinfi_RGB_1plane > sinfi_RGB_2plane:
        simmatr = simmatr_from_points(Anchor_W_ort, anchor1,
                                      dichvec_RGB_ort)
    else:
        simmatr = simmatr_from_points(Anchor_W_ort, anchor2,
                                      dichvec_RGB_ort)
    return simmatr

def get_simulation_matrix(hue_angle: np.float64):
    # Anchors for simulation planes
    Anchor_B = np.array([0, 0, 1]) # Blue anchor color for white-blue-yellow plane
    Anchor_tritan = np.array([0, 1, 0.5]) # Anchor color for perpendicular plane

    return get_simulation_matrix_by_anchors(hue_angle, Anchor_B, Anchor_tritan)

def simulate_sRGB(matrix: np.ndarray, image_sRGB: np.ndarray) -> np.ndarray:
    def sRGB_from_linearRGB(im):
        """Converts linearRGB to sRGB. Made on the basis of
        daltonlens.convert.sRGB_from_linearRGB, clipping operation was removed"""
        out = np.empty_like(im)
        small_mask = im < 0.0031308
        large_mask = np.logical_not(small_mask)
        out[small_mask] = im[small_mask] * 12.92
        out[large_mask] = np.power(im[large_mask], 1.0 / 2.4) * 1.055 - 0.055
        return out

    def linearRGB_from_sRGB(im):
        #Converts sRGB to linearRGB (from daltonlens.convert.sRGB_from_linearRGB)
        out = np.empty_like(im)
        small_mask = im < 0.04045
        large_mask = np.logical_not(small_mask)
        out[small_mask] = im[small_mask] / 12.92
        out[large_mask] = np.power((im[large_mask] + 0.055) / 1.055, 2.4)
        return out

    image_linRGB = linearRGB_from_sRGB(image_sRGB)
    image_linRGB_sim = image_linRGB @ matrix.T
    return np.clip(sRGB_from_linearRGB(image_linRGB_sim), 0, 1)


####################   USAGE EXAMPLE  #####################
import PIL.Image

filename = 'source.png'
im = np.asarray(PIL.Image.open(filename), dtype=np.float64) / 255

angle = 40

simmatr = get_simulation_matrix(angle)
ims = simulate_sRGB(simmatr, im)

PIL.Image.fromarray(np.asarray(ims * 255, dtype=np.uint8)).save('result.png')
