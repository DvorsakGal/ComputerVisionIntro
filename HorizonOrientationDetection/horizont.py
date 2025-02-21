import numpy as np
import cv2
from scipy import ndimage

def orientacija_horizonta(slika: np.ndarray) -> float:
    # Normalize the image if it's in uint8 format (0-255 range)
    if slika.dtype == np.uint8:
        slika = slika / 255.0

    image = slika.mean(2)
    sigma = 14
    velikost_jedra = int(2 * sigma)
    # koordinate jedra po x dimenziji
    x = np.arange(-velikost_jedra, velikost_jedra + 1)
    # koordinate jedra v obeh dimenzijah v 2d matrikah
    X, Y = np.meshgrid(x, x)
    jedro_tocke = np.array([X.ravel(), Y.ravel()])  # 2 x P
    C = np.eye(2) * 1 / sigma ** 2

    jedro_gauss = np.exp(-1 * (jedro_tocke.T.dot(C) * jedro_tocke.T).sum(1)).reshape(X.shape)
    slika_conv = ndimage.convolve(image, jedro_gauss, mode="nearest", cval=0.0)

    jedro_sobel_dx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    jedro_sobel_dy = jedro_sobel_dx.T

    slika_dx = ndimage.convolve(slika_conv, jedro_sobel_dx, mode='nearest', cval=0.0)
    slika_dy = ndimage.convolve(slika_conv, jedro_sobel_dy, mode='nearest', cval=0.0)

    slika_rob_mag = (slika_dx ** 2 + slika_dy ** 2) ** 0.5  # samo pitagorov izrek
    slika_rob_smer = np.arctan2(slika_dy, slika_dx)

    hist_bins = np.linspace((-np.pi) / 2, (np.pi) / 2, 80)
    hist_smeri, bin_edges = np.histogram(slika_rob_smer, bins=hist_bins, weights=slika_rob_mag)
    max_bin_index = np.argmax(hist_smeri)
    # Compute the corresponding orientation value (average of the bin)
    horizon_orientation = ((bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2)

    if abs(horizon_orientation) <= np.pi / 8:
        if horizon_orientation > 0:
            horizon_orientation += np.pi / 4
        else:
            horizon_orientation -= np.pi / 4
    else:
        if horizon_orientation < 0:
            horizon_orientation += np.pi / 2
        else:
            horizon_orientation -= np.pi / 2

    return horizon_orientation