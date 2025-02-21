import cv2
import numpy as np


def iskalnik_4_kotnikov(img: np.ndarray) -> list[np.ndarray]:
    # Normalize image to uint8 if it's in float32 format
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        raise ValueError("Unsupported image data type. Please provide an image of type uint8 or float32/float64.")

    # Check if the image is truly grayscale (all channels are identical or single-channel)
    if len(img.shape) == 2 or (
            len(img.shape) == 3 and np.allclose(img[:, :, 0], img[:, :, 1]) and np.allclose(img[:, :, 1],
                                                                                            img[:, :, 2])):
        print("Processing a black-and-white image...")
        # Load as grayscale
        img_gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold the image to isolate white shapes
        _, mask = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)

        # Use the mask to preserve only white shapes
        result = cv2.bitwise_and(img_gray, img_gray, mask=mask)

    else:  # Colored image
        print("Processing a colorful image...")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define gray background thresholds in HSV
        lower_gray = np.array([0, 0, 100])  # Adjust as needed
        upper_gray = np.array([180, 50, 255])

        # Create a mask for the gray background
        mask = cv2.inRange(img_hsv, lower_gray, upper_gray)

        # Invert the mask to get colorful shapes
        mask_inv = cv2.bitwise_not(mask)

        # Use the mask to keep only the colorful shapes
        result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_inv)

    # Convert the result to grayscale for further processing
    if len(result.shape) == 3:  # Color image
        slika = result.mean(2)
    else:  # Grayscale image
        slika = result

    T1 = 0.4  # * 255   Scale threshold to match uint8 range
    T2 = 0.7  # * 255   Scale threshold to match uint8 range
    maska_s1 = slika < T1
    maska_s3 = slika > T2
    contours, _ = cv2.findContours(np.uint8(maska_s3), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    contours_large = [c for c in contours if cv2.contourArea(c, False) > 100.]
    contours_neg_orient = [c for c in contours_large if cv2.contourArea(c, True) < 0.]
    contours_approx = [cv2.approxPolyDP(c, 2., True) for c in contours_neg_orient]
    rectangular_contours = [c for c in contours_approx if len(c) == 4]
    # Generate a list of 4x2 corner coordinates for rectangular contours
    corner_coordinates = [np.array([[pt[1], pt[0]] for pt in c[:, 0]]) for c in rectangular_contours]

    return corner_coordinates

