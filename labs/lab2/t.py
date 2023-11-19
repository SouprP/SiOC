import numpy as np
import cv2
from func_timer import timer

@timer
def rotate_nearest_neighbor(image, angle_degrees):
    """
    Rotate an image using nearest-neighbor interpolation.

    Parameters:
    - image: 3D NumPy array representing the image (height, width, channels).
    - angle_degrees: Rotation angle in degrees.

    Returns:
    - Rotated image.
    """

    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Get image dimensions
    height, width, channels = image.shape

    # Calculate the rotation matrix
    rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                [np.sin(angle_radians), np.cos(angle_radians)]])

    # Calculate the center of the image
    center = np.array([width / 2, height / 2])

    # Calculate the new dimensions to fit the rotated image
    new_width = int(np.ceil(width * np.abs(np.cos(angle_radians)) + height * np.abs(np.sin(angle_radians))))
    new_height = int(np.ceil(width * np.abs(np.sin(angle_radians)) + height * np.abs(np.cos(angle_radians))))

    # Calculate the translation matrix to keep the rotated image centered
    translation_matrix = np.array([[1, 0, (new_width - width) / 2],
                                   [0, 1, (new_height - height) / 2]])
    
    # Combine the rotation and translation matrices
    transformation_matrix = np.hstack([rotation_matrix, translation_matrix[:, :-1]])

    # Add a row [0, 0, 1] to make the matrix square
    transformation_matrix = np.vstack([transformation_matrix, np.array([0, 0, 1])])

    # Perform rotation using nearest-neighbor interpolation
    rotated_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            # Calculate the original coordinates in the unrotated image
            original_coords = np.dot(np.linalg.inv(transformation_matrix),
                                     np.array([x - center[0], y - center[1], 1]))

            # Extract the integer part of the coordinates
            x_int, y_int = original_coords[:2].astype(int)

            # Check if the original coordinates are within the image boundaries
            if 0 <= x_int < width and 0 <= y_int < height:
                rotated_image[y, x, :] = image[y_int, x_int, :]

    return rotated_image

@timer
def rotate_bilinear(image, angle_degrees):
    """
    Rotate an image using bilinear interpolation.

    Parameters:
    - image: 3D NumPy array representing the image (height, width, channels).
    - angle_degrees: Rotation angle in degrees.

    Returns:
    - Rotated image.
    """

    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Get image dimensions
    height, width, channels = image.shape

    # Calculate the rotation matrix
    rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                [np.sin(angle_radians), np.cos(angle_radians)]])

    # Calculate the center of the image
    center = np.array([width / 2, height / 2])

    # Calculate the new dimensions to fit the rotated image
    new_width = int(np.ceil(width * np.abs(np.cos(angle_radians)) + height * np.abs(np.sin(angle_radians))))
    new_height = int(np.ceil(width * np.abs(np.sin(angle_radians)) + height * np.abs(np.cos(angle_radians))))

    # Calculate the translation matrix to keep the rotated image centered
    translation_matrix = np.array([[1, 0, (new_width - width) / 2],
                                   [0, 1, (new_height - height) / 2]])

    # Combine the rotation and translation matrices
    transformation_matrix = np.dot(np.hstack([rotation_matrix, np.zeros((2, 1))]),
                                   np.vstack([np.zeros((1, 2)), translation_matrix]))

    # Perform rotation using bilinear interpolation
    rotated_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            # Calculate the original coordinates in the unrotated image
            original_coords = np.dot(np.linalg.inv(transformation_matrix),
                                     np.array([x - center[0], y - center[1], 1]))

            # Extract the integer and fractional parts of the coordinates
            x_int, y_int = original_coords[:2].astype(int)
            dx, dy = original_coords[:2] - np.floor(original_coords[:2])

            # Check if the original coordinates are within the image boundaries
            if 0 <= x_int < width - 1 and 0 <= y_int < height - 1:
                # Perform bilinear interpolation
                top_left = image[y_int, x_int, :] * (1 - dx) * (1 - dy)
                top_right = image[y_int, x_int + 1, :] * dx * (1 - dy)
                bottom_left = image[y_int + 1, x_int, :] * (1 - dx) * dy
                bottom_right = image[y_int + 1, x_int + 1, :] * dx * dy

                rotated_image[y, x, :] = np.clip(top_left + top_right + bottom_left + bottom_right, 0, 255)

    return rotated_image

@timer
def rotate_bicubic(image, angle_degrees):
    """
    Rotate an image using bicubic interpolation.

    Parameters:
    - image: 3D NumPy array representing the image (height, width, channels).
    - angle_degrees: Rotation angle in degrees.

    Returns:
    - Rotated image.
    """

    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Get image dimensions
    height, width, channels = image.shape

    # Calculate the rotation matrix
    rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                [np.sin(angle_radians), np.cos(angle_radians)]])

    # Calculate the center of the image
    center = np.array([width / 2, height / 2])

    # Calculate the new dimensions to fit the rotated image
    new_width = int(np.ceil(width * np.abs(np.cos(angle_radians)) + height * np.abs(np.sin(angle_radians))))
    new_height = int(np.ceil(width * np.abs(np.sin(angle_radians)) + height * np.abs(np.cos(angle_radians))))

    # Calculate the translation matrix to keep the rotated image centered
    translation_matrix = np.array([[1, 0, (new_width - width) / 2],
                                   [0, 1, (new_height - height) / 2]])

    # Combine the rotation and translation matrices
    transformation_matrix = np.dot(np.hstack([rotation_matrix, np.zeros((2, 1))]),
                                   np.vstack([np.zeros((1, 2)), translation_matrix]))

    # Perform rotation using bicubic interpolation
    rotated_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            # Calculate the original coordinates in the unrotated image
            original_coords = np.dot(np.linalg.inv(transformation_matrix),
                                     np.array([x - center[0], y - center[1], 1]))

            # Extract the integer and fractional parts of the coordinates
            x_int, y_int = original_coords[:2].astype(int)
            dx, dy = original_coords[:2] - np.floor(original_coords[:2])

            # Check if the original coordinates are within the image boundaries
            if 1 <= x_int < width - 2 and 1 <= y_int < height - 2:
                # Perform bicubic interpolation
                values = image[y_int - 1:y_int + 3, x_int - 1:x_int + 3, :]

                coefficients_x = np.array([get_cubic_weight(t + 1 - dx) for t in range(-1, 3)])
                coefficients_y = np.array([get_cubic_weight(t + 1 - dy) for t in range(-1, 3)])

                interpolated_value = 0

                for i in range(3):  # Loop over channels
                    interpolated_value += np.sum(coefficients_x * values[:, :, i] * coefficients_y.T)

                rotated_image[y, x, :] = np.clip(interpolated_value, 0, 255)

    return rotated_image

def get_cubic_weight(t):
    """
    Calculate the cubic interpolation weight for a given parameter.

    Parameters:
    - t: Parameter for interpolation.

    Returns:
    - Cubic interpolation weight.
    """

    a = -0.5

    if abs(t) <= 1:
        return (a + 2) * abs(t)**3 - (a + 3) * abs(t)**2 + 1
    elif 1 < abs(t) <= 2:
        return a * abs(t)**3 - 5 * a * abs(t)**2 + 8 * a * abs(t) - 4 * a
    else:
        return 0


IMG : np.ndarray = cv2.imread("labs/lab2/tests/monke_512.jpg")

#rotate_nearest_neighbor(IMG, 45)
#rotate_bilinear(IMG, 45)
#rotate_bicubic(IMG, 45)