import cv2

# Load the reference image
ref_image = cv2.imread('ref_image.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize a list to store the registered images
registered_images = []

# Load each knife image and perform registration
for i in range(1, 9):
    # Load the current image
    curr_image = cv2.imread(f'knife_{i}.jpg', cv2.IMREAD_GRAYSCALE)
    
    # Perform registration
    warp_mode = cv2.MOTION_EUCLIDEAN
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-5)
    (cc, warp_matrix) = cv2.findTransformECC(ref_image, curr_image, warp_matrix, warp_mode, criteria)

    # Apply the transformation to the current image
    registered_image = cv2.warpAffine(curr_image, warp_matrix, (ref_image.shape[1], ref_image.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # Add the registered image to the list
    registered_images.append(registered_image)

# Display the registered images
for i, registered_image in enumerate(registered_images):
    cv2.imshow(f'Knife {i+1}', registered_image)

# Wait for a key press and then exit
cv2.waitKey(0)
cv2.destroyAllWindows()
