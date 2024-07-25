from panorama import PanoramaCreator
import imutils
import cv2
import os

# Get number of images and their filenames from the user
num_images = int(input("Enter the number of images to stitch: "))
print("Enter the filenames in left-to-right order:")

filenames = []
for i in range(num_images):
    filenames.append(input(f"Enter filename for image {i + 1} with path and extension: "))

# Read and resize images
images = []
for filename in filenames:
    if not os.path.isfile(filename):
        print(f"Error: File '{filename}' does not exist.")
        exit(1)
    image = cv2.imread(filename)
    if image is None:
        print(f"Error: Unable to read '{filename}'.")
        exit(1)
    images.append(image)

# Resize all images to the same height
min_height = min(image.shape[0] for image in images)
images = [imutils.resize(image, height=min_height) for image in images]

# Create panorama object and stitch images
stitcher = PanoramaCreator()
if num_images == 2:
    result, keypoints_image = stitcher.stitch([images[0], images[1]], display_matches=True)
else:
    result, keypoints_image = stitcher.stitch([images[-2], images[-1]], display_matches=True)
    for i in range(num_images - 2):
        result, keypoints_image = stitcher.stitch([images[-i - 3], result], display_matches=True)

# Display and save results
cv2.imshow("Keypoint Matches", keypoints_image)
cv2.imshow("Panorama", result)

cv2.imwrite("output/matched_points.jpg", keypoints_image)
cv2.imwrite("output/panorama_image.jpg", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
