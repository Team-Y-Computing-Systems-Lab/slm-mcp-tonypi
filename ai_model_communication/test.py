import cv2 
import numpy as np 


# image_path = "/home/rlwagun/Files/llms/neuro-symbolic-ai/ai_model_communication/test/cute_cats1036.jpg"# "/home/rlwagun/Files/llms/neuro-symbolic-ai/ai_model_communication/test/cute_cats268.png"
# image = cv2.imread(image_path)
crop_val = 200
video_path = '/home/rlwagun/Files/llms/neuro-symbolic-ai/ai_model_communication/patient03_clipped.avi'
cap = cv2.VideoCapture(video_path)


# Check if video opened successfully
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

count = 0 # 9999

while True:
    
    ret, image = cap.read()
    # If frame read was not successful, break the loop
    if not ret:
        break
    

    image = image[:,crop_val:-crop_val,:]
    print("x", image.shape )
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("image shape: ", image.shape)
    # image = Image.fromarray(image) # .convert("RGB")

    # Step 1: Read the image
    # image = cv2.imread('image.jpg')
    # Convert the image from BGR to RGB
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 2: Convert the image to a suitable color space (HSV is often good for segmentation)
    image_hsv = image_rgb # cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Step 3: Reshape the image to 2D array of pixels
    pixels = image_hsv.reshape((-1, 3))

    # Step 4: Apply K-means clustering to the pixels
    k = 2  # We want two regions: dark and damp
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(np.float32(pixels), k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # Step 5: Convert labels back to image
    segmented_image = labels.reshape(image_hsv.shape[:2])

    # Compute brightness as mean of RGB (you could also convert to HSV[V])
    brightness = np.mean(centers, axis=1)
    dark_cluster = np.argmin(brightness)  # index of the darker cluster

    # Step 6: Create the mask for dark pixels using the correct cluster
    mask_dark = (segmented_image == dark_cluster).astype(np.uint8)
    # # Step 6: Create the mask for dark pixels (label 0 is dark, 1 is damp)
    # mask_dark = (segmented_image == 0).astype(np.uint8)  # binary mask

    # Step 7: Convert mask to 3 channels so it can be blended
    mask_colored = cv2.cvtColor(mask_dark * 255, cv2.COLOR_GRAY2BGR)

    # Pick an overlay color (e.g., red for dark areas)
    overlay_color = np.zeros_like(image_rgb)
    overlay_color[:, :] = (0, 0, 255)  # Red in RGB

    # Apply the color only on dark areas
    overlay_masked = cv2.bitwise_and(overlay_color, mask_colored)

    # Step 8: Blend the overlay with the original image
    overlayed = cv2.addWeighted(image_rgb, 0.7, overlay_masked, 0.3, 0)

    # Convert to BGR before saving
    overlayed_bgr = cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f"test/cute_cats{count}.jpg", overlayed_bgr)
    count += 1 
    # cv2.imshow('test', overlayed_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
