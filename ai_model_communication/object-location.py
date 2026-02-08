import cv2

# Global variables
drawing = False
ix, iy = -1, -1
bounding_box = None
paused = False
roi_window_name = "Selected ROI"

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, bounding_box, frame, paused

    if not paused:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        bounding_box = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = frame.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.putText(img_copy, f"({x}, {y})", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.imshow("Camera", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        bounding_box = (x1, y1, x2 - x1, y2 - y1)

        print(f"Selected Bounding Box: Top-left ({x1}, {y1}), Width: {x2 - x1}, Height: {y2 - y1}")

        # Draw final box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Top-left: ({x1},{y1})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Camera", frame)

        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            cv2.imshow(roi_window_name, roi)
        else:
            print("Empty ROI. Selection may be out of bounds.")

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", draw_rectangle)

print("Press SPACE to pause/resume the frame and select ROI.")
print("Press ESC to exit.")

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        paused = not paused
        if paused:
            print("Frame paused. Draw your ROI.")
        else:
            print("Resuming camera feed.")
            cv2.destroyWindow(roi_window_name)

cap.release()
cv2.destroyAllWindows()
