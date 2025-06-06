import cv2
import numpy as np
import os
import time

def detect_qr_code(frame):
    # Initialize OpenCV QR code detector
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(frame)
    
    if points is not None:
        # Draw bounding box around QR code and show decoded data
        points = points[0].astype(int)
        for j in range(len(points)):
            cv2.line(frame, tuple(points[j]), tuple(points[(j + 1) % len(points)]), (0, 255, 0), 2)
        cv2.putText(frame, f"QR: {data}", (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return frame, data

def detect_barcode(frame):
    # Convert to grayscale and apply gradient edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_y = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(grad_x, grad_y)
    gradient = cv2.convertScaleAbs(gradient)

    # Blur and threshold the image
    blurred = cv2.blur(gradient, (9, 9))
    _, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    # Morphological operations to close gaps between barcode edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    barcode_detected = False  # Flag to track if a barcode has been detected

    for contour in contours:
        # Filter out small contours to avoid false positives
        if cv2.contourArea(contour) > 1000:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)  # Use np.int32 for compatibility
            cv2.drawContours(frame, [box], -1, (0, 255, 0), 3)
            cv2.putText(frame, "Barcode detected", (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            barcode_detected = True  # Set flag to True when barcode is detected
            break  # Exit after first detection

    return frame, barcode_detected

def main():
    folder_path = "images"  # Path to the folder with images
    qr_code_scanned = False  # Flag to track if a QR code has been scanned
    barcode_scanned = False  # Flag to track if a barcode has been scanned

    # List all image files in the folder
    image_files = [f for f in os.listdir(C:/Users/vaish/Desktop/images) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        frame = cv2.imread(os.path.join(r"C:/Users/vaish/Desktop/images", images))
        if frame is None:
            continue
        # Detect QR codes
        frame, qr_data = detect_qr_code(frame)

        # Detect barcodes
        frame, barcode_detected = detect_barcode(frame)

        # If QR data is found and has not been scanned yet, print it with timestamp
        if qr_data and not qr_code_scanned:
            print(f"QR Code Detected: {qr_data} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            qr_code_scanned = True  # Set flag to True after detecting the QR code

        # If a barcode is detected and has not been scanned yet, print it with timestamp
        if barcode_detected and not barcode_scanned:
            print(f"Barcode Detected at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            barcode_scanned = True  # Set flag to True after detecting the barcode

        # Display the output frame
        cv2.imshow("QR and Barcode Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        # Reset flags after displaying each image
        qr_code_scanned = False
        barcode_scanned = False
    cv2.destroyAllWindows()
if name == "main":
main()