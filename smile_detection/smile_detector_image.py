"""
Python model for face smile detection
Implementation: cv2
"""
# https://dontrepeatyourself.org/post/smile-detection-with-python-opencv-and-haar-cascade/
import cv2

# Size of Image

width = 400
height = 300

# Mark Color
mint = (29, 233, 182)

def SmileDetector_image( image ):
    # Load the image
    # image = cv2.imread("../face/face9.jpg")
    # image = cv2.imread("person.jpeg")

    # Resize the image
    image = cv2.resize(image, (width, height))

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the haar cascades face and smile detectors
    face_detector = cv2.CascadeClassifier("../haarcascades/haarcascade_frontalface_default.xml")
    smile_detector = cv2.CascadeClassifier("../haarcascades/haarcascade_smile.xml")

    # Detect faces in the grayscale image
    face_rects = face_detector.detectMultiScale(gray, 1.1, 8)

    # Loop over the face bounding boxes
    for (x, y, w, h) in face_rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), mint, 2)
        # extract the face from the grayscale image
        roi = gray[y:y + h, x:x + w]

        # Apply the smile detector to the face roi
        smile_rects, rejectLevels, levelWeights = smile_detector.detectMultiScale3(roi, 2.5, 20, outputRejectLevels=True)

        # If there was no detection, we consider this a "no smiling" detection
        if len(levelWeights) == 0:
            cv2.putText(image, "Not Smiling", (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, mint, 3)
        else:
            # If `levelWeights` is below 2, we classify this as "Not Smiling"
            if max(levelWeights) < 2:
                cv2.putText(image, "Not Smiling", (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, mint, 3)
            # Otherwise, there is a smiling in the face ROI
            else:
                cv2.putText(image, "Smiling", (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, mint, 3)


# cv2.imshow("image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()