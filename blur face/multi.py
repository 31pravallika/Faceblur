import cv2
import numpy as np

def blur(img, k):
    h, w = img.shape[:2]
    kh, kw = h // k, w // k
    if kh % 2 == 0:
        kh -= 1
    if kw % 2 == 0:
        kw -= 1
    img = cv2.GaussianBlur(img, ksize=(kh, kw), sigmaX=0)
    return img

def pixelate_face(image, blocks=10):
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]
            roi = image[startY:endY, startX:endX]
            pixelated_roi = cv2.resize(roi, (endX - startX, endY - startY), interpolation=cv2.INTER_NEAREST)
            image[startY:endY, startX:endX] = pixelated_roi
    return image

factor = 3  
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

selected_face = None
selected_face_index = -1

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.5, 5)

    # Draw rectangles around all detected faces
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Check if a face is selected and draw a blue rectangle around it
    if selected_face_index >= 0 and selected_face_index < len(faces):
        (x, y, w, h) = faces[selected_face_index]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        selected_face = frame[y:y + h, x:x + w]
    else:
        selected_face = None

    # Apply blurring to the selected face
    if selected_face is not None:
        blurred_face = blur(selected_face, factor)
        (h, w) = (selected_face.shape[0], selected_face.shape[1])
        resized_blurred_face = cv2.resize(blurred_face, (w, h), interpolation=cv2.INTER_NEAREST)
        frame[y:y + h, x:x + w] = pixelate_face(resized_blurred_face)

    cv2.imshow('Live', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key >= 48 and key <= 57:
        selected_face_index = key - 49  # Pressing 1 selects the first face, 2 selects the second face, and so on

cap.release()
cv2.destroyAllWindows()
