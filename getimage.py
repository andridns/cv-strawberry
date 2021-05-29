import cv2
cam = cv2.VideoCapture(0)
_, image = cam.read()
cv2.imshow("Webcam", image)
