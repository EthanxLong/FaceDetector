import cv2
import keyboard
from random import randrange 

# Pretrained data on faces loaded (haar cascade algo)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture a video from a webcam
webcam = cv2.VideoCapture(0)

r = randrange(256)
g = randrange(256)
b = randrange(256)
# iterate over webcam frames
while True:
    
    if keyboard.is_pressed('q'):
        print('Stopped..')
        break
    
    # Read current frame 
    successful_framge_read, frame = webcam.read()

    # Convert to grayscale for efficiency
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    
    # Draw rectangles around each face            
    for (x, y, w, h) in face_coordinates:   
        cv2.rectangle(frame, (x, y), (x + w, y + h), (r, g, b), 10)
    
    # Display window with the frame
    cv2.imshow("Ethan Long's Face Detector", frame)
    cv2.waitKey(1)

# Release webcam window     
webcam.release()

'''
# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw rectangles around each face            vv Draw green here
for (x, y, w, h) in face_coordinates:   
    cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 10)

#print(face_coordinates)

# Display the image
cv2.imshow("Ethan Long's Face Detector", img)
cv2.waitKey()

print("Code Completed")
'''
