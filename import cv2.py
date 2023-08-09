
#😃😃reading small images

#img = cv.imread('cat.jpg')

#cv.imshow('Cat',img)

#cv.waitKey(0)

#😃😃reading video in cv
#cap = cv.VideoCapture('dance.mp4')

#while True:
 #   isTrue, frame = cap.read()
  #  cv.imshow("Video",frame)

   # if cv.waitKey(20) & 0xFF == ord('d'):
    #    break

#cap.release()
#cv.destroyAllWindows()

# 😃😃 resizing and rescaling of videos and img

# def rescaleFrame(frame, scale = 0.25):
#     width = int(frame.shape[1]*scale)
#     height = int(frame.shape[0]*scale)
#     dimensions = (width, height)
#     return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)
# #cv.waitKey(0)
# img = cv.imread('cat.jpg')
# resized_image = rescaleFrame(img)
# cv.imshow('Cat', resized_image)
# cap = cv.VideoCapture('dance.mp4')


# while True:
#     isTrue, frame = cap.read()

#     frame_resized = rescaleFrame(frame)
#     cv.imshow("Video",frame)
#     cv.imshow('Video Resized', frame_resized)
#     if cv.waitKey(20) & 0xFF == ord('d'):
#         break

# cap.release()
# cv.destroyAllWindows()

# 😃😃 masking of an image
# import numpy as np

# img = cv.imread('cat.jpg')
# cv.imshow('Biral',img)

# blank = np.zeros(img.shape[:2], dtype='uint8')
# cv.imshow("Blank img",blank)

# mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2),100,255,-1)
# cv.imshow('Mask', mask)

# masked = cv.bitwise_and(img,img,mask=mask)
# cv.imshow('Masked image', masked)
# cv.waitKey(0)

# 😃😃 Edge detection
# img = cv.imread('cat.jpg')
# cv.imshow('Biral',img)

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray',gray)

# # loplaction
# lap = cv.Laplacian(gray,cv.CV_64F)
# lap = np.uint8(np.absolute(lap))
# cv.imshow('Laplaction', lap)
# cv.waitKey(0)

import cv2 as cv
import numpy as np

img = cv.imread('smile.jpg')
cv.imshow('Dattkelano',img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Person', gray)

haar_c = cv.CascadeClassifier('harr_face.xml')

faces_rect = haar_c.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)

print(f'Number of faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)
cv.imshow('Detected Faces', img)

cv.waitKey(0)