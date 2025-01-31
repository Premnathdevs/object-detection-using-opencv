import cv2   #opencv
import time  #delay
import imutils  #resize

vi = cv2.VideoCapture(0)  #cam id
time.sleep(1)

firstFrame=None
area = 500

while True:
    _,img = vi.read()   #read from the camera
    text = "Normal"
    img = imutils.resize(img, width=1000)  #resize
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   #color 2 gray scale img
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)  #Smoothened
    if firstFrame is None:
            firstFrame = gaussianImg  #capturing the first frame
            continue
    imgDiff = cv2.absdiff(firstFrame, gaussianImg)  #absolute difference
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]
    threshImg = cv2.dilate(threshImg, None, iterations=2) #left overs- erotion or dilation
    res = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, #make Complete contours
            cv2.CHAIN_APPROX_SIMPLE)
    res = imutils.grab_contours(res)
    for c in res:
            if cv2.contourArea(c) < area:   #make full area
                    continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (350, 255, 350), 2)
            text = "Moving Object detect"
    print(text)
    cv2.putText(img, text, (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("cameraFeed",img)
    de= cv2.waitKey(1) & 0xFF
    if de == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
