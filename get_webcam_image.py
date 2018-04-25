import cv2

def get_feed(cam):
    while(True):
        # Capture frame-by-frame
        ret, frame = cam.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam = cv2.VideoCapture(0)
get_feed(cam)

# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()