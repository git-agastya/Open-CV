import cv2

#load the pre-trained haar ascade classifier for face detection
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

#load mustache image
mustache=cv2.imread(r"mustache.png",cv2.IMREAD_UNCHANGED)

#start the webcam
cap=cv2.VideoCapture(0)

#check if the webcam is opened correclty
if not cap.isOpened():
    print("Error :Could not open webcam.")
    exit()

#keep the webcam running until 'ESC' on keyboard is pressed
while True:
    ret, frame=cap.read()

    if not ret:
        print("Error: Failed to grab the frame.")
        break
    
    #convert the frame to grayscale (required or face detection)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #detect faces in the grayscale image
    faces=face_cascade.detectMultiScale(gray)
    
    #loop over the detected faces and draw rectangles
    for (x,y,w,h) in faces:
        mustache_width=int(w*0.6)
        mustache_height=int(h*0.3)

        resized_mustache=cv2.resize(mustache,(mustache_width,mustache_height))
        
        #calculate mustache width 30%
        x1=x+int(w*0.3)
        #calculate mustache height 60%
        y1=y+int(h*0.6)
        x2=x1+mustache_width
        y2=y1+mustache_height
        
        if y2>frame.shape[0] or x2>frame.shape[1]:
            continue

        m_rgb=resized_mustache[:,:,:3]
        m_alpha=resized_mustache[:,:,3]/255.0
        roi=frame[y1:y2,x1:x2]

        for c in range(3):
            roi[:,:,c]=(1-m_alpha)*roi[:,:,c]+m_alpha*m_rgb[:,:,c]
        frame[y1:y2,x1:x2] =roi
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)




    cv2.imshow("Webcam Feed", frame)


    if cv2.waitKey(1) & 0xFF==27:
        break



cap.release()
cv2.destroyAllWindows() 