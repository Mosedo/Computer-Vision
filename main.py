import cv2
import numpy as np
import imutils

whT=320 #width and height of frame image

cap=cv2.VideoCapture("./design_video.mp4")

classesFile='coco.names';

classNames=[]

conf_thresh=0.4
nms_Threshold=0.4

with open(classesFile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')

modelConfiguration='yolov4-tiny.cfg'
modelWeights='yolov4-tiny.weights'
# modelConfiguration='yolov4.cfg'
# modelWeights='yolov4.weights'
net=cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)



def findObjects(outputs,frame):
    hT,wT,cT=frame.shape
    box=[]
    classIds=[]
    confs=[]
    for output in outputs:
        for detection in output:
            scores=detection[5:]
            classId=np.argmax(scores)
            confidence=scores[classId]
            if confidence > conf_thresh:
                w,h=int(detection[2]*wT),int(detection[3]*hT)
                x,y=int((detection[0]*wT) - w/2),int((detection[1]*hT)-h/2)
                box.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices=cv2.dnn.NMSBoxes(box,confs,conf_thresh,nms_Threshold)
    for i in indices:
        i=i[0]
        bx=box[i]
        x,y,w,h=bx[0],bx[1],bx[2],bx[3]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
        cv2.putText(frame,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)


while True:
    ret,frame=cap.read()

    frame = imutils.resize(frame, width=1000)

    blob=cv2.dnn.blobFromImage(frame,1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames=net.getLayerNames()
    
    #extract only the output layers

    #net.getUnconnectedOutLayers()

    outputNames=[layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

    #foward propagation expect 3 output matrices
    output=net.forward(outputNames)
    #get boxes from output
    findObjects(output,frame)

    cv2.imshow("Detection",frame)
    if cv2.waitKey(1)==ord("q"):
        break


cap.release()
cv2.destroyAllWindows()