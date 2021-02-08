#Import the openCV
import cv2
#Import the numpy as np
import numpy as np
#Import the time
import time


#We use "CV2.dnn.ReadNet()" function for loading the network into memory
#and load the Yolo algorithm data in net
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

#Consider array for loading the objects names
objectclasses = []

#Using 'with' statement open the data
with open("coco.names", "r") as f:
    objectclasses = [line.strip() for line in f.readlines()]

#Now we will use "getLayerNames()" function to get the name of all layers of the network
layer_names = net.getLayerNames()

#"getUnconnectedOutLayers()" It gives the final layers number in the list from net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#"np.random.uniform" returns the random samples as numpy array.
colors = np.random.uniform(0, 255, size=(len(objectclasses), 3))

# load the image
img = cv2.imread("testimg.jpg")

# resize the image
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

#We use "blobFromImage"function for Detecting objects resizes and crops image from center,
#subtract mean values, scales values by scalefactor
#detect objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

#For set the image in network we use "setInput" function
net.setInput(blob)
outs = net.forward(output_layers)

# Shows the informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        #"argmax" returns the indices of the maximum values along an axis.
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detection
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Set the co-ordinates of rectangle
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            #append it.
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

#"NMSBoxes" function performs non maximum suppression given boxes and corresponding scores.
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#Now print the indexes...
print(indexes)

#"FONT_HERSHEY_PLAIN" is a small size sans-serif font
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(objectclasses[class_ids[i]])
        color = colors[class_ids[i]]
        #Now set the values in rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        #put the text...
        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

#"imshow" function display an image in the window.
cv2.imshow("Image", img)

# waitKey(0) display the window infinitely until any keypress
cv2.waitKey(0)

#destroyAllWindows() destroys all the windows we created.
cv2.destroyAllWindows()
