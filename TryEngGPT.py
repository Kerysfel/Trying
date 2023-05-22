import cv2
import numpy as np

# Load the network
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

def detect_objects(image):
    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    outputs = net.forward(output_layer_names)

    boxes = []
    class_ids = []
    confidences = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
        
            if confidence > 0.1:
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, box_width, box_height) = box.astype("int")
                x = int(center_x - (box_width / 2))
                y = int(center_y - (box_height / 2))

                boxes.append([x, y, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    return boxes, class_ids, confidences


# Load the images
image1 = cv2.imread('images/im1.jpg')
image2 = cv2.imread('images/im2.jpg')

boxes1, class_ids1, confidences1 = detect_objects(image1)
boxes2, class_ids2, confidences2 = detect_objects(image2)

# Find matches between the class IDs of the detected objects in both images
matches = set(class_ids1) & set(class_ids2)

# Draw bounding boxes around the matched objects in both images
for i in range(len(class_ids1)):
    if class_ids1[i] in matches:
        box = boxes1[i]
        cv2.rectangle(image1, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)

for i in range(len(class_ids2)):
    if class_ids2[i] in matches:
        box = boxes2[i]
        cv2.rectangle(image2, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)

# Display the images
cv2.imwrite("images/GPTim1.jpg", image1)
cv2.imwrite("images/GPTim2.jpg", image2)
