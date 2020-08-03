print('hello world')
# %% 1-1) Import nesscesary libraries
import cv2
import numpy as np


# %% 1-2) A function to access the layers of net
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


# %% 1-3) A function to draw result of prediction on initial image (like Rectangle, class name , ... )
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# %% 1-4) Load weights of YOLOv3 and corresponding configuration
scale = 0.00392
classes = None
with open('yolov3.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# %% 1-5) Load corresponfing video and perform object detection in each frame to access number of rectangles of each class

Source_Name = 'Video1'
cap = cv2.VideoCapture(Source_Name + '.avi')

index_array = np.zeros((80))
while (cap.isOpened()):

    ret, image = cap.read()

    if ret == True:

        Width = image.shape[1]
        Height = image.shape[0]

        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True,
                                     crop=False)  # Blob detection with initial Scale

        net.setInput(blob)  # Pass the initial blob candidates to YOLOv3

        outs = net.forward(get_output_layers(net))  # Get output and prediction using YOLOv3

        # Reach rectangle confidences and inidices using YOLOv3:

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # For each detected rectangle draws predicted class names:
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

            index_array[class_ids[i]] = int(index_array[class_ids[i]] + 1)

        cv2.imshow("object detection", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()

# %% 1-6) Save all Target clips independently corresponding to detected rectangles

fourcc = cv2.VideoWriter_fourcc(*'MJPG')

index_array_nonZero = np.where(index_array != 0)[0]
index_array_nonZero_FrameNumber = index_array[np.where(index_array != 0)[0]]

for NonMax_Counter in np.arange(0, len(index_array_nonZero)):

    NonMax = index_array_nonZero[NonMax_Counter]

    cap = cv2.VideoCapture(Source_Name + '.avi')

    out_WriteFinal = cv2.VideoWriter(Source_Name + '_yolov3_Targets_' + str(NonMax) + '.avi', fourcc, 30.0,
                                     (int(cap.get(3)), int(cap.get(4))))

    while (cap.isOpened()):

        ret, image = cap.read()

        if ret == True:

            Width = image.shape[1]
            Height = image.shape[0]

            blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)

            outs = net.forward(get_output_layers(net))

            class_ids = []
            confidences = []
            boxes = []
            conf_threshold = 0.5
            nms_threshold = 0.4

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)
                        w = int(detection[2] * Width)
                        h = int(detection[3] * Height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            if np.sum(class_ids == NonMax) != 0:  # Save specific class corresponding to specific rectangle

                image_Target = np.uint8(np.zeros(image.shape))

                for i in indices:
                    i = i[0]
                    box = boxes[i]
                    x = box[0]
                    y = box[1]
                    w = box[2]
                    h = box[3]

                    if class_ids[i] == NonMax:  # Checks if class number equals to specific class number in FOR LOOP

                        image_Target[round(y): round(y + h), round(x):round(x + w)] = image[round(y): round(y + h),
                                                                                      round(x):round(x + w)]

                        draw_prediction(image_Target, class_ids[i], confidences[i], round(x), round(y), round(x + w),
                                        round(y + h))

                cv2.imshow("object detection targetClass: " + str(NonMax), image_Target)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                out_WriteFinal.write(image_Target)
        else:
            break

    cap.release()
    out_WriteFinal.release()
    cv2.destroyAllWindows()

# %% 1-7) Merge Videos
np.save(Source_Name + '_index_array_nonZero_FrameNumber.npy', index_array_nonZero_FrameNumber)
index_array_nonZero_FrameNumber = np.load(Source_Name + '_index_array_nonZero_FrameNumber.npy')

index_array_nonZero_FrameNumber_sorted = np.sort(
    index_array_nonZero_FrameNumber)  # Sort number of detected class members
index_array_nonZero_sorted = index_array_nonZero[np.argsort(index_array_nonZero_FrameNumber)]

for i in np.arange(0, len(index_array_nonZero_sorted) - 1):  # For LOOP on class members to merge

    NonMax = index_array_nonZero_sorted[i]

    if i == 0:

        cap = cv2.VideoCapture('a.avi')

    else:

        cap = cv2.VideoCapture(Source_Name + 'Merge' + str(i - 1) + '.avi')

    out_WriteFinal = cv2.VideoWriter(Source_Name + 'Merge' + str(i) + '.avi', fourcc, 30.0,
                                     (int(cap.get(3)), int(cap.get(4))))

    cap_NonMax = cv2.VideoCapture(Source_Name + '_yolov3_Targets_' + str(NonMax) + '.avi')

    while (cap.isOpened()):

        ret, image = cap.read()

        if ret == True:

            if (cap_NonMax.isOpened()):

                ret_NonMax, image_NonMax = cap_NonMax.read()

                if ret_NonMax == True:
                    image = cv2.add(image, image_NonMax)

            else:

                cap_NonMax.release()

            cv2.imshow('Merged', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            out_WriteFinal.write(image)



        else:
            break

    cap.release()
    out_WriteFinal.release()
    cv2.destroyAllWindows()

# %% 1-8) Add Background to final VIDEO
Source_Name = 'Video2'
cap_NonMax = cv2.VideoCapture(Source_Name + '.avi')
cap = cv2.VideoCapture(Source_Name + 'Merge' + str(i) + '.avi')
fourcc = cv2.VideoWriter_fourcc(*'WMV2')

out_WriteFinal = cv2.VideoWriter(Source_Name + '_MergeFinal_DEEPLEARNING.avi', fourcc, 30.0,
                                 (int(cap.get(3)), int(cap.get(4))))
Background = cv2.imread('Background_' + Source_Name + '.jpg')

Frame_Counter = 0
Thr = 40
while (cap.isOpened()):

    ret, image = cap.read()

    if ret == True:

        Frame_Counter = Frame_Counter + 1

        print("Frame_Counter: " + str(Frame_Counter))

        if (cap_NonMax.isOpened()):

            ret_NonMax, image_NonMax = cap_NonMax.read()

            if (ret_NonMax == True):
                image[np.sum(image, axis=2) <= Thr, :] = Background[np.sum(image, axis=2) <= Thr, :]

        cv2.imshow('Merged', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        out_WriteFinal.write(image)



    else:
        break

cap.release()
out_WriteFinal.release()
cv2.destroyAllWindows()











