import keras.backend as K
from keras.models import load_model
import xml.etree.ElementTree as ET
from keras.preprocessing.image import load_img, img_to_array, array_to_img
#from keras.utils import multi_gpu_model
import numpy as np
from datetime import datetime
import os
from settings import setting
import cv2

import itertools

if setting["architecture"] == "Yolo_V1":
    import Util_V1 as U
elif setting["architecture"] == "Yolo_V2":
    import Util_V2 as U

def draw(image, boxes, scores, classes, all_classes = None):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """

    boxes = K.eval(boxes)
    scores = K.eval(scores)
    classes = K.eval(classes)
    for box, score, cl in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, 'C'+str(cl),
                    (int((x2+x1)/2.0), int((y1+y2)/2.0) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)
    print()
    image = image.transpose()
    image = array_to_img(image, data_format='channels_first')
    image.show()
    return image


def Matching(x1_pred, y1_pred, x2_pred, y2_pred, x_max_true, x_min_true, y_max_true, y_min_true):
    intersect_x1 = K.maximum(x1_pred, x_min_true)
    intersect_y1 = K.maximum(y1_pred, y_min_true)
    intersect_x2 = K.minimum(x2_pred, x_max_true)
    intersect_y2 = K.minimum(y2_pred, y_max_true)

    area_1 = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    area_2 = (x_max_true - x_min_true) * (y_max_true - y_min_true)

    intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)

    IOU = intersect_area / (area_1 + area_2 - intersect_area)
    matching = K.greater_equal(IOU, 0.6)
    return matching

# def Calculating_Accuracy(train_label, results):
#     scores, boxes, classes = U.generating_consequences(results)
#     x1, y1, x2, y2 = U.transform_to_coordinate(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
#     x_max_true = train_label[:, 0]
#     x_min_true = train_label[:, 1]
#     y_max_true = train_label[:, 2]
#     y_min_true = train_label[:, 3]
#     x_max_true = np.reshape(x_max_true, (-1, 1))
#     x_min_true = np.reshape(x_min_true, (-1, 1))
#     y_max_true = np.reshape(y_max_true, (-1, 1))
#     y_min_true = np.reshape(y_min_true, (-1, 1))
#     IOU = Matching(x1_pred=x1, y1_pred=y1, x2_pred=x2, y2_pred=y2, x_max_true=x_max_true, x_min_true=x_min_true, y_max_true=y_max_true, y_min_true=y_min_true)
#
#     IOU = K.cast(IOU, 'float32')
#     mat = K.max(IOU, axis=1)
#     mat = 1 - mat
#
#     localization_loss = (K.square(x1 - x_min_true) + K.square(x2 - x_max_true) + K.square(y1 - y_min_true) + K.square
#         (y2 - y_max_true))
#     localization_loss = K.mean(localization_loss, axis = -1) * mat
#     groundtrue_class = train_label[:, 6]
#     groundtrue_class = np.reshape(groundtrue_class, (10, 1))
#     classification_loss = K.equal(classes, groundtrue_class)
#     classification_loss = K.cast(x=classification_loss, dtype='float32')
#     class_mat = K.max(classification_loss, axis= -1)
#     classification_loss = 1 - classification_loss
#     classification_loss = K.mean(classification_loss, axis = -1) * class_mat
# #    localization_loss = K.mean(localization_loss, axis = -1)
# #    classification_loss = K.mean(classification_loss, axis = -1)
#     print('The localization loss is', K.eval(localization_loss))
#     print('The classification loss is', K.eval(classification_loss))
#     return localization_loss, classification_loss

def LOAD_IMAGE(path = 'C:/Users/ZJZ20\Downloads/DeepLearnData/img_0.jpg'):
    image1 = load_img(path)
    train_example = img_to_array(image1, data_format='channels_first')
    img = array_to_img(train_example, data_format='channels_first')
    img.show()
    train_example = train_example.transpose()
    return train_example

def readXML(f='C:/Users/ZJZ20\Downloads/DeepLearnData/img_0.xml'):
    mapped = {}
    tree = ET.parse(f)
    root = tree.getroot()
    for elem in root:
        if len(elem) == 0:
            mapped[elem.tag] = elem.text
        for subelem in elem:
            if len(subelem) == 0:
                mapped[subelem.tag] = subelem.text
            for sub2elem in subelem:
                if len(sub2elem) == 0:
                    mapped[sub2elem.tag] = sub2elem.text
    xmax = int(mapped['xmax'])
    xmin = int(mapped['xmin'])
    ymax = int(mapped['ymax'])
    ymin = int(mapped['ymin'])
    C_index = int(mapped['name'][1])
    yTrue = np.zeros((7), dtype='float32')
    yTrue = [xmax, xmin, ymax, ymin, 0, 0, C_index]
    return yTrue

def load_DATA(srcDir):
    fileCount = len([name for name in os.listdir(srcDir) if name.endswith(".xml")])
    train_data = np.empty(shape=[fileCount, 640, 480, 3], dtype='float32')
    train_label = np.empty(shape=[fileCount, 7], dtype='float32')

    count = 0
    for filename in os.listdir(srcDir):
        if not filename.endswith(".xml"): continue
        count += 1
        xmlFile = srcDir + "/" + filename
        print(xmlFile)
        imgFile = xmlFile.replace(".xml", ".jpg")
        print(imgFile)
        train_data[count - 1] = LOAD_IMAGE(imgFile)
        train_label[count - 1] = readXML(xmlFile)
    return [train_data, train_label]


def main(iniModel, setting):
    if setting['loss'] == 'Loss_v2':
        lossFunction = U.Loss_v2
    elif setting['loss'] == 'Loss_v3':
        lossFunction = U.Loss_v3
    model = load_model('weight-setting5.h5', custom_objects={setting["loss"]: lossFunction})
    folderCount = 1
    for count in range(0, folderCount, 1):
        srcDir = "D:/Dataset/YoLo/0"
        print("+++ run: " + srcDir + " " + str(datetime.now()) + "+++")
        train_data, train_label = load_DATA(srcDir=srcDir)
        results = model.predict(train_data, batch_size=1)
        scores, boxes, classes = U.generating_consequences(results)
        x1, y1, x2, y2 = U.transform_to_coordinate(boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3])
        x1 = K.reshape(x=x1, shape = (-1,1))
        x2 = K.reshape(x=x2, shape = (-1,1))
        y1 = K.reshape(x=y1, shape = (-1,1))
        y2 = K.reshape(x=y2, shape = (-1,1))
        boxes = K.concatenate([x1, y1, x2, y2])
        image = draw(image=train_data[0], boxes=boxes, scores=scores, classes=classes)
if __name__ == '__main__':
    print("+++ start: " + str(datetime.now()) + "+++")
    iniModel = True
    if os.path.isfile(setting["weight_file"]):
       iniModel = False
    main(iniModel, setting)
    print("+++ finished: " + str(datetime.now()) + "+++")
