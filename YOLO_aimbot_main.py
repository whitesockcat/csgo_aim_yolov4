#================================================================
#
#   File name   : YOLO_aimbot_main.py
#   Author      : PyLessons
#   Created date: 2020-10-06
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : CSGO main yolo aimbot script
#
#================================================================
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import subprocess
import time
from datetime import datetime
import cv2
import mss
import numpy as np
import tensorflow as tf
from yolov3.utils import *
from yolov3.configs import *
from yolov3.yolov4 import read_class_names
from tools.Detection_to_XML import CreateXMLfile
import random

# pyautogui settings
import pyautogui # https://totalcsgo.com/commands/mouse
pyautogui.MINIMUM_DURATION = 0.15
pyautogui.MINIMUM_SLEEP = 0.5
pyautogui.PAUSE = 0

NUM_CLASS = read_class_names(TRAIN_CLASSES)

def draw_enemy(image, bboxes, CLASSES=YOLO_COCO_CLASSES, show_label=True, show_confidence = True, Text_colors=(255,255,0), rectangle_colors='', tracking=False):   
    
    num_classes = len(NUM_CLASS)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    detection_list = []

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        # put object rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

        x, y = int(x1+(x2-x1)/2), int(y1+(y2-y1)/2)
        cv2.circle(image,(x,y), 3, (50,150,255), -1)
        detection_list.append([NUM_CLASS[class_ind], x, y])

        if show_label:
            # get text label
            score_str = " {:.2f}".format(score) if show_confidence else ""

            if tracking: score_str = " "+str(score)

            label = "{}".format(NUM_CLASS[class_ind]) + score_str

            # get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale, thickness=bbox_thick)
            # put filled text rectangle
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

    return image, detection_list

def detect_enemy(Yolo, original_image, input_size=416, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.2, iou_threshold=0.35, rectangle_colors=''):
    image_data = image_preprocess(original_image, [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    if YOLO_FRAMEWORK == "tf":
        pred_bbox = Yolo.predict(image_data)

    elif YOLO_FRAMEWORK == "trt":
        batched_input = tf.constant(image_data)
        result = Yolo(batched_input)
        pred_bbox = []
        for key, value in result.items():
            value = value.numpy()
            pred_bbox.append(value)
        
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    
    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='nms')

    image, detection_list = draw_enemy(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
        
    return image, detection_list, bboxes

def getwindowgeometry():
    while True:
        output = subprocess.getstatusoutput(f'xdotool search --name Counter-Strike: getwindowgeometry')
        if output[0] == 0:
            t1 = time.time()
            LIST = output[1].split("\n")
            Window = LIST[0][7:]
            Position = LIST[1][12:-12]
            x, y = Position.split(",")
            x, y = int(x), int(y)
            screen = LIST[1][-2]
            Geometry =  LIST[2][12:]
            w, h = Geometry.split("x")
            w, h = int(w), int(h)
            
            outputFocus = subprocess.getstatusoutput(f'xdotool getwindowfocus')[1]
            if outputFocus == Window:
                return x, y, w, h
            else:
                subprocess.getstatusoutput(f'xdotool windowfocus {Window}')
                print("Waiting for window")
                time.sleep(5)
                continue

def move_mouse(x1,y1,x2,y2,use_relate_xy):
    
    duration = 0# 0.05 + int(random.random()*100)/800
    try:
        if use_relate_xy:
            pyautogui.move(x2,y2,duration,tween=pyautogui.easeOutQuad)
        else:
            pyautogui.moveTo(x1,y1,duration,tween=pyautogui.easeOutQuad)
    except Exception as e:
        if use_relate_xy:
            pyautogui.move(x2,y2,duration)
        else:
            pyautogui.moveTo(x1,y1,duration)

offset = 30
times = []
sct = mss.mss()
yolo = Load_Yolo_model()
#x, y, w, h = getwindowgeometry()
#monitor = {"top": 80, "left": 0, "width": consts.width, "height": consts.height}
x=0
y=34
w=int(1600/2)
h=int(900/2)
pyautogui.FAILSAFE=False
mouseDown_time = 0
show_debug_window = True
while True:
    try:
        t1 = time.time()

        img = np.array(sct.grab({"top": y + int(h-h /2), "left": x + int(w -w/2), "width": w, "height": h, "mon": -1}))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image, detection_list, bboxes = detect_enemy(yolo, np.copy(img), input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
        cv2.circle(image,(int(w/2),int(h/2)), 3, (255,255,255), -1) # center of weapon sight
        #print("detection_list:",len(detection_list))
        th_list, t_list = [], []
        ch_list, c_list = [], []
        for detection in detection_list:
            diff_x = (int(w/2) - int(detection[1]))*-1
            diff_y = (int(h/2) - int(detection[2]))*-1
            if detection[0] == "th":
                th_list += [diff_x, diff_y]
            elif detection[0] == "t":
                t_list += [diff_x, diff_y]
            elif detection[0] == "ch":
                ch_list += [diff_x, diff_y]
            elif detection[0] == "c":
                c_list += [diff_x, diff_y]

        move=()
        movehead =()
        if mouseDown_time > 0 :
            if time.time() - mouseDown_time >1:
                pyautogui.mouseUp(button='left')
                mouseDown_time = 0
                pyautogui.keyUp('ctrl')
            #else:
            #    pyautogui.click()        

        use_relate_xy = False
        #优先打头
        if len(th_list)>0:
            new = min(th_list[::2], key=abs)
            index = th_list.index(new)

            x1 = w/2 + x + th_list[index]
            y1 = h/2 + y + th_list[index+1]
            
            movehead = (x1,y1)
            movehead = (th_list[index],th_list[index+1])

            move_mouse(x1,y1,th_list[index],th_list[index+1],use_relate_xy)
            #pyautogui.click()
            if abs(th_list[index])<50 and abs(th_list[index+1])<50:
                pyautogui.mouseDown(button='left')
                pyautogui.keyDown('ctrl')
                if mouseDown_time == 0:
                    print("head mouseDown")
                    mouseDown_time = time.time()
                    
            #else:
            #    pyautogui.mouseUp(button='left')

        if len(t_list)>0:
            new = min(t_list[::2], key=abs)
            index = t_list.index(new)
            x2 = w/2 + x + t_list[index]
            y2 = h/2 + y + t_list[index+1]
            #xx = t_list[index]
            move = (x2,y2)
            move = (t_list[index],t_list[index+1])
            move_mouse(x2,y2,t_list[index],t_list[index+1],use_relate_xy)
            #pyautogui.click()
            if abs(t_list[index])<50 and abs(t_list[index+1])<50:
                pyautogui.mouseDown(button='left')
                pyautogui.keyDown('ctrl')
                if mouseDown_time == 0:
                    print("body mouseDown")
                    mouseDown_time = time.time()
                    
            #else:
            #    pyautogui.mouseUp(button='left')

        t2 = time.time()
        times.append(t2-t1)
        times = times[-50:]
        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        print("h",movehead,"\tb",move,"\tFPS %1.2f"% fps,"\tdetect:",len(detection_list),
        "\tth",len(th_list),"\tt",len(t_list),"\tch",len(ch_list),"\tct",len(c_list))

        if show_debug_window:
            image = cv2.putText(image, "Time: {:.1f}FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            img_test = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            windowname= "OpenCV/Numpy normal"

            #cv2.namedWindow(windowname, 0)
            #cv2.resizeWindow(windowname, 480, 320)
            cv2.imshow(windowname, img_test)

        if cv2.waitKey(5) & 0xFF == ord("/"):
            #cv2.destroyAllWindows()
            show_debug_window = not show_debug_window   
            #break

    except Exception as e:
        print("err",e)
        #raise e
        #pass