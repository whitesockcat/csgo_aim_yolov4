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
from ctypes import windll
import sys
'''
if not windll.shell32.IsUserAnAdmin():
        # 不是管理员就提权
        windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, __file__, None, 1)

'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import subprocess
import time
from datetime import datetime
import cv2
import mss
import numpy as np
import configparser
from yolov3.utils import *
from yolov3.configs import *
from yolov3.yolov4 import read_class_names
from tools.Detection_to_XML import CreateXMLfile
import random
from pymouse import *     # 模拟鼠标所使用的包
from pykeyboard import *   # 模拟键盘所使用的包
import dm_socket
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

mouse = PyMouse()
keyboard = PyKeyboard()   # 键盘的实例k
damo_socket_api_key = ""
def move_mouse(x1,y1,x2,y2,use_relate_xy,cf):
    x2 = x2 * cf["move_relat_x_adj"]
    y2 = y2 * cf["move_relat_y_adj"]
    if not use_relate_xy:
        mouse.move(x2,y2)
        #dm_socket.send_damo("0","down",0,"down","",0,0,x1,y1,damo_socket_api_key)
    else:
        dm_socket.send_damo("0","down",cf["shot_with_key"],"down","left",cf["move_time"] + random.random()/100,x2,y2,-1,-1,damo_socket_api_key)

    '''
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
    '''

def mouse_action(hwnd,action,button):
    dm_socket.send_damo(hwnd,"",0,action,button,0,0,0,-1,-1,damo_socket_api_key)

def key_action(hwnd,action,key_code):
    dm_socket.send_damo(hwnd,action,key_code,"","",0,0,0,-1,-1,damo_socket_api_key)

def get_config():
    global damo_socket_api_key
    path = "settings.ini"

    config = configparser.ConfigParser()
    config.read(path)
    cf = {}
    cf["user_role"] = config.get("Settings", "user_role")
    cf["window_left"] = config.getint("Settings", "window_left")
    cf["window_top"] = config.getint("Settings", "window_top")
    cf["window_width"] = config.getint("Settings", "window_width")
    cf["window_height"] = config.getint("Settings", "window_height")
    cf["move_time"] = config.getfloat("Settings", "move_time")
    cf["move_relat_x_adj"] = config.getfloat("Settings", "move_relat_x_adj")
    cf["move_relat_y_adj"] = config.getfloat("Settings", "move_relat_y_adj")
    cf["shot_area_adj"] = config.getfloat("Settings", "shot_area_adj")
    cf["shot_with_key"] = config.getint("Settings", "shot_with_key")
    cf["shot_hold_time"] = config.getfloat("Settings", "shot_hold_time")
    cf["shot_vertical_fix"] = config.getint("Settings", "shot_vertical_fix")
    cf['score_threshold'] = config.getfloat("Settings", "score_threshold")
    cf['iou_threshold'] = config.getfloat("Settings", "iou_threshold")
    damo_socket_api_key = config.get("Settings", "api_active_key")
    return cf   

def take_action(shot_type,detect_list,cf):
    global mouseDown_time
    new = min(detect_list[::2], key=abs)
    index = detect_list.index(new)

    x1 = w/2 + x + detect_list[index]
    y1 = h/2 + y + detect_list[index+1]
    x2 = detect_list[index]
    y2 = detect_list[index+1]
    
    movehead = (x1,y1)
    movehead = (detect_list[index],detect_list[index+1])

    #move_mouse(x1,y1,detect_list[index],detect_list[index+1],use_relate_xy,cf)
    #pyautogui.click()
    mouse_action_1 = ""
    diff_min = 60
    if shot_type == "head":
        diff_min = 40

    if abs(detect_list[index])<diff_min*cf["shot_area_adj"] and abs(detect_list[index+1])<diff_min*cf["shot_area_adj"]:
        #pyautogui.mouseDown(button='left')
        #mouse_action(hwnd,"down","left")
        mouse_action_1 = "left"
        print("head mouseDown")
        #dm_socket.send_damo(hwnd,"down",17,"down","left",0,0,0,-1,-1,damo_socket_api_key)
        #pyautogui.keyDown('ctrl')
        #key_action(hwnd,"down",17)
        #keyboard.press_key(keyboard.control_key)
        if  mouseDown_time == 0:
            mouseDown_time = time.time()

    x2 = int( x2 * cf["move_relat_x_adj"])
    y2 = int(y2 * cf["move_relat_y_adj"])
    dm_socket.send_damo("0","down",cf["shot_with_key"],"down",mouse_action_1,cf["move_time"] + random.random()/100,
                        x2,y2,-1,-1,damo_socket_api_key)

    #else:
    #    pyautogui.mouseUp(button='left')

    return movehead

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
use_relate_xy = True
hwnd = windll.user32.FindWindowW(None, "Counter-Strike: Global Offensive")
cf = get_config()
loop_counter = 0
while True:
    try:
        t1 = time.time()
        loop_counter += 1
        if loop_counter %40 ==0:
            cf = get_config()

        x=cf["window_left"]
        y=cf["window_top"]
        w=int(cf["window_width"]/2)
        h=int(cf["window_height"]/2)

        img = np.array(sct.grab({"top": y + int(h-h /2), "left": x + int(w -w/2), "width": w, "height": h, "mon": -1}))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image, detection_list, bboxes = detect_enemy(yolo, np.copy(img), input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES, 
        score_threshold=cf['score_threshold'], iou_threshold=cf['iou_threshold'], rectangle_colors=(255,0,0))
        cv2.circle(image,(int(w/2),int(h/2)), 3, (255,255,255), -1) # center of weapon sight
        #print("detection_list:",len(detection_list))
        th_list, t_list = [], []
        ch_list, c_list = [], []
        for detection in detection_list:
            diff_x = (int(w/2) - int(detection[1]))*-1
            #防止枪上抬，预先修正下 ，越大，往下修正越多
            diff_y = (int(h/2) - int(detection[2]))*-1 + cf["shot_vertical_fix"]
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
            if time.time() - mouseDown_time > cf["shot_hold_time"]:
                #pyautogui.mouseUp(button='left')
                mouse_action(hwnd,"up","left")
                mouseDown_time = 0
                key_action(hwnd,"up",cf["shot_with_key"])
                #pyautogui.keyUp('ctrl')
                #keyboard.release_key(keyboard.control_key)
            #else:
            #    pyautogui.click()        
        
        
        if cf["user_role"] == "ct":
            #玩家是 警
            #优先打头
            if len(th_list)>0:
                movehead = take_action("head",th_list,cf)
            elif len(t_list)>0:
                movehead = take_action("body",t_list,cf)
        else:
            #玩家是 匪
            #优先打头
            if len(ch_list)>0:
                movehead = take_action("head",ch_list,cf)
            elif len(c_list)>0:
                movehead = take_action("body",c_list,cf)

        t2 = time.time()
        times.append(t2-t1)
        times = times[-50:]
        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        print(cf["user_role"],"h",movehead,"\tFPS %1.2f"% fps,"\tdetect:",len(detection_list),
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