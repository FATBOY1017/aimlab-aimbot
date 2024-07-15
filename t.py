import pydirectinput
from ultralytics import YOLO
import cv2
import cvzone
import math
import pygetwindow as gw
from PIL import ImageGrab
import numpy as np
import keyboard
import win32api, win32con
import time
import threading

WINDOW_TITLE = 'aimlab_tb'
CONFIDENCE_THRESHOLD = 0.2

model = YOLO("YOLO-Weights/yolov8s.pt")

running = False

def toggle_running():
    global running
    running = not running
    if running:
        print("开始运行.")
    else:
        print("停止.")

keyboard.on_press_key('shift', lambda _: toggle_running())

def get_window_image(window_title):
    window = gw.getWindowsWithTitle(window_title)[0]
    bbox = (window.left, window.top, window.right, window.bottom)
    img = ImageGrab.grab(bbox)
    img_np = np.array(img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_np, window.left, window.top, window.width, window.height

def process_image():
    global running

    while True:
        if running:
            img, window_left, window_top, window_width, window_height = get_window_image(WINDOW_TITLE)
            if img is not None:
                results = model(img, stream=True)
                targets = []

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        w, h = x2 - x1, y2 - y1
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        cls = int(box.cls[0])
                        currentClass = model.names[cls]

                        if currentClass == "sports ball" and conf >= CONFIDENCE_THRESHOLD:
                            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                            screen_x = window_left + center_x
                            screen_y = window_top + center_y
                            targets.append((screen_x, screen_y, conf))


                targets.sort(key=lambda x: x[2])

                if targets:

                    screen_x, screen_y, _ = targets[0]
                    current_mouse_x, current_mouse_y = pydirectinput.position()

                    move_x = screen_x - current_mouse_x
                    move_y = screen_y - current_mouse_y


                    pydirectinput.moveRel(move_x, move_y, relative=True)

                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                    time.sleep(0.01)
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


                else:

                    screen_center_x = window_left + window_width // 2
                    screen_center_y = window_top + window_height // 2
                    current_mouse_x, current_mouse_y = pydirectinput.position()
                    move_x = screen_center_x - current_mouse_x
                    move_y = screen_center_y - current_mouse_y
                    pydirectinput.moveRel(move_x, move_y, relative=True)





thread = threading.Thread(target=process_image)
thread.start()
