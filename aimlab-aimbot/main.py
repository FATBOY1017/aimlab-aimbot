import pydirectinput
from ultralytics import YOLO
import cv2
import cvzone
import math
import pygetwindow as gw
from PIL import ImageGrab
import numpy as np
import keyboard


def get_window_image(window_title):
    window = gw.getWindowsWithTitle(window_title)[0]
    bbox = (window.left, window.top, window.right, window.bottom)
    img = ImageGrab.grab(bbox)
    img_np = np.array(img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_np, window.left, window.top, window.width, window.height


window_title = 'aimlab_tb'

model = YOLO("YOLO-Weights/yolov8l.pt")

className = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
             "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
             "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
             "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
             "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
             "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
             "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
             "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
             "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
             "teddy bear", "hair drier", "toothbrush"
             ]

running = False


def toggle_running():
    global running
    if not running:
        running = True
        print("开始运行.")
    else:
        running = False
        print("停止.")


keyboard.on_press_key('shift', lambda _: toggle_running())

while True:
    if running:
        img, window_left, window_top, window_width, window_height = get_window_image(window_title)
        if img is not None:
            results = model(img, stream=True)
            found_ball = False

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    w, h = x2 - x1, y2 - y1
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    currentClass = className[cls]

                    if currentClass == "sports ball" and conf >= 0.2:
                        found_ball = True
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        screen_x = window_left + center_x
                        screen_y = window_top + center_y

                        current_mouse_x, current_mouse_y = pydirectinput.position()

                        move_x = screen_x - current_mouse_x
                        move_y = screen_y - current_mouse_y

                        pydirectinput.moveRel(move_x, move_y, relative=True)

                        cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                                           scale=1, thickness=1, offset=5)
                        cvzone.cornerRect(img, (x1, y1, w, h), l=8)

            if not found_ball:
                screen_center_x = window_left + window_width // 2
                screen_center_y = window_top + window_height // 2
                current_mouse_x, current_mouse_y = pydirectinput.position()
                move_x = screen_center_x - current_mouse_x
                move_y = screen_center_y - current_mouse_y
                pydirectinput.moveRel(move_x, move_y, relative=True)

            resized_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("Image", resized_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
keyboard.unhook_all()
