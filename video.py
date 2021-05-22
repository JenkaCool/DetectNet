#
#  This program for human detection in the selected area by neural network
#
#  Copyright (c) 2021, Evgeniya Balyuk <balyuk@petrsu.ru>
#
#  This code is licensed under a MIT-style license.

import cv2
import numpy as np
import time
from tracker import *


# Создаём трекер
tracker = EuclideanDistTracker()

# Подгрузка Yolo
net = cv2.dnn.readNet("weights/yolov3.weights", "cfg/yolov3.cfg")
# Считываем классы
classes = []
with open("classes/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
# Получаем имена всех слоев сети
layer_names = net.getLayerNames()
# Получаем индекс выходных слоев
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# Генерируем цвет для визуализации прямоугольныых границ объекта
colors = np.random.uniform(0, 255, size=(len(classes), 3))
# Задаём стиль шрифта
font = cv2.FONT_HERSHEY_PLAIN

# Захват камеры
cap = cv2.VideoCapture("videos/highway.mp4")
# Начало отсёта времени
starting_time = time.time()
# Вводим счётчик кадров
frame_id = 0

# Удаляем фон для детектора
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# Обрабатываем видеопоток
while True:
    # Получаем кадр
    _, frame = cap.read()
    # Обновляем счётчик
    frame_id += 1
    # Получаем параметры кадра
    height, width, channels = frame.shape

    #print(height, width)

    # Задаём определённую зону кадра,
    # в которой будем вести счёт людей
    reg_x1 = 470
    reg_x2 = 870
    reg_y1 = 550
    reg_y2 = 720

    reg_of_det = frame[reg_y1: reg_y2, reg_x1: reg_x2]

    # Получаем маску кадра
    mask = object_detector.apply(reg_of_det)
    # Убираем шум
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    # Находим контуры
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        # Рассматриваем найденный контур
        area = cv2.contourArea(contour)
        # Если содержит > 100 прикселей,
        # то записываем параметры и добавляем в массив "обнаружения"
        if area > 100:
            #cv2.drawContours(reg_of_det, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            detections.append([x, y, w, h])

    # Отслеживание объектов
    №
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(reg_of_det, str(id + 1), (x, y - 15), font, 1, (255, 0, 0), 2)
        cv2.rectangle(reg_of_det, (x, y), (x + w, y + h), (0, 255, 0), 2)


    # Обнаружение объектов в кадре

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    # Выводим информацию на экран
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:
                # Координаты границ обнаруженного объекта
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Координаты границ
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    # Устроняем несколько ограничивающих рамок
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

    # В выделенной зоне отрисовываем
    # границы объектов и их id
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 2, (255, 255, 255), 2)

    # Выводим частоту кадров на экран
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 70), font, 2, (0, 0, 0), 2)

    # Выводим окно маски, выделенной зоны и кадра
    cv2.imshow("Mask", mask)
    cv2.imshow("Detector of people in a certain area", reg_of_det)
    cv2.imshow("Video", frame)
    key = cv2.waitKey(25)
    if key & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()