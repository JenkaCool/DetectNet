#
#  A program for tracking objects that have been identified in the selected area.
#
#  Copyright (c) 2021, Evgeniya Balyuk <balyuk@petrsu.ru>
#
#  This code is licensed under a MIT-style license.

import math


class EuclideanDistTracker:
    def __init__(self):
        # Сохраняем центральные позиции объектов
        self.center_points = {}
        # Подсчитываем кол-во идентификаторов
        # Каждый раз, когда обнаруживается новый идентификатор объекта,
        # счетчик увеличивается на единицу
        self.id_count = 0


    def update(self, objects_rect):
        # Границы и идентификаторы объектов
        objects_bbs_ids = []

        # Получить центральную точку нового объекта
        # для каждого rect из objects_rect
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Проверяем, был ли этот объект уже обнаружен
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # Если обнаружен новый объект, мы добавляем ему идентификатор
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Очистить словарь по центральным точкам,
        # чтобы удалить идентификаторы, которые больше не используются
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Обновить словарь с удаленными идентификаторами
        self.center_points = new_center_points.copy()
        return objects_bbs_ids



