import time
import cv2
import mediapipe.python.solutions as mp
import numpy as np
import pyautogui as pya
from decimal import *
from win32api import GetSystemMetrics
import math
from typing import *

pya.FAILSAFE = False


def decimal_rounding(num, decimals: int = 0):
    num = Decimal(str(num))
    decimals = Decimal(str('1.' + '0' * decimals))
    return float(num.quantize(decimals))


def get_angle(landmarks: np.ndarray, connections: list[tuple[int, int]]) -> float:
    def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.arccos(dot / norm)

    feature_vectors = np.array(list(
        map(lambda cnt: landmarks[cnt[1]] - landmarks[cnt[0]], connections)))  # grab 3D vectors for each connection

    angle = angle_between(v1=feature_vectors[0], v2=feature_vectors[1])
    return angle * 180 / math.pi  # convert rad to degrees


def scale_coords(point: Tuple[float, float], dims: Tuple[int, int], translate: Tuple[int, int] = (0, 0)) -> Tuple:
    return int(point[0] * dims[0]) + translate[0], int(point[1] * dims[1]) + translate[1]


def load_configs():
    with open('configs', 'r') as f:
        return {split[0]: eval(split[1]) for line in f.readlines() if (split := line.replace('\n', '').split('='))}


if __name__ == '__main__':
    # Program parameters
    config = load_configs()
    print(config)
    move_distance_threshold: int = 5  # min difference in pixels required to move
    bbx_scale = .4  # scale of full res width, .1-1# initialize key variables
    smoothing = 3

    with mp.hands.Hands(static_image_mode=False, model_complexity=1, max_num_hands=1, min_detection_confidence=0.7,
                        min_tracking_confidence=0.7) as hands:

        # create access references
        mp_drawing = mp.drawing_utils
        mp_drawing_styles = mp.drawing_styles
        DrawingSpec: object = mp.drawing_utils.DrawingSpec  # DrawingSpec(color=(B, G, R))

        # determine dimensions of display
        disp_dim = (GetSystemMetrics(0), GetSystemMetrics(1))
        cap = cv2.VideoCapture(0)
        success, image = cap.read()
        cap_dim = image.shape[:2][::-1]
        print(f'WinRes={disp_dim}')
        print(f'CapRes={cap_dim}')

        # initialize movement box
        bbx_scale = max(.1, min(bbx_scale, 1))  # lock between .1-1
        bbx_offset_x, bbx_offset_y = round((1 - bbx_scale) / 2, 2), round((1 - bbx_scale) / 2, 2)  # begin in center
        bounding_box = (
            [bbx_offset_x, bbx_offset_y], [bbx_offset_x + bbx_scale, bbx_offset_y + bbx_scale])  # saved as ratio

        # runtime variable
        key_lmd = {
            'thumb_click': [(1, 2), (2, 3)], 'index_click': ((5, 6), (6, 7)), 'middle_click': ((9, 10), (10, 11)),
            'ring_click': ((13, 14), (14, 15)), 'pinky_click': ((17, 18), (18, 19)), 'index_tip': 8, 'index_knuckle': 6,
            'middle_tip': 12, 'middle_knuckle': 10, 'index_base': 5, 'thumb_tip': 4
            }
        pTime = 0
        locked = False
        thumb_active: bool = False
        left_active: bool = False
        right_active: bool = False
        ring_active: bool = False
        pinky_active: bool = False

        print('Starting Detection')

        while True:  # cap.isOpened():
            success, image = cap.read()
            if not success:
                print('Ignoring empty camera frame.')
                continue
            image = cv2.flip(image, 1)

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # convert into (21, 3) nparray with (x, y, z)
                    landmark_list: List[List[float]] = np.array([[lmd.x, lmd.y, lmd.z] for lmd in hand_landmarks.landmark])

                    # calculate position relative to bounding box
                    pointer_abs_pos: List[float] = landmark_list[key_lmd['index_base']]
                    uncapped_rel_pos = (
                        round((pointer_abs_pos[0] - bounding_box[0][0]) / (bounding_box[1][0] - bounding_box[0][0]), 2),
                        round((pointer_abs_pos[1] - bounding_box[0][1]) / (bounding_box[1][1] - bounding_box[0][1]), 2))
                    capped_rel_pos: Tuple[float] = max(0.0, min(1.0, uncapped_rel_pos[0])), max(0.0, min(1.0, uncapped_rel_pos[
                        1]))  # apply floor and ceiling at 0, 1

                    angle_thumb: float = get_angle(landmark_list, key_lmd['thumb_click'])
                    angle_index: float = get_angle(landmark_list, key_lmd['index_click'])
                    angle_middle: float = get_angle(landmark_list, key_lmd['middle_click'])
                    angle_ring: float = get_angle(landmark_list, key_lmd['ring_click'])
                    angle_pinky: float = get_angle(landmark_list, key_lmd['pinky_click'])

                    # EVENTS

                    # all 5 fingers down, close
                    if angle_thumb >= 30 and angle_index >= 60 and angle_middle >= 60 and angle_ring >= 60 and angle_pinky >= 60:
                        print('ENDING')
                        pya.mouseUp(button='right', _pause=False)
                        pya.mouseUp(button='left', _pause=False)
                        quit()

                    # thumb bent, lock inputs
                    if config['thumb_lock']:
                        if angle_thumb >= 30 and not thumb_active:  # finger down
                            thumb_active = True
                            locked = not locked
                            print('LOCKED' if locked else 'UNLOCKED')
                        if angle_thumb <= 10:
                            if thumb_active:
                                thumb_active = False

                    # hand moving
                    if config['move'] and not locked:
                        mouse_pos = pya.position()
                        new_pos = scale_coords(point=capped_rel_pos, dims=disp_dim)

                        # dynamic bbx
                        x_tran = 0
                        y_tran = 0
                        # x pos goes outside box, shift box
                        if uncapped_rel_pos[0] > 1:
                            x_tran = (uncapped_rel_pos[0] - 1) * bbx_scale
                        elif uncapped_rel_pos[0] < 0:
                            x_tran = uncapped_rel_pos[0] * bbx_scale
                        # y pos goes outside box, shift box
                        if uncapped_rel_pos[1] > 1:
                            y_tran = (uncapped_rel_pos[1] - 1) * bbx_scale
                        elif uncapped_rel_pos[1] < 0:
                            y_tran = uncapped_rel_pos[1] * bbx_scale
                        for p in bounding_box:
                            p[0] += x_tran if 0 < bounding_box[0][0] + x_tran and bounding_box[1][0] + x_tran < 1 else 0
                            p[1] += y_tran if 0 < bounding_box[0][1] + y_tran and bounding_box[1][1] + y_tran < 1 else 0

                        if abs(new_pos[0] - mouse_pos[0] + new_pos[1] - mouse_pos[1]) >= move_distance_threshold:
                            c1 = mouse_pos[0] + (new_pos[0] - mouse_pos[0]) / smoothing
                            c2 = mouse_pos[1] + (new_pos[1] - mouse_pos[1]) / smoothing
                            pya.moveTo(c1, c2, _pause=False)

                    # left click from index
                    if config['left_click'] and not locked:
                        if angle_index >= 60:  # finger down
                            if not left_active:
                                print('LEFT DOWN')
                                pya.mouseDown(button='left', _pause=False)
                                left_active = True
                        elif angle_index <= 40:  # finger up
                            if left_active:
                                print('LEFT UP')
                                pya.mouseUp(button='left', _pause=False)
                                left_active = False

                    # right click from middle
                    if config['right_click'] and not locked:
                        if not right_active:
                            if angle_middle >= 60:  # finger down
                                pya.mouseDown(button='right', _pause=False)
                                right_active = True
                        elif right_active:
                            if angle_middle <= 40:  # finger up
                                pya.mouseUp(button='right', _pause=False)
                                right_active = False

                    # scoll up and down through ring and pinky
                    if config['scroll'] and not locked:
                        if not pinky_active and angle_ring >= 60:  # finger down
                            ring_active = True
                            pya.scroll(-100, _pause=False)
                        elif not ring_active and angle_pinky >= 60:  # finger down
                            pinky_active = True
                            pya.scroll(100, _pause=False)
                        if angle_ring <= 40:  # finger up
                            ring_active = False
                        if angle_pinky <= 40:  # finger up
                            pinky_active = False

                    # display graphics
                    if config['display']:
                        # draw all hand connections
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp.hands.HAND_CONNECTIONS,
                                                  landmark_drawing_spec=DrawingSpec(color=(255, 255, 255), thickness=2,
                                                                                    circle_radius=1),
                                                  connection_drawing_spec=DrawingSpec(color=(0, 0, 0), thickness=2))

                        # Highlight fingers when active
                        mp_drawing.draw_landmarks(image, hand_landmarks, [(1, 2), (2, 3), (3, 4)],
                                                  landmark_drawing_spec=DrawingSpec(color=(255, 255, 255), thickness=2,
                                                                                    circle_radius=1),
                                                  connection_drawing_spec=DrawingSpec(
                                                      color=(0, 255, 0) if thumb_active else (211, 0, 148),
                                                      thickness=2))
                        mp_drawing.draw_landmarks(image, hand_landmarks, [(5, 6), (6, 7), (7, 8)],
                                                  landmark_drawing_spec=DrawingSpec(color=(255, 255, 255), thickness=2,
                                                                                    circle_radius=1),
                                                  connection_drawing_spec=DrawingSpec(
                                                      color=(0, 255, 0) if left_active else (0, 0, 255), thickness=2))
                        mp_drawing.draw_landmarks(image, hand_landmarks, [(9, 10), (10, 11), (11, 12)],
                                                  landmark_drawing_spec=DrawingSpec(color=(255, 255, 255), thickness=2,
                                                                                    circle_radius=1),
                                                  connection_drawing_spec=DrawingSpec(
                                                      color=(0, 255, 0) if right_active else (255, 0, 0), thickness=2))
                        mp_drawing.draw_landmarks(image, hand_landmarks, [(13, 14), (14, 15), (15, 16)],
                                                  landmark_drawing_spec=DrawingSpec(color=(255, 255, 255), thickness=2,
                                                                                    circle_radius=1),
                                                  connection_drawing_spec=DrawingSpec(
                                                      color=(0, 255, 0) if ring_active else (0, 255, 255), thickness=2))
                        mp_drawing.draw_landmarks(image, hand_landmarks, [(17, 18), (18, 19), (19, 20)],
                                                  landmark_drawing_spec=DrawingSpec(color=(255, 255, 255), thickness=2,
                                                                                    circle_radius=1),
                                                  connection_drawing_spec=DrawingSpec(
                                                      color=(0, 255, 0) if pinky_active else (211, 0, 148),
                                                      thickness=2))

                        # draw dots on key landmarks
                        cv2.circle(image, scale_coords(point=landmark_list[6], dims=cap_dim), radius=1, thickness=5,
                                   color=(0, 0, 255))
                        cv2.circle(image, scale_coords(point=landmark_list[10], dims=cap_dim), radius=1, thickness=5,
                                   color=(255, 0, 0))
                        cv2.circle(image, scale_coords(point=pointer_abs_pos, dims=cap_dim), radius=1, thickness=5,
                                   color=(255, 255, 0))

                        # update and redraw the bounding box
                        web_bbx = tuple(map(lambda coord: scale_coords(point=coord, dims=cap_dim), bounding_box))
                        cv2.rectangle(image, web_bbx[0], web_bbx[1], color=(255, 255, 0), thickness=2)

                        # display  position, left angle, right angle
                        cv2.putText(image, f'{capped_rel_pos}', (0, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
                        cv2.putText(image, f'{round(angle_index, 2)}', (0, 80), cv2.FONT_HERSHEY_PLAIN, 2,
                                    (0, 0, 255) if not left_active else (0, 255, 0), 2)  # show finger angle
                        cv2.putText(image, f'{round(angle_middle, 2)}', (0, 120), cv2.FONT_HERSHEY_PLAIN, 2,
                                    (255, 0, 0) if not right_active else (0, 255, 0), 2)  # show finger angle
                        cv2.putText(image, f'{round(angle_ring, 2)}', (0, 160), cv2.FONT_HERSHEY_PLAIN, 2,
                                    (0, 255, 255) if not ring_active else (0, 255, 0), 2)  # show finger angle
                        cv2.putText(image, f'{round(angle_pinky, 2)}', (0, 200), cv2.FONT_HERSHEY_PLAIN, 2,
                                    (211, 0, 148) if not pinky_active else (0, 255, 0), 2)  # show finger angle

            if config['display']:
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                cv2.putText(image, f'{int(fps)}', (cap_dim[0] - 40, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                cv2.namedWindow('MediaPipe Hands', cv2.WINDOW_AUTOSIZE)
                if config['absolute_display']:
                    cv2.setWindowProperty('MediaPipe Hands', cv2.WND_PROP_TOPMOST, 1)
                cv2.imshow('MediaPipe Hands', image)

            if cv2.waitKey(5) & 0xFF == 27:  # esc
                pya.mouseUp(button='left')
                pya.mouseUp(button='right')
                quit()

    cap.release()
    cv2.destroyAllWindows()
