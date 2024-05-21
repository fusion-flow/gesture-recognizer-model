#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import itertools

import cv2 as cv
import numpy as np
import mediapipe as mp
from model import KeyPointClassifier

cap = cv.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)


def main_call():

    while True:
        success, img = cap.read()
        img = cv.flip(img, 1)
        final_val, score = main(img)
        print("final_val: ", final_val, "score: ", score)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()


def main(img=None, min_detection_confidence=0.25, min_tracking_confidence=0.5):
    # Model load #############################################################

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    if img is None:
        raise ValueError("Image is not given")

    # Make sure the image is a valid numpy array
    if not isinstance(img, np.ndarray):
        raise ValueError("Invalid input type")

    # Ensure the image has three channels (BGR)
    if img.shape[2] != 3:
        raise ValueError("Invalid number of channels")

    keypoint_classifier = KeyPointClassifier()

   
    keypoint_classifier_labels = ["1", "2", "3", "4", "5"]
    # print(keypoint_classifier_labels)

    # image acquisition #####################################################
    image = img

    image = cv.flip(image, 1)
    debug_image = copy.deepcopy(image)

    # Detection implementation #############################################################
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    
    print("detecting hand")

    if results.multi_hand_landmarks is None:
        print("No hand")
        return 0, 0
    #  ####################################################################
    final_val = 0

    j = 0
    for hand_landmarks, handedness in zip(
        results.multi_hand_landmarks, results.multi_handedness
    ):
        if j == 2:
            break
        # Landmark calculation
        landmark_list = calc_landmark_list(debug_image, hand_landmarks)

        # Conversion to relative coordinates / normalized coordinates
        pre_processed_landmark_list = pre_process_landmark(landmark_list)

        # Hand sign classification
        hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
        print("hand_sign_id: ", hand_sign_id)
        final_val += int(keypoint_classifier_labels[hand_sign_id])
        j += 1

    # if both hands are availale take average of both confidence scores
    score = 0
    # print("results: ", results.multi_handedness)
    num_hands = len(results.multi_handedness)
    print("num_hands: ", num_hands)
    if num_hands == 2:
        score += results.multi_handedness[0].classification[0].score
        score += results.multi_handedness[1].classification[0].score
    else:
        score += results.multi_handedness[0].classification[0].score

    score /= 2
    # return with 4 decimal places
    print("final_val: ", final_val, "score: ", round(score, 4))
    return final_val, round(score, 4)


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = "model/keypoint_classifier/keypoint.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return


if __name__ == "__main__":
    main_call()
