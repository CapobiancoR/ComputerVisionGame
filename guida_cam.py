# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 10:50:26 2023

@author: Giorgio
"""

import cv2
import mediapipe as mp
import math
from filterpy.kalman import KalmanFilter
import numpy as np



mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.02
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    

    frame_rgb = cv2.GaussianBlur(frame_rgb, (5, 5), 0)
    
    results = hands.process(frame_rgb)

    left_hand_status = "Unknown"
    right_hand_status = "Unknown"
    steering_direction = "Straight"
    line_angle = 0.0
    thumb1 = None
    thumb2 = None

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_landmarks = hand_landmarks.landmark
            
            thumb_tip = hand_landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks[mp_hands.HandLandmark.PINKY_TIP]

            thumb_mcp = hand_landmarks[mp_hands.HandLandmark.THUMB_MCP]
            index_mcp = hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            middle_mcp = hand_landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            ring_mcp = hand_landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
            pinky_mcp = hand_landmarks[mp_hands.HandLandmark.PINKY_MCP]

            index_pip = hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]

            # Definisci la distanza soglia tra i tip delle dita e i rispettivi MCP
            threshold_distance_tip_mcp = 0.8
            threshold_distance_pip_thumb_tip = 0.1

            hand_closed = (
                ((thumb_tip.x - thumb_mcp.x)**2 + (thumb_tip.y - thumb_mcp.y)**2)**0.5 < threshold_distance_tip_mcp and
                ((index_tip.x - index_mcp.x)**2 + (index_tip.y - index_mcp.y)**2)**0.5 < threshold_distance_tip_mcp and
                ((middle_tip.x - middle_mcp.x)**2 + (middle_tip.y - middle_mcp.y)**2)**0.5 < threshold_distance_tip_mcp and
                ((ring_tip.x - ring_mcp.x)**2 + (ring_tip.y - ring_mcp.y)**2)**0.5 < threshold_distance_tip_mcp and
                ((pinky_tip.x - pinky_mcp.x)**2 + (pinky_tip.y - pinky_mcp.y)**2)**0.5 < threshold_distance_tip_mcp and
                ((index_pip.x - thumb_tip.x)**2 + (index_pip.y - thumb_tip.y)**2)**0.5 < threshold_distance_pip_thumb_tip
            )

            # Determina se la mano è la sinistra o la destra in base alla posizione del pollice
            if thumb_tip.x < index_tip.x:
                left_hand_status = "Closed" if hand_closed else "Open"
                thumb2 = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
            else:
                right_hand_status = "Closed" if hand_closed else "Open"
                thumb1 = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))

        if thumb1 and thumb2 and left_hand_status == "Closed" and right_hand_status == "Closed":
            # Calcola l'inclinazione rispetto al punto centrale della linea
            center_point = ((thumb1[0] + thumb2[0]) // 2, (thumb1[1] + thumb2[1]) // 2)
            line_angle = math.degrees(math.atan2(thumb2[1] - thumb1[1], thumb2[0] - thumb1[0]))

            # Disegna un asse tra i due pollici
            cv2.line(frame, thumb1, thumb2, (0, 255, 0), 2)

            # Aggiorna la direzione di sterzata
            steering_threshold_lower = -10
            steering_threshold_upper = 10
            cv2.putText(frame, f"Line Angle: {line_angle:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if steering_threshold_lower < line_angle < steering_threshold_upper:
                steering_direction = "Straight"
            elif line_angle >= steering_threshold_upper:
                steering_direction = "Right"
            else:
                steering_direction = "Left"


    # Aggiungi la scritta sullo stato della mano sopra il frame
    cv2.putText(frame, f"Left Hand Status: {left_hand_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Right Hand Status: {right_hand_status}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Steering Direction: {steering_direction}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


