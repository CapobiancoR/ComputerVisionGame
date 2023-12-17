# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 15:37:58 2023

@author: Giorgio
"""

import cv2
import mediapipe as mp
import math
import pygame
import sys

# Inizializza Pygame
pygame.init()

# Impostazioni Pygame
window_size = (800, 600)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Hand Tracking Game")

# Inizializza il font di Pygame
font = pygame.font.Font(None, 36)

# Inizializza Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Impostazioni della pallina
ball_radius = 20
ball_color = (255, 0, 0)
ball_speed = 10
ball_direction = 0  # 0 per fermo, 1 per destra, -1 per sinistra

# Posizione iniziale della pallina
ball_x = window_size[0] // 2
ball_y = window_size[1] // 2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                ((thumb_tip.x - thumb_mcp.x) ** 2 + (thumb_tip.y - thumb_mcp.y) ** 2) ** 0.5 < threshold_distance_tip_mcp and
                ((index_tip.x - index_mcp.x) ** 2 + (index_tip.y - index_mcp.y) ** 2) ** 0.5 < threshold_distance_tip_mcp and
                ((middle_tip.x - middle_mcp.x) ** 2 + (middle_tip.y - middle_mcp.y) ** 2) ** 0.5 < threshold_distance_tip_mcp and
                ((ring_tip.x - ring_mcp.x) ** 2 + (ring_tip.y - ring_mcp.y) ** 2) ** 0.5 < threshold_distance_tip_mcp and
                ((pinky_tip.x - pinky_mcp.x) ** 2 + (pinky_tip.y - pinky_mcp.y) ** 2) ** 0.5 < threshold_distance_tip_mcp
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
                ball_direction = 0
            elif line_angle >= steering_threshold_upper:
                steering_direction = "Right"
                ball_direction = -1
            else:
                steering_direction = "Left"
                ball_direction = 1

    # Aggiorna la posizione della pallina in base all'angolo
    ball_x += ball_speed * ball_direction * (abs(line_angle)/90)*4

    # Controlla i limiti dello schermo per la pallina
    ball_x = max(ball_radius, min(window_size[0] - ball_radius, ball_x))


    # Pulisci lo schermo
    screen.fill((255, 255, 255)) 

    # Disegna la pallina
    pygame.draw.circle(screen, ball_color, (int(ball_x), int(ball_y)), ball_radius)

    # Aggiorna lo schermo
    pygame.display.flip()

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
