import pygame
import cv2
import mediapipe as mp
import math
import random
import time
import numpy as np
pygame.init()

# Impostazioni del gioco
screen_width, screen_height = 1920, 1080
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()


# Carica l'asset del gioco
car_image = pygame.image.load("boattt.png").convert_alpha()
original_car_width, original_car_height = car_image.get_size()
car_width, car_height = 300, 200  # Nuove dimensioni della barca
car_image = pygame.transform.scale(car_image, (car_width, car_height)) #regola grandezza barca
car_image = pygame.transform.flip(car_image, True, False) #trasla la barca per visualizzarla al dritto

#car_mask = pygame.mask.from_surface(car_image)
# Posizione iniziale della barca
car_image_x = 0
car_image_y = (screen_height - car_height) // 2
# Carica l'immagine di sfondo
background1 = pygame.image.load("watertex.jpg")
background2 = pygame.transform.flip(background1, True, False)
background_width, background_height = background1.get_size()
overlap = 0  # Lunghezza della zona di sovrapposizione
# Posizioni iniziali delle due copie dell'immagine di sfondo


background1_x, background2_x = 0, background_width

# Carica le immagini degli ostacoli con le relative maschere di collisione
obstacle_images = [pygame.image.load("barrel.png").convert_alpha(),
    # Aggiungi altre immagini degli ostacoli con le relative maschere...
    ]
# Asset sulla destra
obstacles = []
obstacle_width, obstacle_height = 50, 60
obstacle_images[0] = pygame.transform.scale(obstacle_images[0], (obstacle_width, obstacle_height)) #regola grandezza ostacolo
obstacle_speed = 10
coin_image = pygame.image.load("coin.png").convert_alpha()
coins = []
coin_width, coin_height = 50, 60
coin_image = pygame.transform.scale(coin_image, (coin_width, coin_height)) # regola grandezza soldo
coin_speed = 10
# Font per il testo "Game Over"
game_over_font = pygame.font.Font(None, 100)

bullet_x = car_image_x + car_width  # Posizione iniziale della pallina
bullet_y = car_image_y + car_height // 2  # Posizione iniziale della pallina

# Dichiarazione delle variabili
bullets = []  # Lista delle palline
bullet_speed = 20  # Velocità di spostamento delle palline
last_shot_time = time.time()
shoot_cooldown = 0.75  # Tempo in secondi tra uno sparo e l'altro
obstacle_spawn_prob = 0.01  # Probabilità iniziale di spawn
obstacle_spawn_interval = 3  # Intervallo di tempo per aumentare la probabilità (in secondi)
obstacle_last_spawn_time = time.time()
punteggio_monete = 0
coin_spawn_prob = 0.005  # Probabilità iniziale di spawn
coin_spawn_interval = 4  # Intervallo di tempo per aumentare la probabilità (in secondi)
coin_last_spawn_time = time.time()
start_time = pygame.time.get_ticks()  # Ottieni il tempo di inizio in millisecondi
distanza_percorsa = 0

vite = 3
cuori = []
cuore_image = pygame.image.load("cuore.png").convert_alpha()
cuore_width, cuore_height = 50, 50  # Regola le dimensioni del cuore a tuo piacimento
cuore_image = pygame.transform.scale(cuore_image, (cuore_width, cuore_height))
cuore_spawn_prob = 0.001  # Probabilità iniziale di spawn
cuore_spawn_interval = 5  # Intervallo di tempo per aumentare la probabilità (in secondi)
cuore_last_spawn_time = time.time()

max_obstacles = 15  # Modifica il numero massimo di ostacoli desiderato
max_coins = 5      # Modifica il numero massimo di monete desiderato
max_cuori = 1      # Modifica il numero massimo di cuori desiderato




def check_collision(rect1, rect2):
    return rect1.colliderect(rect2)  #vede se le due immagini si sovrappongono (collidono)
# Funzione per visualizzare il testo "Game Over"
def show_game_over_text():
    game_over_text = game_over_font.render("Game Over", True, (255, 0, 0)) 
    screen.blit(game_over_text, (screen_width // 2 - 200, screen_height // 2 - 50)) #mostra la scritta game over
    pygame.display.flip() #aggiorna il display del gioco per ricaricare tutta la grafica
    pygame.time.wait(2000)  # Attendere 2 secondi
    reset_game()

# Funzione per reimpostare la posizione della nave e degli asset e tutti gli altri parametri
def reset_game():
    global car_image_x, car_image_y, obstacles, coins, cuori, bullets, obstacle_spawn_prob, coin_spawn_prob, distanza_percorsa, punteggio_monete, vite,cuore_spawn_prob
    car_image_x = 0
    car_image_y = (screen_height - car_height) // 2
    obstacles = []
    coins = []
    bullets = []
    cuori=[]
    vite=3
    obstacle_spawn_prob = 0.01  # Puoi reimpostare la probabilità iniziale di spawn se necessario
    coin_spawn_prob = 0.005
    cuore_spawn_prob = 0.001
    distanza_percorsa = 0
    punteggio_monete = 0
# Impostazioni della mano, TODO: provare a fare fine-tuning sugli ultimi due parametri
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.2, #threshold di rilevamento mani
    min_tracking_confidence=0.02 #threshold di tracking del mani in movimento
)
#avvia telecamera SE NE HAI UNA ESTERNA USA QUELLA ALTRIMENTI QUELLA PREDEFINITA
cap = cv2.VideoCapture(0)

# Aggiorna la direzione di sterzata
steering_threshold_lower = -10
steering_threshold_upper = 10






while cap.isOpened():
    clock.tick(30)  # Imposta il numero di fotogrammi al secondo desiderato
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #converti frame in immagine a colori
    frame_rgb = cv2.GaussianBlur(frame_rgb, (5, 5), 0) #applica filtro per aiutare il rilevamento delle mani

    results = hands.process(frame_rgb)

    # ... Resto del codice di rilevamento delle mani ...
    left_hand_status = "Unknown"
    right_hand_status = "Unknown"
    steering_direction = "Straight"
    line_angle = 0.0


    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_landmarks = hand_landmarks.landmark

            wrist = hand_landmarks[mp_hands.HandLandmark.WRIST]

            hand_tips = [                                                                          # ORDER : THUMB,INDEX,MIDDLE,RING,PINKY
                                    hand_landmarks[mp_hands.HandLandmark.THUMB_TIP],                #calcola coordinate della giuntura tip (estremità) del pollice
                                    hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP],         # ripeti per indice
                                    hand_landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],        #ecc...
                                    hand_landmarks[mp_hands.HandLandmark.RING_FINGER_TIP],
                                    hand_landmarks[mp_hands.HandLandmark.PINKY_TIP]
                        ]

            hand_bottom= [
                                    hand_landmarks[mp_hands.HandLandmark.THUMB_MCP], #calcola coordinate della giuntura mcp (base) del pollice
                                    hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP], # ripeti per indice
                                    hand_landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP], # ecc...
                                    hand_landmarks[mp_hands.HandLandmark.RING_FINGER_MCP],
                                    hand_landmarks[mp_hands.HandLandmark.PINKY_MCP]
                            ]

            for e in hand_tips:
                cv2.circle(frame, (int(e.x * frame.shape[1]),int(e.y  * frame.shape[0])), 1, (255,0,0), 2)
            
            for e in hand_bottom:
                cv2.circle(frame, (int(e.x * frame.shape[1]),int(e.y* frame.shape[0])), 1, (255,0,0), 2)

            cv2.circle(frame, (int(wrist.x * frame.shape[1]),int(wrist.y* frame.shape[0])), 1, (255,0,0), 2)

            

            threshold_thumb     = 0.05
            threshold_thumb_2   = 0
            threshold_thumb_3   = 0.05
            threshold_general   = 0.05
           

            open=True
            for i in range(1,5):                                            #ESCLUDIAMO IL POLLICE DA QUESTO CALCOLO
                if ((hand_tips[i].y-hand_bottom[i].y)> (hand_bottom[i].y -wrist.y - threshold_general) and 
                   (hand_tips[i].y-hand_bottom[i].y)< (hand_bottom[i].y -wrist.y + threshold_general)):
                        open=True
                else:
                        open=False
                
            #print(hand_tips*frame.shape[1])
            if (open and ((hand_tips[0].y )> (hand_bottom[1].y - threshold_thumb))  and 
               (hand_bottom[0].y < wrist.y - threshold_thumb_2)                    and
               (hand_tips[0].y - hand_tips[1].y)> threshold_thumb_3    ):#and (hand_tips[0].y > (hand_tips[1].y - threshold_thumb ) and hand_tips[0].y < (hand_tips[1].y + threshold_thumb )) ):
                    if hand_tips[0].x < hand_bottom[1].x:
                        print (f"Left Open [thumb: {hand_tips[0].y}     middle_b : {hand_bottom[2].y}]",end="\r")
                        left_hand_status="Open"
                        left_hand_tips     = hand_tips
                        left_hand_bottoms  = hand_bottom
                        thumb2= (int(hand_bottom[1].x * frame.shape[1]), int(hand_bottom[1].y * frame.shape[0]))
                    else:
                        right_hand_status="Open"
                        right_hand_tips     = hand_tips
                        right_hand_bottoms  = hand_bottom
                        thumb1  = (int(hand_bottom[1].x * frame.shape[1]), int(hand_bottom[1].y * frame.shape[0]))
            else :            
                if hand_tips[0].x < hand_bottom[1].x:
                    left_hand_status="Closed"
                    thumb2= (int(hand_bottom[1].x * frame.shape[1]), int(hand_bottom[1].y * frame.shape[0]))
                else:
                    thumb1  = (int(hand_bottom[1].x * frame.shape[1]), int(hand_bottom[1].y * frame.shape[0]))
                    right_hand_status="Closed"


        # INIZIO CODICE PER STERZO (NON FUNZIONANTE !!!!!!!!!!!!)

        try:
            if(left_hand_status == "Closed" and right_hand_status=="Closed"):
                center_point = ((thumb1[0] + thumb2[0]) // 2, (thumb1[1] + thumb2[1]) // 2)
                line_angle = math.degrees(math.atan2(thumb2[1] - thumb1[1], thumb2[0] - thumb1[0]))

            # Disegna un asse tra i due pollici
                cv2.line(frame, thumb1, thumb2, (0, 255, 0), 2)
                #mostra a schermo l'angolo
                cv2.putText(frame, f"Line Angle: {line_angle:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    #determina la direzione di sterzata sulla base dell'angolo trovato
                if steering_threshold_lower < line_angle < steering_threshold_upper:
                        steering_direction = "Straight"
                elif line_angle >= steering_threshold_upper:
                        steering_direction = "Right"
                else:
                        steering_direction = "Left"
        except Exception as e:
                print(str(e))
        

        


    # Aggiungi la scritta sullo stato della mano sopra il frame
    cv2.putText(frame, f"Left Hand Status: {left_hand_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Right Hand Status: {right_hand_status}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Steering Direction: {steering_direction}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Tracking', frame)

    #IMPORTANTE: quì inizia la parte di gioco
    # Aggiorna la posizione della barca lungo l'asse y in base all'inclinazione della linea
    speed = 25  # Regola la velocità di spostamento

    if steering_threshold_lower < line_angle < steering_threshold_upper:
        delta_y = 0
    elif line_angle >= steering_threshold_upper:
        delta_y = int(speed * math.sin(math.radians(line_angle)))
    else:
        delta_y = int(speed * math.sin(math.radians(line_angle)))
    new_car_image_y = car_image_y - delta_y

    # Verifica se la nuova posizione è all'interno dei limiti dello schermo
    if 0 <= new_car_image_y <= screen_height - car_height:
        car_image_y = new_car_image_y
    # Disegna l'asset del gioco sulla schermata
    #screen.fill((255, 255, 255))  # Sfondo bianco
    
    
    speed=10
    # Sposta le immagini in orizzontale
    background1_x -= speed
    background2_x -= speed

    # Se un'immagine raggiunge la fine della finestra, riportala alla posizione iniziale
    if background1_x <= -background_width:
        background1_x = background2_x + background_width

    if background2_x <= -background_width:
        background2_x = background1_x + background_width


    # Disegna le immagini nello sfondo
    screen.blit(background1, (background1_x, 0))
    screen.blit(background2, (background2_x, 0))




    # Aggiorna la lista degli ostacoli
    for obstacle in obstacles:
        obstacle['x'] -= obstacle_speed
    # Rimuovi gli ostacoli che sono usciti dallo schermo
    obstacles = [obstacle for obstacle in obstacles if obstacle['x'] + obstacle_width > 0]
    # Aggiorna la lista degli ostacoli
    
    for coin in coins:
        coin['x'] -= coin_speed
    # Rimuovi gli ostacoli che sono usciti dallo schermo
    coins = [coin for coin in coins if coin['x'] + coin_width > 0]
    # Aggiungi nuovi ostacoli casuali con probabilità basata sul tempo trascorso
    current_time = time.time()
    time_since_last_spawn = current_time - obstacle_last_spawn_time
    time_since_last_spawn_coin = current_time - coin_last_spawn_time
    time_since_last_spawn_cuore = current_time - cuore_last_spawn_time
    # Aggiorna la lista degli ostacoli
    cuore_speed = 10
    for cuore in cuori:
        cuore['x'] -= cuore_speed
    # Rimuovi gli ostacoli che sono usciti dallo schermo
    cuori = [cuore for cuore in cuori if cuore['x'] + cuore_width > 0]


    #obstacle_last_spawn_time = current_time - obstacle_last_spawn_time
    if len(obstacles)<max_obstacles and random.random() < obstacle_spawn_prob:
        new_obstacle = {
            'image': random.choice(obstacle_images),
            'x': screen_width,
            'y': random.randint(0, screen_height - obstacle_height),
        }
        obstacles.append(new_obstacle)
    # Aumenta progressivamente la probabilità di spawn
    if time_since_last_spawn > obstacle_spawn_interval:
        obstacle_spawn_prob *= 1.1  # Puoi modificare il fattore di aumento come preferisci
        obstacle_last_spawn_time = current_time
    if len(coins)<max_coins and random.random() < coin_spawn_prob:
        new_coin = {
            'image': coin_image,
            'x': screen_width,
            'y': random.randint(0, screen_height - coin_height),
        }
        coins.append(new_coin)
    # Aumenta progressivamente la probabilità di spawn
    #coin_last_spawn_time = current_time - coin_last_spawn_time
    if time_since_last_spawn_coin > coin_spawn_interval:
        coin_spawn_prob *= 1.1  # Puoi modificare il fattore di aumento come preferisci
        coin_last_spawn_time = current_time
        
    if len(cuori)<max_cuori and random.random() < cuore_spawn_prob:
        new_cuore = {
            'image': cuore_image,
            'x': screen_width,
            'y': random.randint(0, screen_height - cuore_height),
        }
        cuori.append(new_cuore)
    # Aumenta progressivamente la probabilità di spawn
    #cuore_last_spawn_time = current_time - cuore_last_spawn_time
    if time_since_last_spawn_cuore > cuore_spawn_interval:
        cuore_spawn_prob *= 1.1  # Puoi modificare il fattore di aumento come preferisci
        cuore_last_spawn_time = current_time
        
        
    # Disegna gli ostacoli
    for obstacle in obstacles:
        screen.blit(obstacle['image'], (obstacle['x'], obstacle['y']))
    # Disegna gli ostacoli
    for coin in coins:
        screen.blit(coin['image'], (coin['x'], coin['y']))
    for cuore in cuori:
        screen.blit(cuore['image'], (cuore['x'], cuore['y']))
    # Disegna l'asset del gioco sulla schermata
    screen.blit(car_image, (car_image_x, car_image_y))
    # Verifica collisione con ostacoli
    car_rect = pygame.Rect(car_image_x, car_image_y, car_width, car_height)
    for obstacle in obstacles:
        obstacle_rect = pygame.Rect(obstacle['x'], obstacle['y'], obstacle_width, obstacle_height)
        if check_collision(car_rect, obstacle_rect):
            vite -= 1
            #Rimuovi l'ostacolo colpito
            obstacles.remove(obstacle)
            if vite <= 0:
                show_game_over_text()


    for coin in coins:
        coin_rect = pygame.Rect(coin['x'], coin['y'], coin_width, coin_height)
        if check_collision(car_rect, coin_rect):
            # La nave ha colpito una moneta, incrementa il punteggio
            punteggio_monete += 1
            coins.remove(coin)  # Rimuovi la moneta dalla lista
    for cuore in cuori:
       cuore_rect = pygame.Rect(cuore['x'], cuore['y'], cuore_width, cuore_height)
       if check_collision(car_rect, cuore_rect):
           # La nave ha preso un cuore, incrementa le vite
           if(vite<3):
               vite += 1
           cuori.remove(cuore)  # Rimuovi il cuore dalla lista
    for bullet in bullets:
        bullet['x'] += bullet_speed
        bullet_radius = 18
        pygame.draw.circle(screen, (0, 0, 0), (int(bullet['x']), int(bullet['y'])), bullet_radius)  # Crea una pallina nera
        # Verifica collisione con ostacoli
        bullet_rect = pygame.Rect(bullet['x'], bullet['y'], bullet_radius, bullet_radius)
        for obstacle in obstacles:
            obstacle_rect = pygame.Rect(obstacle['x'], obstacle['y'], obstacle_width, obstacle_height)
            if check_collision(bullet_rect, obstacle_rect):
                # Rimuovi l'ostacolo colpito
                obstacles.remove(obstacle)
                # Rimuovi la pallina
                if(len(bullets)>0):
                    bullets.remove(bullet)
                break  # Esci dal ciclo degli ostacoli dopo la prima collisione
        for coin in coins:
            coin_rect = pygame.Rect(coin['x'], coin['y'], coin_width, coin_height)
            if check_collision(bullet_rect, coin_rect):
                # Rimuovi l'ostacolo colpito
                coins.remove(coin)
                # Rimuovi la pallina
                if(len(bullets)>0):
                    bullets.remove(bullet)
                break  # Esci dal ciclo degli ostacoli dopo la prima collisione
        for cuore in cuori:
            cuore_rect = pygame.Rect(cuore['x'], cuore['y'], cuore_width, cuore_height)
            if check_collision(bullet_rect, cuore_rect):
                if(vite<3):
                    vite += 1
                # Rimuovi l'ostacolo colpito
                cuori.remove(cuore)
                # Rimuovi la pallina
                if(len(bullets)>0):
                    bullets.remove(bullet)
                break  # Esci dal ciclo degli ostacoli dopo la prima collisione
    # Dopo la parte del rilevamento della mano
    current_time = time.time()
    if left_hand_status == "Open" or right_hand_status == "Open":
        if current_time - last_shot_time > shoot_cooldown:
            last_shot_time = current_time
            new_bullet = {'x': car_image_x + car_width, 'y': car_image_y + car_height // 2}
            bullets.append(new_bullet)
    # Nel ciclo principale del gioco
    current_time = pygame.time.get_ticks()  # Ottieni il tempo corrente in millisecondi
    elapsed_time = (current_time - start_time) / 1000  # Tempo trascorso in secondi

    # Ogni 5 secondi, aggiorna la distanza percorsa
    if elapsed_time >= 1:
        distanza_percorsa += 1
        start_time = current_time  # Resetta il tempo di inizio per il prossimo intervallo di 5 secondi
    font = pygame.font.Font(None, 36)  # Scegli una dimensione del carattere
    punteggio_text = font.render(f"Monete: {punteggio_monete}", True, (255, 255, 255))
    distanza_text = font.render(f"Distanza: {distanza_percorsa} metri", True, (255, 255, 255))
    vite_text = font.render(f"Vite: {vite}", True, (255, 255, 255))
    screen.blit(punteggio_text, (10, 10))
    screen.blit(distanza_text, (10, 50))
    screen.blit(vite_text, (10, 100))
    pygame.display.flip()    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            cv2.destroyAllWindows()
            pygame.quit()
            quit()
    
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
