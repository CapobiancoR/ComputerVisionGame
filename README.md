# 🏴‍☠️ Pirate of the Sea – Hand Gesture Recognition Game

A Computer Vision project developed as part of the **Computer Vision** course at *Sapienza University of Rome (2023/2024)*  
by **Alessio Buccarella**, **Riccardo Capobianco**, and **Giorgio Scavalli Veccia**.

---

## 🎯 Overview

**Pirate of the Sea** is an interactive real-time game that leverages **hand gesture recognition** to control the main character through webcam-based input.  
Players steer a pirate ship, collect coins, and shoot enemies using simple hand gestures — no physical controllers required.  

The system is built on **Google’s MediaPipe** framework, which provides efficient, pre-trained models for real-time hand tracking and landmark detection.

---

## 🧠 Core Idea

The project demonstrates how **hand pose estimation** can be used for real-time human–computer interaction.  
After detecting hand landmarks (keypoints), the system analyzes their **relative spatial positions** to infer whether the hand is **open**, **closed**, or performing a specific gesture.

Instead of relying solely on neural network classification, the system uses a **set of mathematical inequalities (disequations)** to identify hand configurations — a method **invented and implemented by Riccardo Capobianco**.

This approach allows:
- High interpretability (transparent decision-making)
- Lightweight real-time computation
- Robustness to variations in lighting and orientation

---

## 🧩 System Architecture

1. **Palm Detection**  
   MediaPipe detects the palm region and bounding box for each hand.

2. **Hand Landmark Detection**  
   The framework tracks **21 3D keypoints** for each hand (including wrist, finger joints, and tips).

3. **Hand Pose Analysis**  
   The custom **disequation-based system** computes relative distances and positions between landmarks to classify gestures.

4. **Gesture-to-Action Mapping**  
   Each recognized pose triggers a specific in-game behavior:
   - ✋ **Open hand** → Shoot  
   - 🤚 **Closed hand** → Drive  
   - 👍 **Thumb extended** → Special move  

---

## 🧮 Disequation-Based Hand Pose Recognition

The core innovation of this project lies in the **rule-based system of inequalities** designed by **Riccardo Capobianco**, used to recognize gestures through geometric analysis of the hand landmarks.

Each hand landmark is represented as a coordinate pair `(x, y)`.  
Relationships between these coordinates are formalized as **disequations** that define valid regions for each gesture configuration.

### ✋ Finger Validation System

For each finger `i`:

```python
if (hand_tips[i].y - hand_bottom[i].y) > (hand_bottom[i].y - wrist.y - threshold_general) and \
   (hand_tips[i].y - hand_bottom[i].y) < (hand_bottom[i].y - wrist.y + threshold_general):
    open = True
else:
    open = False
````

This verifies whether each finger’s tip is sufficiently above its base relative to the wrist — indicating an **extended (open)** finger.

---

### 👍 Thumb Position Validation

```python
if (open and (hand_tips[0].y > hand_bottom[1].y - threshold_thumb)
    and (hand_bottom[0].y < wrist.y - threshold_thumb_2)
    and (hand_tips[0].y - hand_tips[1].y) > threshold_thumb_3):
    # Thumb extended – open hand gesture
```

These inequalities ensure that the thumb is positioned within a valid angular range relative to the index finger and wrist.

---

### 🤚 Hand Side Identification

```python
if hand_tips[0].x < hand_bottom[1].x:
    left_hand_status = "Open"
else:
    right_hand_status = "Open"
```

This rule determines whether the detected hand is **left** or **right**, based on the relative horizontal positions of thumb and index finger.

---

### 🧭 Geometric Model Summary

| Validation Type       | Mathematical Condition                                                          | Purpose                            |
| --------------------- | ------------------------------------------------------------------------------- | ---------------------------------- |
| Finger validation     | `A.y > B.y - Th1` and `A.y < B.y + Th1`                                         | Checks if a finger is extended     |
| Hand first validation | Applies to all fingers 1–4                                                      | Determines open/closed status      |
| Thumb validation      | `Thumb_tip.y < Finger_bottom.y + Th2` and `Thumb_tip.y > Finger_bottom.y - Th2` | Ensures thumb position consistency |
| Side detection        | `thumb.x < index.x` or `thumb.x > index.x`                                      | Detects left/right hand            |
| Rotation detection    | `angle = arctan((yB - yA) / (xB - xA))`                                         | Computes hand rotation degrees     |

This system transforms the gesture recognition process into a **set of geometric constraints**, enabling **deterministic**, **explainable**, and **fast** classification of hand poses.

---

## 🔢 Mathematical Foundation

Each gesture can be represented as a **constraint satisfaction problem (CSP)** where:

* Variables = keypoint coordinates
* Constraints = inequalities between coordinates

Example for an open hand:
[
\forall i \in {1,2,3,4}, \quad (y_{tip_i} - y_{base_i}) > \delta_{min} \quad \land \quad (y_{tip_i} - y_{base_i}) < \delta_{max}
]

If all constraints are satisfied → **Gesture = OPEN HAND**.

---

## 🕹️ Gameplay Integration

Each camera frame is analyzed in real-time:

* Landmarks are extracted via MediaPipe
* Geometric constraints are evaluated
* Recognized gestures trigger in-game events

### Game Controls

| Gesture        | Action                |
| -------------- | --------------------- |
| ✋ Open hand    | **Shoot cannonballs** |
| 🤚 Closed hand | **Steer the ship**    |
| 👍 Thumb up    | **Catch coins**       |

The objective is to **navigate the sea**, **collect treasures**, and **avoid obstacles**, becoming the **richest pirate** on the ocean! 🌊💰

---

## 🧰 Technologies Used

* **Python**
* **OpenCV** – video capture and visualization
* **MediaPipe** – real-time hand detection and keypoint tracking
* **NumPy / Math** – numerical operations and angle computation
* **Custom rule-based system** – inequality-based gesture recognition

---

## 📈 Results

* Accurate gesture recognition based purely on geometric constraints
* Real-time performance on standard laptop webcams
* Stable left/right hand detection and rotation estimation

This hybrid approach achieves **interpretable and efficient gesture recognition** without relying on deep neural network training.

---

## 👥 Authors & Contributions

* **Riccardo Capobianco** – *System architect & inventor of the disequation-based gesture recognition logic, MediaPipe integration/ CV pipeline*
* **Giorgio Scavalli Veccia** – *Game mechanics, UX, and performance optimization*

---

> “A pirate controls his ship not with a wheel, but with motion and logic.” ⚓
> — *Pirate of the Sea, 2024*


---

