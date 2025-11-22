import cv2 as cv
import numpy as np
import mediapipe as mp
import tensorflow as tf

TFLITE_MODEL_PATH = "gesture.tflite"
LABELS_PATH = "labels_1.txt"

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    LABELS = [line.strip() for line in f.readlines()]

def preprocess_landmarks(landmarks_xy):
    lm = np.array(landmarks_xy, dtype=np.float32)
    base = lm[0].copy()
    lm -= base
    lm = lm.flatten()
    max_val = np.max(np.abs(lm))
    if max_val > 0:
        lm = lm / max_val
    return lm

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)


cap = cv.VideoCapture(0)

print("Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    img = cv.flip(frame, 1)
    h, w = img.shape[:2]

    # Convert to RGB
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    res = hands.process(rgb)

    label_show = "No Hand"

    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]

        landmarks_pixel = [
            [int(lm.x * w), int(lm.y * h)] for lm in hand.landmark
        ]

        # Draw the landmarks
        for x, y in landmarks_pixel:
            cv.circle(img, (x, y), 3, (0, 255, 0), -1)

        # Preprocess and infer
        inp = preprocess_landmarks(landmarks_pixel).astype(np.float32)
        inp = inp.reshape(1, -1)

        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])[0]
        pred_idx = int(np.argmax(output))
        label_show = LABELS[pred_idx]

    # Draw result
    cv.putText(
        img, f"Prediction: {label_show}",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1, (0, 255, 0), 2
    )

    # Display
    cv.imshow("Webcam Hand Gesture", img)

    # ESC để thoát
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
hands.close()
