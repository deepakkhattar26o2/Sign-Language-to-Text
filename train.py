import cv2
import mediapipe as mp
import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import TensorBoard 

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # converting color from bgr to rgb
    image.flags.writeable = False #saves memory during processing   
    results = model.process(image)#make prediction poses
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22, 76), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(121,44, 250), thickness=1, circle_radius=1),)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117, 66), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(245,66, 230), thickness=1, circle_radius=1),)

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

#Folders for collections
DATA_PATH = os.path.join("collection")

#Actions that we try to detect
actions = np.array(['YES', 'iloveyou', 'HELLO'])
#Thirty videos worth of data
no_sequences = 30
#Videos are going to be 30 frames in length
sequence_length = 30

label_map = {label : num for num, label in enumerate(actions)}

sequences, labels = [], []

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

log_dir = os.path.join("Logs")
tb_callback = TensorBoard(log_dir=log_dir)

X = np.array(sequences)
Y = to_categorical(labels).astype(int)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)

model = Sequential()

model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))

model.add(LSTM(128, return_sequences=True, activation='relu'))

model.add(LSTM(64, return_sequences=False, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(32, activation='relu'))

model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy']) #categorical used for multi-class classification

model.fit(X_train, Y_train, epochs=2000, callbacks=tb_callback)

model.save('demo6.h5')
