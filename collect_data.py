import cv2
import mediapipe as mp
import numpy as np
import os

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# an image and mp holistic model::returns landmarks as results
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

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

cap = cv2.VideoCapture(0)
#set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                #read frame
                ret, frame = cap.read()
                #make detections-cc
                image , results = mediapipe_detection(frame, holistic)
                #drawing real time landmarks
                draw_styled_landmarks(image, results)
                #Collection Logic
                if frame_num==0:
                    #image, text, position, font type, font size, color, line width, line type
                    cv2.putText(image, 'Starting Collection', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15 ,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 4, cv2.LINE_AA)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15 ,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)            
                #New export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints, allow_pickle=True, fix_imports=True)
                #display frame
                cv2.imshow("OpenCV Feed", image)
                
                #close the display
                if cv2.waitKey(10) & 0xFF ==ord('q'):
                    break
    cap.release()
    cv2.destroyAllWindows()