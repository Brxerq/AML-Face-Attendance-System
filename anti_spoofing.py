# Import all the libraries
import cv2
import dlib
import numpy as np
import os
import time
import mediapipe as mp
from skimage import feature

# I'm setting up the face and hand detectors here.
class AntiSpoofingSystem:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # Here I initialize MediaPipe for hand gesture detection.
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)


        # This code is for Webcam if you have Jetson kit change value from 0 to 1.
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # I create a directory to save the captured images if it doesn't exist.
        self.save_directory = "Person"
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)


        # Iam loading the Pre-trained model to detect smartphones.
        self.net_smartphone = cv2.dnn.readNet('yolov4.weights', 'PreTrained_yolov4.cfg')
        with open('PreTrained_coco.names', 'r') as f:
            self.classes_smartphone = f.read().strip().split('\n')


        # Setting some thresholds for eye aspect ratio to detect blinks.
        self.EAR_THRESHOLD = 0.2
        self.BLINK_CONSEC_FRAMES = 4

        # Initializing some variables to keep track of eye states and blink counts.
        self.left_eye_state = False
        self.right_eye_state = False
        self.left_blink_counter = 0
        self.right_blink_counter = 0

        # Variables to manage smartphone detection.
        self.smartphone_detected = False
        self.smartphone_detection_frame_interval = 20
        self.frame_count = 0

        # New attributes for student data
        self.student_id = None
        self.student_name = None


    # It is calculating the eye aspect ratio to detect blinks.
    def calculate_ear(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)


    # Analyzing the texture of the face to check for liveness.
    def analyze_texture(self, face_region):
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(gray_face, P=8, R=1, method="uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 58), range=(0, 58))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-5)
        return np.sum(lbp_hist[:10]) > 0.3

    # Detecting hand using MediaPipe.
    def detect_hand_gesture(self, frame):
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return results.multi_hand_landmarks is not None

    # Detecting smartphones in the frame to prevent System Bypass.
    def detect_smartphone(self, frame):
        if self.frame_count % self.smartphone_detection_frame_interval == 0:
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (224, 224), swapRB=True, crop=False)
            self.net_smartphone.setInput(blob)
            output_layers_names = self.net_smartphone.getUnconnectedOutLayersNames()
            detections = self.net_smartphone.forward(output_layers_names)

            for detection in detections:
                for obj in detection:
                    scores = obj[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.3 and self.classes_smartphone[class_id] == 'cell phone':
                        center_x = int(obj[0] * frame.shape[1])
                        center_y = int(obj[1] * frame.shape[0])
                        width = int(obj[2] * frame.shape[1])
                        height = int(obj[3] * frame.shape[0])
                        left = int(center_x - width / 2)
                        top = int(center_y - height / 2)

                        cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 0, 255), 2)
                        cv2.putText(frame, 'Smartphone Detected', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        self.smartphone_detected = True
                        self.left_blink_counter = 0
                        self.right_blink_counter = 0
                        return

        self.frame_count += 1
        self.smartphone_detected = False

    # Checking if the user blinked to confirm their presence.
    def detect_blink(self, left_ear, right_ear):
        if self.smartphone_detected:
            self.left_eye_state = False
            self.right_eye_state = False
            self.left_blink_counter = 0
            self.right_blink_counter = 0
            return False

        # Incrementing blink counter if a blink is detected.
        if left_ear < self.EAR_THRESHOLD:
            if not self.left_eye_state:
                self.left_eye_state = True
        else:
            if self.left_eye_state:
                self.left_eye_state = False
                self.left_blink_counter += 1

        if right_ear < self.EAR_THRESHOLD:
            if not self.right_eye_state:
                self.right_eye_state = True
        else:
            if self.right_eye_state:
                self.right_eye_state = False
                self.right_blink_counter += 1


        # Resetting blink counters after a successful blink detection.
        if self.left_blink_counter > 0 and self.right_blink_counter > 0:
            self.left_blink_counter = 0
            self.right_blink_counter = 0
            return True
        else:
            return False

    # Main loop to process the video feed.
    def run(self, update_frame_callback=None):
        blink_count = 0
        hand_gesture_detected = False
        image_captured = False
        last_event_time = time.time()
        event_timeout = 60
        message_displayed = False

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Detecting smartphones in the frame.   
            self.detect_smartphone(frame)

            # Displaying a warning if a smartphone is detected.
            if self.smartphone_detected:
                cv2.putText(frame, "Mobile phone detected, can't record attendance", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                blink_count = 0

            # Processing each frame to detect faces, blinks, and hand gestures.
            if not self.smartphone_detected:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)

                for face in faces:
                    landmarks = self.predictor(gray, face)
                    leftEye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
                    rightEye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])

                    ear_left = self.calculate_ear(leftEye)
                    ear_right = self.calculate_ear(rightEye)

                    if self.detect_blink(ear_left, ear_right):
                        blink_count += 1

                    # Prionting and Incrementing blink Count
                    cv2.putText(frame, f"Blink Count: {blink_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    hand_gesture_detected = self.detect_hand_gesture(frame)

                    # Indicating when a hand gesture is detected.
                    if hand_gesture_detected:
                        cv2.putText(frame, "Hand Gesture Detected", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
                    expanded_region = frame[max(y - h // 2, 0):min(y + 3 * h // 2, frame.shape[0]),
                                            max(x - w // 2, 0):min(x + 3 * w // 2, frame.shape[1])]

                    # Checking if the conditions are met to capture the image.
                    if blink_count >= 5 and hand_gesture_detected and self.analyze_texture(expanded_region) and not message_displayed:
                        cv2.putText(frame, "Please hold still for 2 seconds...", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imshow("Frame", frame)
                        cv2.waitKey(1)
                        time.sleep(2)
                        message_displayed = True

                    if message_displayed and not image_captured:
                        timestamp = int(time.time())
                        picture_name = f"{self.student_id}_{timestamp}.jpg"
                        cv2.imwrite(os.path.join(self.save_directory, picture_name), expanded_region)
                        image_captured = True

            if update_frame_callback:
                update_frame_callback(frame)

            cv2.imshow("Frame", frame)
            if image_captured or (time.time() - last_event_time > event_timeout and not hand_gesture_detected):
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

        #If person if real and did all the required features then his attendance will be marked if not then it will print no person detected.
        if image_captured:
            print(f"Person detected. Face image captured and saved as {picture_name}.")
        elif not hand_gesture_detected:
            print("No real person detected")

if __name__ == "__main__":
    anti_spoofing_system = AntiSpoofingSystem()
    anti_spoofing_system.run()
