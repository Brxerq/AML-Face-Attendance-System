import cv2
import dlib
import numpy as np
import os
import time
import mediapipe as mp
from skimage import feature

class AntiSpoofingSystem:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.anti_spoofing_completed = False
        self.blink_count = 0
        self.image_captured = False
        self.captured_image = None
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.save_directory = "Person"
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        self.net_smartphone = cv2.dnn.readNet('yolov4.weights', 'Pretrained_yolov4 (1).cfg')
        with open('PreTrained_coco.names', 'r') as f:
            self.classes_smartphone = f.read().strip().split('\n')

        self.EAR_THRESHOLD = 0.25
        self.BLINK_CONSEC_FRAMES = 4

        self.left_eye_state = False
        self.right_eye_state = False
        self.left_blink_counter = 0
        self.right_blink_counter = 0

        self.smartphone_detected = False
        self.smartphone_detection_frame_interval = 30
        self.frame_count = 0

        # New attributes for student data
        self.student_id = None
        self.student_name = None

    def calculate_ear(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    def analyze_texture(self, face_region):
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(gray_face, P=8, R=1, method="uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 58), range=(0, 58))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-5)
        return np.sum(lbp_hist[:10]) > 0.3

    def detect_hand_gesture(self, frame):
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return results.multi_hand_landmarks is not None

    def detect_smartphone(self, frame):
        if self.frame_count % self.smartphone_detection_frame_interval == 0:
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            self.net_smartphone.setInput(blob)
            output_layers_names = self.net_smartphone.getUnconnectedOutLayersNames()
            detections = self.net_smartphone.forward(output_layers_names)

            for detection in detections:
                for obj in detection:
                    scores = obj[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5 and self.classes_smartphone[class_id] == 'cell phone':
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

    def access_verified_image(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Perform anti-spoofing checks
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        # Check if a face is detected
        if len(faces) == 0:
            return None
        
        # Assume the first detected face is the subject
        face = faces[0]
        landmarks = self.predictor(gray, face)
        
        # Check for blink detection (assuming you have this method correctly implemented)
        leftEye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
        rightEye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
        ear_left = self.calculate_ear(leftEye)
        ear_right = self.calculate_ear(rightEye)
        if not self.detect_blink(ear_left, ear_right):
            return None

        # Check for hand gesture (assuming you have this method correctly implemented)
        if not self.detect_hand_gesture(frame):
            return None

        # Check if a smartphone is detected
        self.detect_smartphone(frame)
        if self.smartphone_detected:
            return None
        
        # Check texture for liveness (assuming you have this method correctly implemented)
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        expanded_region = frame[max(y - h // 2, 0):min(y + 3 * h // 2, frame.shape[0]),
                                max(x - w // 2, 0):min(x + 3 * w // 2, frame.shape[1])]
        if not self.analyze_texture(expanded_region):
            return None

        return frame

    def detect_blink(self, left_ear, right_ear):
        if self.smartphone_detected:
            self.left_eye_state = False
            self.right_eye_state = False
            self.left_blink_counter = 0
            self.right_blink_counter = 0
            return False

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

        if self.left_blink_counter > 0 and self.right_blink_counter > 0:
            self.left_blink_counter = 0
            self.right_blink_counter = 0
            return True
        else:
            return False

    def run(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Detect smartphone in the frame
        self.detect_smartphone(frame)

        if self.smartphone_detected:
            cv2.putText(frame, "Mobile phone detected, can't record attendance", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            for face in faces:
                landmarks = self.predictor(gray, face)
                leftEye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
                rightEye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])

                ear_left = self.calculate_ear(leftEye)
                ear_right = self.calculate_ear(rightEye)

                if self.detect_blink(ear_left, ear_right):
                    self.blink_count += 1
                    cv2.putText(frame, f"Blink Count: {self.blink_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Check if conditions for image capture are met
                if self.blink_count >= 5 and not self.image_captured:
                    # Capture the image and reset blink count
                    self.save_image(frame)
                    self.blink_count = 0
                    self.image_captured = True

        return frame

    def save_image(self, frame):
        # Implement logic to save the frame as an image
        timestamp = int(time.time())
        image_name = f"captured_{timestamp}.png"
        cv2.imwrite(os.path.join(self.save_directory, image_name), frame)
        self.captured_image = frame
        print(f"Image captured and saved as {image_name}")

    def get_captured_image(self):
        # Return the captured image with preprocessing applied (if necessary)
        captured_frame = self.captured_image
        if captured_frame is not None:
            # Apply any additional preprocessing needed for consistency with GestureRecognition
            # For example, resizing, color conversion, etc.
            # captured_frame = your_preprocessing_method(captured_frame)
            return captured_frame
        return None

if __name__ == "__main__":
    anti_spoofing_system = AntiSpoofingSystem()
    anti_spoofing_system.run()
