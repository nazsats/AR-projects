import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # Settings
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        # Model Path
        model_path = os.path.join(os.getcwd(), 'hand_landmarker.task')
        
        # Create HandLandmarker options
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.maxHands,
            min_hand_detection_confidence=self.detectionCon,
            min_hand_presence_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon)
        
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None
        
        # Manual Hand Connections
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), 
            (0, 5), (5, 6), (6, 7), (7, 8), 
            (0, 9), (9, 10), (10, 11), (11, 12), 
            (0, 13), (13, 14), (14, 15), (15, 16), 
            (0, 17), (17, 18), (18, 19), (19, 20), 
            (5, 9), (9, 13), (13, 17) 
        ]
        self.tipIds = [4, 8, 12, 16, 20]

    def detectHands(self, img, timestamp_ms):
        # Convert the image to MediaPipe Image format
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        
        # Detect (VIDEO MODE)
        self.results = self.detector.detect_for_video(mp_image, timestamp_ms)
        return self.results

    def drawHands(self, img):
        if self.results and self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                # Draw Connections
                h, w, c = img.shape
                for start_idx, end_idx in self.hand_connections:
                    start_point = hand_landmarks[start_idx]
                    end_point = hand_landmarks[end_idx]
                    
                    x1, y1 = int(start_point.x * w), int(start_point.y * h)
                    x2, y2 = int(end_point.x * w), int(end_point.y * h)
                    
                    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)

                # Draw Points
                for landmark in hand_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results and self.results.hand_landmarks:
            if handNo < len(self.results.hand_landmarks):
                myHand = self.results.hand_landmarks[handNo]
                h, w, c = img.shape
                for id, lm in enumerate(myHand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([id, cx, cy])
                    if draw and id == 8: # Index finger tip
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return self.lmList
        self.lmList = []
        if self.results and self.results.hand_landmarks:
            if handNo < len(self.results.hand_landmarks):
                myHand = self.results.hand_landmarks[handNo]
                h, w, c = img.shape
                for id, lm in enumerate(myHand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([id, cx, cy])
                    if draw and id == 8: # Index finger tip
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers = []
        if len(self.lmList) == 0:
            return fingers

        # Thumb (Simple Logic for now: Right Hand assumed or relative to x)
        # Note: Tasks API landmarks are normalized. 
        # Logic: Check if tip x is variable relative to knuckle?
        # Handedness is better but for simple implementation:
        # Check if Thumb Tip is to the Left/Right of IP joint depending on hand
        # For simplicity, keeping the logic: tip x < tip-1 x (Right Hand palm facing)
        
        # Note: This logic assumes right hand. 
        # Ideally check handedness from results.handedness
        
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        if not success:
            break
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
