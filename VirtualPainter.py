import cv2
import numpy as np
import os
import HandTrackingModule as htm
import time

def main():
    #######################
    brushThickness = 15
    eraserThickness = 50
    #######################

    folderPath = "Header"
    # Load Header Image
    header_path = os.path.join(os.getcwd(), 'UI_Header.png')
    if os.path.exists(header_path):
        header = cv2.imread(header_path)
        header = cv2.resize(header, (1280, 125)) # Force resize to match UI area
    else:
        print("Header image not found! Using fallback.")
        header = None

    cap = cv2.VideoCapture(0)
    width = 1280
    height = 720
    cap.set(3, width)
    cap.set(4, height)

    detector = htm.HandDetector(detectionCon=0.85, maxHands=1) # Optimization: Max 1 hand

    # Default color (Pink)
    drawColor = (147, 20, 255) 
    
    imgCanvas = np.zeros((height, width, 3), np.uint8)

    xp, yp = 0, 0

    while True:
        # 1. Import Image
        success, img = cap.read()
        if not success:
            break
        img = cv2.resize(img, (width, height)) # FORCE RESIZE to ensure match with canvas
        img = cv2.flip(img, 1) # Mirror image

        # 2. Performance Check: Resize for inference
        imgSmall = cv2.resize(img, (640, 360))

        # 3. Find Hand Landmarks (Passing Timestamp for VIDEO mode on SMALL image)
        timestamp_ms = int(time.time() * 1000)
        detector.detectHands(imgSmall, timestamp_ms)
        
        # 4. Draw works on ANY image size because landmarks are normalized
        detector.drawHands(img) 
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            # Tip of Index and Middle fingers
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

            # Check which fingers are up
            fingers = detector.fingersUp()
            
            # --- SELECTION MODE ---
            # If Selection Mode - Two fingers are up
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0 # Reset previous points
                
                # Brush Preview (Ring)
                cv2.circle(img, (x1, y1), 20, drawColor, 3) 

                # Check for Click in Header
                if y1 < 125:
                    if 0 < x1 < 320: # Pink Region
                        drawColor = (147, 20, 255) 
                    elif 320 < x1 < 640: # Green Region
                        drawColor = (0, 255, 0)
                    elif 640 < x1 < 960: # Blue Region
                        drawColor = (255, 0, 0) # BGR
                    elif 960 < x1 < 1280: # Eraser Region
                        drawColor = (0, 0, 0)
                
            # --- DRAWING MODE ---
            # If Drawing Mode - Index finger is up
            if fingers[1] and not fingers[2]:
                # Brush Preview: Filled circle with active color
                if drawColor == (0, 0, 0):
                     cv2.circle(img, (x1, y1), 20, (255, 255, 255), 2) # Empty ring for eraser
                else:
                    cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                if drawColor == (0, 0, 0):
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                else:
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                
                xp, yp = x1, y1

        # 4. Draw Header Image
        if header is not None:
            # Overlay header on top of image
            img[0:125, 0:1280] = header

        # 5. Active Selection Highlight
        # Draw a glowing box around the selected color
        if drawColor == (147, 20, 255): # Pink
             cv2.rectangle(img, (0, 0), (320, 125), (255, 255, 255), 3)
        elif drawColor == (0, 255, 0): # Green
             cv2.rectangle(img, (320, 0), (640, 125), (255, 255, 255), 3)
        elif drawColor == (255, 0, 0): # Blue
             cv2.rectangle(img, (640, 0), (960, 125), (255, 255, 255), 3)
        elif drawColor == (0, 0, 0): # Eraser
             cv2.rectangle(img, (960, 0), (1280, 125), (255, 255, 255), 3)

        # 6. Canvas Merging
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img,imgInv)
        img = cv2.bitwise_or(img,imgCanvas)

        cv2.imshow("Virtual Painter", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
