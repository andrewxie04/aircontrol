import cv2
import mediapipe as mp
import pyautogui

# Initialize Mediapipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize Mediapipe drawing utility
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Initialize previous zoom distance
prev_zoom_distance = 0

while True:
    # Read frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    results = hands.process(rgb_frame)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        # Iterate over each detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the index finger tip landmark
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Get the thumb tip landmark
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Calculate the distance between index finger tip and thumb tip
            distance = ((index_finger_tip.x - thumb_tip.x)**2 + (index_finger_tip.y - thumb_tip.y)**2)**0.5

            # Check if the distance is below a threshold (pinch gesture)
            if distance < 0.05:
                # Perform a mouse click
                pyautogui.click()

        # Check if two hands are detected
        if len(results.multi_hand_landmarks) == 2:
            # Get the index finger tip landmarks of both hands
            index_finger_tip1 = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_tip2 = results.multi_hand_landmarks[1].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Calculate the distance between the index finger tips
            zoom_distance = ((index_finger_tip1.x - index_finger_tip2.x)**2 + (index_finger_tip1.y - index_finger_tip2.y)**2)**0.5

            # Check if the zoom distance is increasing or decreasing
            if zoom_distance > prev_zoom_distance:
                # Perform zoom in
                pyautogui.keyDown('ctrl')
                pyautogui.scroll(50)
                pyautogui.keyUp('ctrl')
            elif zoom_distance < prev_zoom_distance:
                # Perform zoom out
                pyautogui.keyDown('ctrl')
                pyautogui.scroll(-50)
                pyautogui.keyUp('ctrl')

            # Update the previous zoom distance
            prev_zoom_distance = zoom_distance

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Check for 'q' key to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()