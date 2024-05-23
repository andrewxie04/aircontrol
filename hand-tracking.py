import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize Mediapipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize Mediapipe drawing utility
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Set the desired width and height of the video capture
desired_width = 800
desired_height = 800
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Initialize cursor bubble parameters
bubble_radius = 20
bubble_color = (0, 255, 0)  # Green color
cursor_position = (0, 0)
pinch_threshold = 0.05
click_threshold = 0.8  # Adjust this value for the closed fist threshold
scroll_threshold = 0.2  # Adjust this value for the fist scroll threshold
scroll_scaling_factor = 0.1  # Adjust this value to control the scrolling sensitivity

# Initialize click flag, drag flag, and scroll flag
click_flag = False
drag_flag = False
scroll_flag = False
prev_fist_y = 0

# Initialize smoothing parameters
smoothing_factor = 0.5
smoothed_cursor_position = (0, 0)

while True:
    # Read frame from the video capture
    ret, frame = cap.read()

    # Unflip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    results = hands.process(rgb_frame)

    # Check if a hand is detected
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw hand landmarks on the frame
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get the index finger tip landmark
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # Get the thumb tip landmark
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

        # Calculate the distance between index finger tip and thumb tip
        pinch_distance = ((index_finger_tip.x - thumb_tip.x)**2 + (index_finger_tip.y - thumb_tip.y)**2)**0.5

        # Check if the hand is in a closed fist position
        fist_landmarks = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                          hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                          hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                          hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]]
        fist_distances = [((landmark.x - hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x)**2 +
                           (landmark.y - hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y)**2)**0.5
                          for landmark in fist_landmarks]
        fist_avg_distance = sum(fist_distances) / len(fist_distances)

        # Calculate the cursor position based on the index finger tip position
        cursor_x = int(index_finger_tip.x * frame.shape[1])
        cursor_y = int(index_finger_tip.y * frame.shape[0])

        # Apply smoothing to the cursor position
        smoothed_cursor_position = (
            smoothing_factor * cursor_x + (1 - smoothing_factor) * smoothed_cursor_position[0],
            smoothing_factor * cursor_y + (1 - smoothing_factor) * smoothed_cursor_position[1]
        )

        # Update the cursor position with the smoothed values
        cursor_position = (int(smoothed_cursor_position[0]), int(smoothed_cursor_position[1]))

        # Move the cursor on the screen
        pyautogui.moveTo(cursor_position[0], cursor_position[1])

        # Check if the fingers are pinched together for dragging
        if pinch_distance < pinch_threshold:
            if not drag_flag:
                pyautogui.mouseDown()
                drag_flag = True
        else:
            if drag_flag:
                pyautogui.mouseUp()
                drag_flag = False

        # Check if the hand is in a closed fist position for clicking
        if fist_avg_distance < click_threshold:
            if not click_flag:
                pyautogui.click()
                click_flag = True
        else:
            click_flag = False

        # Check if the hand is in a fist position for scrolling
        if fist_avg_distance < scroll_threshold:
            if not scroll_flag:
                scroll_flag = True
                prev_fist_y = cursor_y
            else:
                scroll_amount = int((cursor_y - prev_fist_y) * scroll_scaling_factor)
                pyautogui.scroll(scroll_amount)
                prev_fist_y = cursor_y
        else:
            scroll_flag = False

    # Create a blank image for the cursor bubble
    bubble_image = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

    # Draw the cursor bubble on the bubble image
    cv2.circle(bubble_image, cursor_position, bubble_radius, bubble_color, -1)

    # Combine the frame and the bubble image
    output_frame = cv2.addWeighted(frame, 1, bubble_image, 0.7, 0)

    # Display the output frame
    cv2.imshow('Hand Gesture Recognition', output_frame)

    # Check for 'q' key to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()