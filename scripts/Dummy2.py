import cv2
import mediapipe as mp
import csv

# Set up mediapipe instance
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Open a video capture object
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Set up mediapipe drawing styles
drawing_styles = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Read stored landmarks from CSV
with open('landmarks.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    stored_landmarks = [list(map(float, row)) for row in csv_reader]

# Initialize a list to store the Euclidean distance errors
euclidean_distances = []

print(stored_landmarks)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # If the frame is not read correctly or the video capture is not successful, break
        if not ret:
            break

        # Convert the BGR image to RGB before processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Draw the pose landmarks on the frame
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            # Compare the detected landmarks with stored landmarks
            detected_landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in results.pose_landmarks.landmark]
    
            for i, (stored, detected) in enumerate(zip(stored_landmarks, detected_landmarks)):
                # Perform your comparison here
                # Example comparison (Euclidean distance)
                euclidean_distance = ((stored[0] - detected[0]) ** 2 + (stored[1] - detected[1]) ** 2 + (stored[2] - detected[2]) ** 2) ** 0.5
                euclidean_distances.append(euclidean_distance)
                
                
                
                if euclidean_distance > 0.25:
                # Draw a box around the point
                    x, y = int(detected[0] * image.shape[1]), int(detected[1] * image.shape[0])
                    box_size = 10  # Define the size of the box
                    cv2.rectangle(image, (x - box_size, y - box_size), (x + box_size, y + box_size), (0, 255, 0), 2)

                # Draw the point on the image
                cv2.circle(image, (int(detected[0] * image.shape[1]), int(detected[1] * image.shape[0])), 5, (0, 0, 255), -1)
                cv2.putText(image, f'{i}: {euclidean_distance:.2f}', (int(detected[0] * image.shape[1]) + 10, int(detected[1] * image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                drawing_styles,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            )

        # Show the frame
        cv2.imshow("Pose Detection", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()

# Print Euclidean distances at the end
print("Euclidean distances:")
for i, distance in enumerate(euclidean_distances):
    print(f'Point {i}: {distance}')
import cv2
import mediapipe as mp
import csv

# Set up mediapipe instance
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Open a video capture object
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Set up mediapipe drawing styles
drawing_styles = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Read stored landmarks from CSV
with open('landmarks.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    stored_landmarks = [list(map(float, row)) for row in csv_reader]

# Initialize a list to store the Euclidean distance errors
euclidean_distances = []

print(stored_landmarks)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # If the frame is not read correctly or the video capture is not successful, break
        if not ret:
            break

        # Convert the BGR image to RGB before processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Draw the pose landmarks on the frame
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            # Compare the detected landmarks with stored landmarks
            detected_landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in results.pose_landmarks.landmark]
    
            for i, (stored, detected) in enumerate(zip(stored_landmarks, detected_landmarks)):
                # Perform your comparison here
                # Example comparison (Euclidean distance)
                euclidean_distance = ((stored[0] - detected[0]) ** 2 + (stored[1] - detected[1]) ** 2 + (stored[2] - detected[2]) ** 2) ** 0.5
                euclidean_distances.append(euclidean_distance)
                
                
                
                if euclidean_distance > 0.4:
                # Draw a box around the point
                    x, y = int(detected[0] * image.shape[1]), int(detected[1] * image.shape[0])
                    box_size = 10  # Define the size of the box
                    cv2.rectangle(image, (x - box_size, y - box_size), (x + box_size, y + box_size), (0, 255, 0), 2)

                # Draw the point on the image
                cv2.circle(image, (int(detected[0] * image.shape[1]), int(detected[1] * image.shape[0])), 5, (0, 0, 255), -1)
                cv2.putText(image, f'{i}: {euclidean_distance:.2f}', (int(detected[0] * image.shape[1]) + 10, int(detected[1] * image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                drawing_styles,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            )

        # Show the frame
        cv2.imshow("Pose Detection", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()

# Print Euclidean distances at the end
print("Euclidean distances:")
for i, distance in enumerate(euclidean_distances):
    print(f'Point {i}: {distance}')
