# # TechVidvan Human pose estimator
# # import necessary packages

# import cv2
# import mediapipe as mp
# import csv 
# import os
# # initialize Pose estimator

# def imagePoints():

#     mp_drawing = mp.solutions.drawing_utils
#     mp_pose = mp.solutions.pose

#     pose = mp_pose.Pose(
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5)

#     # create capture object
#     cap = cv2.imread('S2.jpg')
#     temp_file = open('temp.csv', 'w', newline='')
#     writer = csv.writer(temp_file)
#     # writer.writerow(['x', 'y', 'z', 'visibility'])

#     # while cap.isOpened():
#     #     # read cap from capture object
#     # _, cap = cap.read()

#     try:
#         # convert the cap to RGB format
#         RGB = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)

#         # process the RGB cap to get the result
#         results = pose.process(RGB)

#         print(results.pose_landmarks)
#         # draw detected skeleton on the cap
#         mp_drawing.draw_landmarks(
#             cap, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#         # show the final output
#         cv2.imshow('Output', cap)

#         for landmark in results.pose_landmarks.landmark:
#             writer.writerow([landmark.x, landmark.y, landmark.z, landmark.visibility])

#         # close the temporary CSV file
        
#     except :
#         print("Error")
#     # if cv2.waitKey(1) == ord('q'):
#     #     break
        
#     temp_file.close()
#     os.replace('temp.csv', 'landmarks.csv')
#     # cap.release()
#     # cap.release()
#     cv2.destroyAllWindows()


# imagePoints()