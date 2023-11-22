from flask import Flask, request,render_template,redirect,flash,url_for,Response, jsonify
from flask_cors import CORS, cross_origin
import time
import base64
from PIL import Image
from io import BytesIO

import cv2
import mediapipe as mp
import csv 
import os
from flask_mysqldb import MySQL
import shutil
from werkzeug.utils import secure_filename
from flask import send_from_directory
import math

app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''

app.config['MYSQL_DB'] ='pose_estimation'
app.config['SECRET_KEY']='mykey'
mysql = MySQL(app=app)
CORS(app, support_credentials=True)

cwd = os.getcwd()
path1 = cwd+'/static/refImages'

def imagePoints(filename, username):

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    print("filename2 : "+filename)

    # create capture object
    cap = cv2.imread(path1+"/"+username+"/"+filename);
    temp_file = open("temp.csv", 'w', newline='')
    writer = csv.writer(temp_file)
    # writer.writerow(['x', 'y', 'z', 'visibility'])

    # while cap.isOpened():
    #     # read cap from capture object
    # _, cap = cap.read()

    try:
        # convert the cap to RGB format
        RGB = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)

        # process the RGB cap to get the result
        results = pose.process(RGB)

        print(results.pose_landmarks)
        # draw detected skeleton on the cap
        mp_drawing.draw_landmarks(
            cap, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # show the final output
        cv2.imshow('Output', cap)

        for landmark in results.pose_landmarks.landmark:
            writer.writerow([landmark.x, landmark.y, landmark.z, landmark.visibility])

        # close the temporary CSV file
        
    except Exception as e:
        print(e)
        print("Error")
    # if cv2.waitKey(1) == ord('q'):
    #     break
    
    temp_file.close()
    os.replace('temp.csv', 'landmarks.csv')
    # cap.release()
    # cap.release()
    cv2.destroyAllWindows()

def main():

    marker = {
    'face': [x for x in range(1, 11)],
    'Right shoulder': [12],
    'Left shoulder': [11],
    'Right Feet': [28, 30, 32],
    'Left Feet': [27, 29, 31],
    'Right knee': [26],
    'Left Knee': [25],
    'Left Arm': [13],
    'Right Arm': [14],
    'Left Wrist': [15, 17, 19, 21],
    'Right Wrist': [16, 18, 20, 22]
    }

# Function to calculate Euclidean distance between two 3D points
    def euclidean_distance_3d(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)

    # Function to calculate distance between two landmarks representing height
    def calculate_height(landmark1, landmark2):
        return euclidean_distance_3d(landmark1, landmark2)

    # Function to calculate distance between two landmarks representing width
    def calculate_width(landmark1, landmark2):
        return euclidean_distance_3d(landmark1, landmark2)

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

                # Calculate height and width
                shoulder_width = calculate_width(detected_landmarks[11], detected_landmarks[12])
                person_height = calculate_height(detected_landmarks[11], detected_landmarks[24])

                for i, (stored, detected) in enumerate(zip(stored_landmarks, detected_landmarks)):
                    # Perform your comparison here
                    # Example comparison (Euclidean distance)
                    euclidean_distance = ((stored[0] - detected[0]) ** 2 + (stored[1] - detected[1]) ** 2 + (
                                stored[2] - detected[2]) ** 2) ** 0.5
                    euclidean_distances.append(euclidean_distance)

                    # Compare height and suggest adjustments
                    if i == 11:  # Left shoulder landmark index
                        height_difference = abs(stored[1] - detected[1])
                        if height_difference > 0.1:  # Adjust the threshold as needed
                            suggestion += f" Adjust your height: Move your shoulders up or down."

                    # Compare width and suggest adjustments
                    if i == 11 or i == 12:  # Left and right shoulder landmark indices
                        width_difference = abs(stored[0] - detected[0])
                        if width_difference > 0.1:  # Adjust the threshold as needed
                            suggestion += f" Adjust your width: Widen or narrow your shoulders."

                    # Suggest corrections based on significant differences
                    if euclidean_distance > 0.2:
                        body_part = None
                        for part, indices in marker.items():
                            if i in indices:
                                body_part = part
                                break

                        if body_part:
                            # Suggestion for adjusting the pose
                            suggestion = f"Suggestion for {body_part}:"

                            # Analyze the difference in coordinates
                            for coord_type, (stored_coord, detected_coord) in zip(['x', 'y', 'z'], zip(stored, detected)):
                                diff = stored_coord - detected_coord
                                if diff > 0:
                                    suggestion += f" Move your {body_part} to the right."
                                elif diff < 0:
                                    suggestion += f" Move your {body_part} to the left."
                                else:
                                    suggestion += f" Keep your {body_part} position."

                            print(suggestion)

                        # Draw a box around the point if error is more than 0.2
                        if euclidean_distance > 0.2:
                            x, y = int(detected[0] * image.shape[1]), int(detected[1] * image.shape[0])
                            box_size = 10  # Define the size of the box
                            cv2.rectangle(image, (x - box_size, y - box_size), (x + box_size, y + box_size), (255, 0, 0), 2)

                        # Draw the point on the image
                        cv2.circle(image, (int(detected[0] * image.shape[1]), int(detected[1] * image.shape[0])), 5,
                                (0, 0, 255), -1)
                        cv2.putText(image, f'{i}: {euclidean_distance:.2f}',
                                    (int(detected[0] * image.shape[1]) + 10, int(detected[1] * image.shape[0])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    drawing_styles,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                )
                # Convert the frame to JPEG format
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()

                # Yield the frame for the streaming response
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break


            # Show the frame
            # cv2.imshow("Pose Detection", image)


            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break

    # Release the VideoCapture object and close the window
    cap.release()
    cv2.destroyAllWindows()

    # Print Euclidean distances at the end
    print("Euclidean distances:")
    for i, distance in enumerate(euclidean_distances):
        print(f'Point {i}: {distance}')


# Define a route for handling POST requests
@app.route('/post_example', methods=['POST','GET'])
@cross_origin(supports_credentials=True)
def post_example():
    if request.method == 'GET':
        # Access data sent with the POST request
        
        # Assuming 'data' contains a URL, you can extract it
        # url = data  # You may want to perform additional validation or parsing

        imagePoints()

        # time.sleep(10)

        main()

        return ""
    else:
        return 'This route only accepts POST requests.'


@app.route('/')
def home():
    cur = mysql.connection.cursor()
    cur.execute("create database if not exists `pose_estimation`")
    mysql.connection.commit()

    sql = "CREATE TABLE IF NOT EXISTS `logs`  (`Time` TIMESTAMP NOT NULL DEFAULT current_timestamp() , `Username` VARCHAR(30) NOT NULL );" 
    cur.execute(sql)
    mysql.connection.commit()

    sql = "CREATE TABLE IF NOT EXISTS `users`(`Name` VARCHAR(50) NOT NULL , `Address` VARCHAR(100) NULL DEFAULT NULL , `Email` VARCHAR(100)  DEFAULT NULL , `Contact` VARCHAR(20) DEFAULT NULL , `Username` VARCHAR(30) NOT NULL , `Password` VARCHAR(30) NOT NULL , PRIMARY KEY (`Username`(30)))"
    cur.execute(sql)
    mysql.connection.commit()

    cur.execute('''
    CREATE TABLE IF NOT EXISTS images (
        id INT AUTO_INCREMENT PRIMARY KEY,
        filename VARCHAR(255) NOT NULL,
        data LONGBLOB
    )
    ''')

    mysql.connection.commit()

    cur.execute('''
    CREATE TABLE IF NOT EXISTS userImage (
        id INT AUTO_INCREMENT UNIQUE,
        filename VARCHAR(255) NOT NULL,
        Username VARCHAR(30) NOT NULL ,PRIMARY KEY(Username,filename)
    );
    ''')

    mysql.connection.commit()

    cur.close()

    # Take us to home

    return render_template('index.html')

@app.route('/signin')
def signin():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('register.html')



@app.route('/registerRes', methods=['POST', 'GET'])
def registration():
    name          = request.form['name']
    address       = request.form['address']
    email         = request.form['email']
    contact       = request.form['contact']
    username      = request.form['username']
    password      = request.form['pswd']

    cur = mysql.connection.cursor()
    


    sql = "SELECT username, email from users"
    cur.execute(sql)
    x = cur.fetchall()
    sameemail=False
    flag=False
    for i in x:
        if(str(i[0]).lower()==username.lower() or str(i[1]).lower()==email.lower()):
            if(str(i[0]).lower()==username.lower()):

                flag=True
                break
            else:
                sameemail=True
                break
        else:
            continue
    
    if(sameemail):
        flash('Email already registered!','danger')
        return redirect(url_for('registration'))
    elif(flag):
        flash('Username already taken!','danger')
        return redirect(url_for('registration'))
    else:
        try:
            sql = "INSERT INTO users(Name,Address,Email,Contact,Username,Password) VALUES (%s,%s,%s,%s,%s,%s)"

            cur.execute(sql,(name,address,email,contact,username,password))
            mysql.connection.commit()
            
            cur.close()
        
        except Exception as e:
            return str(e)
            flash("There was an error registering your email!!",'danger')

    flash("Registration Success!!",category="information")

    return redirect(url_for("signin"))


@app.route('/loginRes', methods=['POST'])
def login():
    username       = request.form['user']
    password       = request.form['pswd']

    cur = mysql.connection.cursor()
     
    sql = "SELECT password from Users where username = %s"
    
    cur.execute(sql, (username,))

    record = cur.fetchall()

    if(len(record)== 0 ):

        flash("User doesn't exist!  Please enter Correct Username")
        return redirect(url_for('signin'))
    
    
    if(password==record[0][0]):
        cur.execute("INSERT INTO `logs`(username) VALUES (%s)",(username,))
        mysql.connection.commit()
        cur.close()
        

        

    
    elif(password != record[0][0]):

        flash("Login Failed! Access Denied.")

        return redirect(url_for('signin'))
    



    return  redirect(url_for('userdash',username=username))

cwd = os.getcwd()
path = cwd+'/pose_correction/scripts'
path1 = cwd+'/static/refImages'

STATIC_FOLDER = 'static'
IMAGE_FOLDER = 'refImages'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['IMAGE_FOLDER'] = os.path.join(STATIC_FOLDER, IMAGE_FOLDER)
# Ensure the refImages folder exists inside the static folder
os.makedirs(app.config['IMAGE_FOLDER'], exist_ok=True)


@app.route('/upload/<string:username>', methods=['GET','POST'])
def upload(username):
    try:

        cur = mysql.connection.cursor()

        if 'image' not in request.files:
            flash("Please provide an image to upload",category="Information")
            return redirect(url_for('home'))
        file = request.files['image']

        if file.filename == '':
            user = username
            
            flash("Please provide an image to be uploaded")            
            return redirect(url_for('uploads',username=user))
        
        
        

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename);
            

            # Save the file to the static/refImages folder
            path = os.path.join(app.config['IMAGE_FOLDER'],username)
            os.makedirs(path, exist_ok=True)
            file.save(os.path.join(path, filename))

            # Optionally, you can store information about the file in the database here
            cur = mysql.connection.cursor()
            try:
                cur.execute("INSERT INTO userImage (Username, fileName) VALUES (%s, %s)", (username, filename))
            except:
                flash("This file has already been uploaded! : Change image name or use another image :)")
            mysql.connection.commit()
            user = username

        return redirect(url_for('uploads',username=user))
    

    except Exception as e:
        return jsonify({'error': str(e)}), 500
# upload image     

@app.route('/uploads/<filename>/<username>')
def uploaded_image(filename,username):
    userpath = os.path.join(path1,username)
    return send_from_directory(userpath, filename)

# Capture image:
@app.route('/capture')
def capture():
    return render_template('captureImage.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

STATIC_FOLDER = 'static'
IMAGE_FOLDER = 'refImages'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['IMAGE_FOLDER'] = os.path.join(STATIC_FOLDER, IMAGE_FOLDER)
# Ensure the refImages folder exists inside the static folder
os.makedirs(app.config['IMAGE_FOLDER'], exist_ok=True)

# route to store image in database
@app.route('/store_data', methods=['POST'])
def store_data():
    try:
        file = request.files.get('file')
        username = request.form.get('username')

        # Check if the file is present
        if file is None or file.filename == '':
            return jsonify({'error': 'No file uploaded'}), 400

        # Check if the file has an allowed extension
        if file and allowed_file(file.filename):
            # Generate a secure filename
            filename = secure_filename(username);

            # Save the file to the static/refImages folder
            file.save(os.path.join(app.config['IMAGE_FOLDER'], filename))

            # Optionally, you can store information about the file in the database here
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO userImage (Username, fileName) VALUES (%s, %s)", (username, filename))
            mysql.connection.commit()

        return redirect(url_for('uploads'),[username])

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# get all referance image 

@app.route('/all_images')
def all_images():
    try:
        cur = mysql.connection.cursor()
        cur.execute('SELECT filename FROM images')
        result = cur.fetchall()

        # Extract filenames from the result
        image_filenames = [row[0] for row in result]

        return render_template('all_images.html', image_filenames=image_filenames)
    except Exception as e:
        # Handle exceptions
        print(e)
        return "An error occurred while retrieving images."



@app.route('/video_feed')
def video_feed():
    return Response(main(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_feed/<filename>/<username>',methods=["GET","POST"])
def camerafeed(filename, username):
    print("fileName1 :"+ filename)
    imagePoints(filename, username)
    return render_template("camera_feed.html", filename=filename, username=username)
@app.route('/userdash/<string:username>')
def userdash(username):
    return render_template('userdash.html',username=username)

@app.route('/uploaded/<string:username>')
def uploads(username):
    
    cur = mysql.connection.cursor()
    cur.execute("SELECT filename FROM userimage WHERE username = %s;",(username,))
    record = cur.fetchall()
    image_filenames = [row[0] for row in record]
    cur.close()
    return render_template('uploads.html', image_filenames=image_filenames, username=username)
    
    



if __name__ == '__main__':
    app.run(debug=True)
