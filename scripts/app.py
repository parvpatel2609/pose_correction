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


app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'sanku@2003'
app.config['MYSQL_DB'] ='pose_estimation'
app.config['SECRET_KEY']='mykey'
mysql = MySQL(app=app)
# hello
CORS(app, support_credentials=True)

def imagePoints():

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)


    # create capture object
    cap = cv2.imread('S2 copy.jpg')
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
                # Convert the frame to JPEG format
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()

                # Yield the frame for the streaming response
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break


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

    sql = "CREATE TABLE IF NOT EXISTS `users`(`Name` VARCHAR(50) NOT NULL , `Address` VARCHAR(100) NULL DEFAULT NULL , `Email` VARCHAR(100) NULL DEFAULT NULL , `Contact` INT(20) NULL DEFAULT NULL , `Username` VARCHAR(30) NOT NULL , `Password` VARCHAR(30) NOT NULL , PRIMARY KEY (`Username`(30)))"
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
        id INT AUTO_INCREMENT PRIMARY KEY,
        filename VARCHAR(255) NOT NULL,
        Username VARCHAR(30) NOT NULL
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
        
        except:
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
    



    return redirect(url_for("home"))

cwd = os.getcwd()
path = cwd+'/pose_correction/scripts'
path1 = cwd+'/static/uploads'

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # request.form['upload'].lower()
        # if 'image' in request.files:
        #     image = request.files['image']
            
        #     if image.filename != '':
        #         # Save the image with a specific filename (S2.jpg in this case)
        #         # image.save(os.path.join(path, 'S2.jpg'))
                
        #         image.save(os.path.join(path1, 'S2.jpg'))

        #         flash("Upload Successful")
        #         source_path = path1+"/S2.jpg"
        #         destination_path = path+"/S2.jpg"

        #         try:
        #             # Check if the source file exists
        #             if os.path.exists(source_path):
        #                 # Copy the file to the destination
        #                 shutil.copyfile(source_path, destination_path)
                    
        #         except Exception as e:
        #             pass
                    


        #         return redirect(url_for('home'))
        
        # flash("Please provide an image to upload",category="Information")

        cur = mysql.connection.cursor()

        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']

        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(path1, filename)
            file.save(file_path)

            # Insert the file information into the MySQL database, including the binary data
            cur.execute('INSERT INTO images (filename, data) VALUES (%s, %s)', (filename, file.read()))
            mysql.connection.commit()
            # Read the binary data of the file

        return render_template('index.html', image_filename=filename)

    except Exception as e:
        #TODO write a code to take image through webcam and save it as a pose 
        print(e)
        return "YET TO COMPLETE THIS PART"
    
# upload image     

@app.route('/uploads/<filename>')
def uploaded_image(filename):
    return send_from_directory(path1, filename)

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

        return jsonify({'message': 'DataUrl stored successfully'}), 200

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
    imagePoints()
    return Response(main(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_feed',methods=["GET","POST"])
def camerafeed():
    return render_template("camera_feed.html")


if __name__ == '__main__':
    app.run(debug=True)
