from flask import Flask,render_template,Response
import cv2
import mediapipe as mp
import numpy as np
from utils import calculate_angle
from utils import provide_feedback
from utils import *




mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


print("Hello World")

app = Flask(__name__)
camera = None

def generate_frames():
    global camera

    # Curl counter variables
    counter = 0
    stage = None



    # Set arm curl angle thresholds
    arm_curl_threshold = (20, 160)
    # Initialize correct and incorrect counters
    correct_counter = 0
    incorrect_counter = 0



    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)  # Use 0 for the default camera (usually webcam)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while camera.isOpened():
                
            ## read the camera frame
            success,frame=camera.read()

            if not success:
                # If reading frames fails, break out of the loop
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # cv2.putText(image, "santosh", 
            #                (25,25), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                     )
        
            # Render detections
            # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            #                         mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
            #                         mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
            #                         )
            
            try:
                landmarks = results.pose_landmarks.landmark

                # Extract coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)

                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                feedback = provide_feedback(angle, arm_curl_threshold)

                # Curl counter logic
                previous_stage = stage

                if angle > 160:
                    stage = "down"
                elif angle < 30 and stage == 'down':
                    stage = "up"

                # Check if angle transitions within expected ranges during up and down motions
                if stage == "up" and previous_stage == "down" and angle >= arm_curl_threshold[0] and angle <= arm_curl_threshold[1]:
                    counter += 1
                    correct_counter += 1
                    print(f"Rep {counter}: Correct")
                elif stage == "down" and previous_stage == "up" and angle > arm_curl_threshold[1]:
                    incorrect_counter += 1
                    print(f"Rep {counter}: Incorrect (angle above threshold)")

                elif stage == "up" and previous_stage =="down" and angle < arm_curl_threshold[0]:
                    incorrect_counter += 1
                    print(f"Rep {counter}: Incorrect (angle below threshold)")

                # Render curl counter, status box, rep data, stage data, and detections
                # (code remains largely the same as in the original responses)

            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (65,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (60,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            # Display feedback on frame
            cv2.putText(image, feedback, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


            
            ret,buffer=cv2.imencode('.jpg',image)
            if not ret:
                # If encoding frame as JPEG fails, continue to the next iteration
                continue
            frame=buffer.tobytes()


            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                camera.release()
                cv2.destroyAllWindows()
                break


            

            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
    if camera is not None:
        camera.release()



# function for the side arm exercise 

def generate_frames_for_side_arm():
    global camera

    threshold_range = (10, 90)

    # Variables for counting exercises
    correct_counter = 0
    incorrect_counter = 0
    last_correct = False
    direction = None
    stage = None  # Start in down position for consistency
    counter = 0



    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)  # Use 0 for the default camera (usually webcam)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while camera.isOpened():
                
            ## read the camera frame
            success,frame=camera.read()

            if not success:
                # If reading frames fails, break out of the loop
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # Extracting landmark coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                # Calculate angles
                left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
                right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_body_angle = calculate_angle(left_ankle, left_hip, left_shoulder)
                right_body_angle = calculate_angle(right_ankle, right_hip, right_shoulder)

                prev_stage = stage

                if check_angles_for_posture_sidearm(left_elbow_angle, right_elbow_angle, left_body_angle, right_body_angle) and check_for_upside_shoulder_sidearm(left_shoulder_angle, right_shoulder_angle):
                    stage = "down"

                elif stage == "down" and check_angles_for_posture_sidearm(left_elbow_angle, right_elbow_angle, left_body_angle, right_body_angle) and check_for_downside_shoulder_sidearm(left_shoulder_angle, right_shoulder_angle):
                    stage = "up"

                    if check_for_threshold_sidearm(left_shoulder_angle, right_shoulder_angle):
                        counter += 1
                        correct_counter += 1
                    else:
                        stage = "up"
                        incorrect_counter += 1

                if stage == "down" and check_for_threshold_low_sidearm(left_shoulder_angle, right_shoulder_angle):
                    stage = "up"
                    incorrect_counter += 1

                if stage == "up" and check_for_threshold_high_sidearm(left_shoulder_angle, right_shoulder_angle):
                    stage = "down"
                    incorrect_counter += 1

                

                # ... (rest of the code for drawing landmarks, displaying angles, etc.)
            
                cv2.putText(image, "Correct Exercises: " + str(correct_counter), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, "Incorrect Exercises: " + str(incorrect_counter), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

                # Visualize angles
                cv2.putText(image, "Left Shoulder Angle: " + str(round(left_shoulder_angle, 2)), (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "Right Shoulder Angle: " + str(round(right_shoulder_angle, 2)), (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "Left Elbow Angle: " + str(round(left_elbow_angle, 2)), (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "Right Elbow Angle: " + str(round(right_elbow_angle, 2)), (25, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "Left Body Angle: " + str(round(left_body_angle, 2)), (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "Right Body Angle: " + str(round(right_body_angle, 2)), (25, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            

            except Exception as e:
                print(e)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )  
            
            ret,buffer=cv2.imencode('.jpg',image)
            if not ret:
                # If encoding frame as JPEG fails, continue to the next iteration
                continue
            frame=buffer.tobytes()


            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                camera.release()
                cv2.destroyAllWindows()
                break


            

            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
    if camera is not None:
        camera.release()












@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cam1')
def another_page():
    return render_template('cam1.html')

@app.route('/camera')
def camera1():
    return render_template('camera.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/side_arm')
def side_arm():
    return Response(generate_frames_for_side_arm(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return "Camera started"


@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        cv2.destroyAllWindows()
        camera = None
    return "Camera stopped"

if __name__=="__main__":
    app.run(debug=True)


