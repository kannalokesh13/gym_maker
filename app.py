from flask import Flask,render_template,Response
import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


print("Hello World")

app = Flask(__name__)
camera = None

def generate_frames():
    global camera
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

            cv2.putText(image, "santosh", 
                           (25,25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
        
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


