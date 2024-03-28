def generate_frames_for_wall_sit():
    global camera

    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)  # Use 0 for the default camera (usually webcam)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        start_time = time.time()  # Start time for counting
        paused_time = 0  # Time when the timer was paused
        is_paused = False  # Flag to indicate if the timer is paused

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

                # Extract coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
                left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]

                # Calculate angle
                left_hip_angle = calc_angle(left_shoulder, left_hip, left_knee)
                right_hip_angle = calc_angle(right_shoulder, right_hip, right_knee)
                left_knee_angle = calc_angle(left_hip, left_knee, left_heel)
                right_knee_angle = calc_angle(right_hip, right_knee, right_heel)

                # Check if angles are in the specified range for both hips and knees
                if check_angles_in_range(left_hip_angle, right_hip_angle, left_knee_angle, right_knee_angle):
                    if is_paused:
                        # Adjust start time based on paused time
                        start_time += (time.time() - paused_time)
                        is_paused = False
                else:
                    if not is_paused:
                        # Pause the timer
                        paused_time = time.time()
                        is_paused = True

                # Display angle values
                cv2.putText(image, str(int(left_hip_angle)),
                            tuple(np.multiply(left_hip, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(int(right_hip_angle)),
                            tuple(np.multiply(right_hip, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(int(left_knee_angle)),
                            tuple(np.multiply(left_knee, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(int(right_knee_angle)),
                            tuple(np.multiply(right_knee, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Check angles for knee health
                if check_angles_in_range_for_wallsit(left_hip_angle, right_hip_angle, left_knee_angle, right_knee_angle):
                    cv2.putText(image, "Good for knees", (20, 120), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.5,
                                (0, 0, 255), 7)
                else:
                    cv2.putText(image, "Make angles to 90 degrees for knees", (20, 120),
                                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255), 3)

                if check_angles_in_range_for_wallsit(left_hip_angle, right_hip_angle, left_knee_angle, right_knee_angle):
                    cv2.putText(image, "Good for knees", (20, 80), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.5,
                                (0, 255, 255), 7)
                else:
                    cv2.putText(image, "Make angles to 90 degrees for knees", (20, 80),
                                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 255), 3)

            except:
                pass

            # Draw landmarks and connections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            # Count and display time
            if not is_paused:
                elapsed_time = int(time.time() - start_time)
            else:
                elapsed_time = int(paused_time - start_time)
            cv2.putText(image, "Time: " + str(elapsed_time) + "s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
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









