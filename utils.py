import numpy as np


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle


def provide_feedback(angle, threshold):
    feedback = ""
    if angle < threshold[0]:
        feedback = "Angle too small. Extend your arm further."
    elif angle > threshold[1]:
        feedback = "Angle too large. Bend your arm more (but still count)."
    else:
        feedback = "Angle within range. Good job!"
    return feedback