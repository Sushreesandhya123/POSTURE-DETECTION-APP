from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import numpy as np
import tempfile

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Utility function
def get_angle(a, b, c):
    """Returns the angle at point b given 3 points a, b, c."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

@app.post("/analyze/")
async def analyze_video(file: UploadFile = File(...)):
    try:
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(await file.read())
            temp_path = temp_video.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return JSONResponse(status_code=400, content={"error": "Failed to read video file."})

        frame_count = 0
        bad_posture_frames = []

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if not results.pose_landmarks:
                continue

            landmarks = results.pose_landmarks.landmark

            try:
                # Extract relevant points
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                toe = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                        landmarks[mp_pose.PoseLandmark.NOSE.value].y]

                # Calculate angles
                back_angle = get_angle(shoulder, hip, knee)
                knee_over_toe = knee[0] > toe[0]
                neck_bend = abs(nose[1] - shoulder[1]) > 0.05

                # Posture rule-based logic
                is_bad = False
                reason = ""

                if back_angle < 150:
                    is_bad = True
                    reason += "Back angle < 150. "
                if knee_over_toe:
                    is_bad = True
                    reason += "Knee over toe. "
                if neck_bend:
                    is_bad = True
                    reason += "Neck bend > 30°. "

                if is_bad:
                    bad_posture_frames.append({
                        "frame": frame_count,
                        "reasons": reason.strip()
                    })

            except Exception as e:
                print(f"[ERROR] Frame {frame_count} failed: {str(e)}")
                continue

        cap.release()

        return {"bad_posture_frames": bad_posture_frames}

    except Exception as e:
        print(f"[SERVER ERROR] {str(e)}")
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})
