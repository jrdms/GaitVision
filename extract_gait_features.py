import cv2
import mediapipe as mp
import pandas as pd
import math

# Initialize MediaPipe pose estimator
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

def calculate_distance(p1, p2):
    # Euclidean distance between two landmarks
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def extract_features(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = 0
    data = []

    left_ankle_y_history = []
    right_ankle_y_history = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # Extract relevant landmarks
            lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            la = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            ra = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

            # Calculate features
            arm_reach = calculate_distance(lw, rw)
            shoulder_width = calculate_distance(ls, rs)
            leg_distance = calculate_distance(la, ra)
            bbox_area = (max([lm.x for lm in landmarks]) - min([lm.x for lm in landmarks])) * \
                        (max([lm.y for lm in landmarks]) - min([lm.y for lm in landmarks]))

            # Store ankle y-coordinates for step detection
            left_ankle_y_history.append(la.y)
            right_ankle_y_history.append(ra.y)

            # Append features for this frame
            data.append({
                "frame": frame_num,
                "arm_reach": arm_reach,
                "shoulder_width": shoulder_width,
                "leg_distance": leg_distance,
                "bbox_area": bbox_area,
                "left_ankle_y": la.y,
                "right_ankle_y": ra.y
            })

        frame_num += 1

    cap.release()

    df = pd.DataFrame(data)

    # Step Detection (simple peak count)
    def count_steps(y_data):
        # Count local minima as steps
        steps = 0
        for i in range(1, len(y_data) - 1):
            if y_data[i] < y_data[i - 1] and y_data[i] < y_data[i + 1]:  # local minima
                steps += 1
        return steps

    left_steps = count_steps(left_ankle_y_history)
    right_steps = count_steps(right_ankle_y_history)
    total_steps = left_steps + right_steps

    # Calculate cadence (steps per minute)
    duration_seconds = len(df) / fps
    cadence = (total_steps / 2) / (duration_seconds / 60)  # steps per minute

    print(f"Estimated total steps: {total_steps}")
    print(f"Estimated cadence: {cadence:.2f} steps/min")

    # Save features to CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved features with step data to {output_csv}")

if __name__ == "__main__":
    # Run feature extraction on default video
    extract_features("gait_video.mp4", "gait_features.csv")
