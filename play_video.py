import cv2

def play_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Display the current frame in a window
        cv2.imshow('Gait Video', frame)

        # Press 'q' to quit playback early
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Play the default video file
    play_video("gait_video.mp4")
