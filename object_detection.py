from ultralytics import YOLO
import cv2
import csv

VIDEO_PATH = 'gait_video.mp4'
OUTPUT_CSV = 'object_detections.csv'

# Load YOLOv8 pre-trained model (nano or small for speed)
model = YOLO('yolov8n.pt')

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_num = 0

    # Open CSV file for writing detections
    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write CSV header
        writer.writerow(['frame', 'class', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'])

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            # Run detection on frame (model returns list of results)
            results = model(frame)[0]

            for result in results.boxes.data.tolist():
                # YOLOv8 result box format: [xmin, ymin, xmax, ymax, confidence, class]
                xmin, ymin, xmax, ymax, conf, cls = result
                cls_name = model.names[int(cls)]

                # Write detection to CSV
                writer.writerow([frame_num, cls_name, conf, int(xmin), int(ymin), int(xmax), int(ymax)])

                # Draw bounding box & label on frame
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls_name} {conf:.2f}", (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # Show detection results in a window
            cv2.imshow('Object Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Run object detection on default video
    main()
