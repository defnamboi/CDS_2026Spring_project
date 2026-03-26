from ultralytics import YOLO
import cv2

model = YOLO("../../runs/segment/yolov8n_coco_bag_seg/run1/weights/best.pt")

video_path = "/../../data/raw_data/IMG_2165.MOV"  # update this path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video")
    exit()

# Optional: save output video
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))

print("Press 'q' to quit, 'p' to pause")
paused = False

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Video ended")
            break

        results = model.predict(
            source=frame,
            conf=0.25,
            verbose=False,
        )

        annotated = results[0].plot()
        out.write(annotated)  # save frame to output video
        cv2.imshow("Suspicious Bag Detection", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused

cap.release()
out.release()
cv2.destroyAllWindows()