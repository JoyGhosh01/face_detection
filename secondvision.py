import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection

cap = cv2.VideoCapture(0)  # Use index 0 for the default webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = face_detection.process(img_rgb)

        if out.detections:
            for detection in out.detections:
                location_data = detection.location_data
                bbox = location_data.relative_bounding_box

                x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

                x1 = int(x1 * W)
                y1 = int(y1 * H)
                w = int(w * W)
                h = int(h * H)

                #frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 10)
                frame[y1:y1+h, x1:x1+w] = cv2.blur(frame[y1:y1+h,x1:x1+w],(50,50))
        cv2.imshow("Detected Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the video
            break

cap.release()
cv2.destroyAllWindows()
