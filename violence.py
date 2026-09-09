import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")
detector = YOLO('Road Yolo.pt')

cap = cv2.VideoCapture(0)

# Previous wrist positions
previous_wrists = {}

# How many suspicious frames are required
REQUIRED_FRAMES = 6
suspicious_frames = 0


def distance(a, b):
    return np.linalg.norm(a - b)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.5,
        verbose=False
    )
    road_results = detector.predict(source=frame, conf=0.20, verbose=False)

    result = results[0]

    violent_frame = False

    if result.keypoints is not None and result.boxes.id is not None:

        keypoints = result.keypoints.xy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)

        # We need at least two people
        if len(keypoints) >= 2:

            for i in range(len(keypoints)):

                for j in range(len(keypoints)):

                    if i == j:
                        continue

                    person_a = keypoints[i]
                    person_b = keypoints[j]

                    id_a = ids[i]

                    # Keypoints
                    nose_a = person_a[0]
                    nose_b = person_b[0]

                    left_wrist = person_a[9]
                    right_wrist = person_a[10]

                    # ------------------------------------------------
                    # 1. Are the people reasonably close?
                    # ------------------------------------------------

                    person_distance = distance(
                        nose_a,
                        nose_b
                    )

                    if person_distance > 350:
                        continue

                    # ------------------------------------------------
                    # 2. Check BOTH wrists
                    # ------------------------------------------------

                    for wrist in [left_wrist, right_wrist]:

                        wrist_key = (id_a, tuple(wrist))

                        # We need a previous position
                        if id_a not in previous_wrists:
                            continue

                        old_wrist = previous_wrists[id_a]

                        # ------------------------------------------------
                        # 3. How fast is the wrist moving?
                        # ------------------------------------------------

                        movement = distance(
                            wrist,
                            old_wrist
                        )

                        # Ignore slow movements such as handshakes
                        if movement < 18:
                            continue

                        # ------------------------------------------------
                        # 4. Is the wrist close to the OTHER person's head?
                        # ------------------------------------------------

                        wrist_to_head = distance(
                            wrist,
                            nose_b
                        )

                        if wrist_to_head < 150:

                            # Stronger evidence of a strike
                            if movement > 30:
                                violent_frame = True

                # Save wrist position for next frame
                previous_wrists[id_a] = (
                    (person_a[9] + person_a[10]) / 2
                )

    # ------------------------------------------------------------
    # TEMPORAL FILTER
    # ------------------------------------------------------------

    if violent_frame:

        suspicious_frames += 1

    else:

        suspicious_frames = max(
            0,
            suspicious_frames - 2
        )

    # ------------------------------------------------------------
    # FINAL DECISION
    # ------------------------------------------------------------

    if suspicious_frames >= REQUIRED_FRAMES:

        text = "VIOLENCE DETECTED"
        color = (0, 0, 255)

    else:

        text = "NORMAL"
        color = (0, 255, 0)

    frame = result.plot()
    frame = road_results[0].plot(img=frame)

    cv2.putText(
        frame,
        text,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        color,
        3
    )

    cv2.putText(
        frame,
        f"Score: {suspicious_frames}/{REQUIRED_FRAMES}",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Violence Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
