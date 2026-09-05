import cv2
def draw_box(frame, box, track_id):
    # pastikan semua elemen float dulu
    box = [float(x) for x in box]

    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(frame, f"ID {track_id}", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)