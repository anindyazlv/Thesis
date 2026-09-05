# =========================================================
# IMPORTS
# =========================================================
import os
import cv2
import time
import glob
import argparse
import numpy as np
import pandas as pd
import motmetrics as mm

from ultralytics import YOLO
from ocsort.ocsort import OCSort

from common.iou import compute_iou
from common.visualizer import draw_box


# =========================================================
# 1. CONFIGURATION
# =========================================================
CONF_THRES = 0.9
IOU_THRES  = 0.5

MAX_AGE  = 50
MIN_HITS = 1
DELTA_T  = 3
INERTIA  = 0.2
USE_BYTE = False

FRAME_RATE = 30

DEFAULT_IMG_DIR = "/home/comvis/Anindya/Tesis/data_video/6_2"
DEFAULT_GT_FILE = "/home/comvis/Anindya/Tesis/data_video/labels_mot360/6_2/gt.txt"
DEFAULT_MODEL   = "/home/comvis/Anindya/Tesis/results/train_results/yolov10n-ghost-att2/weights/best.pt"
BASE_OUT        = "/home/comvis/Anindya/Tesis/video testing/CPUFIX_results_v10n-ghost-att2"


# =========================================================
# 2. CLI ARGUMENT
# =========================================================
parser = argparse.ArgumentParser(description="Single Target Tracking with OC-SORT")

parser.add_argument("--img_dir", type=str, default=DEFAULT_IMG_DIR)
parser.add_argument("--gt_file", type=str, default=DEFAULT_GT_FILE)
parser.add_argument("--model",   type=str, default=DEFAULT_MODEL)

args = parser.parse_args()

IMG_DIR    = args.img_dir
GT_FILE    = args.gt_file
MODEL_PATH = args.model


# =========================================================
# 3. OUTPUT SETUP
# =========================================================
VIDEO_NAME = os.path.basename(os.path.normpath(IMG_DIR))

OUT_DIR = os.path.join(BASE_OUT, "ocsort", VIDEO_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_VIDEO    = os.path.join(OUT_DIR, f"{VIDEO_NAME}_ocsort.mp4")
RESULT_TXT      = os.path.join(OUT_DIR, f"{VIDEO_NAME}_metrics.txt")
CUSTOM_SWITCH_CSV = os.path.join(OUT_DIR, f"{VIDEO_NAME}_custom_switch.csv")
MOT_SWITCH_CSV    = os.path.join(OUT_DIR, f"{VIDEO_NAME}_mot_switch.csv")
FRAME_EVENT_CSV   = os.path.join(OUT_DIR, f"{VIDEO_NAME}_frame_event.csv")


# =========================================================
# 4. LOAD MODEL & TRACKER
# =========================================================
model = YOLO(MODEL_PATH)
model.to("cpu") 
tracker = OCSort(
    det_thresh=CONF_THRES,   # catatan: deteksi sudah difilter manual sebelum masuk tracker,
                              # sehingga det_thresh ini tidak berefek ganda secara signifikan
    max_age=MAX_AGE,
    min_hits=MIN_HITS,
    iou_threshold=IOU_THRES,
    delta_t=DELTA_T,
    inertia=INERTIA,
    use_byte=USE_BYTE
)

# =========================================================
# 5. LOAD GROUND TRUTH
# =========================================================
columns = ['frame', 'id', 'x', 'y', 'width', 'height',
           'conf', 'x8', 'x9', 'x10']

gt_raw  = pd.read_csv(GT_FILE, header=None, names=columns)
gt_data = gt_raw.set_index('frame')


# =========================================================
# 6. PREPARE VIDEO & METRICS
# =========================================================
image_paths = sorted(glob.glob(os.path.join(IMG_DIR, '*.jpg')))
if not image_paths:
    raise FileNotFoundError(f"Tidak ada gambar .jpg di: {IMG_DIR}")

sample_img    = cv2.imread(image_paths[0])
height, width = sample_img.shape[:2]

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    FRAME_RATE,
    (width, height)
)

acc = mm.MOTAccumulator(auto_id=True)


# =========================================================
# 7. TRACKING LOOP
# =========================================================
target_id       = None
last_target_box = None

id_switch_events     = []
frame_event_log      = []
last_matched_pred_id = None

frame_count = 0
start_time  = time.time()


for idx, path in enumerate(image_paths):

    frame_idx = idx + 1
    frame     = cv2.imread(path)

    # ---------------- DETECTION ----------------
    detections = []
    results    = model(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0 and box.conf[0] >= CONF_THRES:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                detections.append([x1, y1, x2, y2, float(box.conf[0])])

    detections = np.array(detections) if detections else np.empty((0, 5))

    # ---------------- TRACKER ----------------
    tracked = tracker.update(detections, (height, width), (height, width))

    has_tracks = (
        tracked is not None and
        tracked.size > 0 and
        tracked.ndim == 2 and
        tracked.shape[1] >= 5
    )

    target_box = None
    target_tid = None

    # ---------------- TARGET SELECTION ----------------
    if has_tracks:

        if target_id is None:
            target_id       = int(tracked[0][4])
            last_target_box = list(map(float, tracked[0][:4]))
            target_box      = last_target_box.copy()
            target_tid      = target_id

        else:
            found = False

            for obj in tracked:
                if obj.shape[0] >= 5:
                    tid = int(obj[4])
                    if tid == target_id:
                        target_box      = list(map(float, obj[:4]))
                        target_tid      = tid
                        last_target_box = target_box.copy()
                        found = True
                        break

            # -------- RE-ID (IoU fallback) --------
            if not found and last_target_box is not None:
                best_iou       = 0.0
                best_candidate = None

                for obj in tracked:
                    if obj.shape[0] >= 5:
                        x1, y1, x2, y2, tid = obj[:5]

                        iou = compute_iou(
                            last_target_box,
                            [float(x1), float(y1), float(x2), float(y2)]
                        )

                        if iou > best_iou:
                            best_iou       = iou
                            best_candidate = obj

                if best_candidate is not None and best_iou >= IOU_THRES:
                    x1, y1, x2, y2, new_tid = best_candidate[:5]
                    new_tid = int(new_tid)

                    id_switch_events.append({
                        "frame": frame_idx,
                        "old_id": target_id,
                        "new_id": new_tid,
                        "iou": round(float(best_iou), 4)
                    })

                    target_id       = new_tid
                    target_box      = [x1, y1, x2, y2]
                    target_tid      = new_tid
                    last_target_box = target_box.copy()

    if target_box is not None:
        draw_box(frame, target_box, target_tid)


    # ---------------- EVALUATION ----------------
    event_type  = "NONE"
    iou_value   = 0.0
    gt_id_val   = None
    pred_id_val = None

    try:
        if frame_idx in gt_data.index:

            gt_frame = gt_data.loc[frame_idx]
            if isinstance(gt_frame, pd.DataFrame):
                gt_frame = gt_frame.iloc[0]

            gx, gy = float(gt_frame['x']), float(gt_frame['y'])
            gw, gh = float(gt_frame['width']), float(gt_frame['height'])

            gt_box = [[gx, gy, gx + gw, gy + gh]]
            gt_id  = [int(gt_frame['id'])]
            gt_id_val = gt_id[0]

            if target_box is not None:
                pred_box = [[target_box[0], target_box[1],
                             target_box[2], target_box[3]]]
                pred_id  = [target_tid]
                pred_id_val = target_tid

                dist = mm.distances.iou_matrix(gt_box, pred_box, max_iou=0.5)
                acc.update(gt_id, pred_id, dist)

                raw = dist[0][0]
                iou_value = 1 - raw if not np.isnan(raw) else 0.0

                if last_matched_pred_id and pred_id_val != last_matched_pred_id:
                    event_type = "SWITCH"
                else:
                    event_type = "MATCH" if iou_value > 0 else "MISS"

                last_matched_pred_id = pred_id_val

            else:
                acc.update(gt_id, [], [])
                event_type = "MISS"
                last_matched_pred_id = None

        else:
            if target_box is not None:
                acc.update([], [target_tid], np.empty((0, 1)))
                event_type = "FP"
            else:
                acc.update([], [], [])

    except Exception as e:
        print(f"[ERROR] Frame {frame_idx}: {e}")

    frame_event_log.append({
        "frame": frame_idx,
        "gt_id": gt_id_val,
        "pred_id": pred_id_val,
        "iou": round(iou_value, 4),
        "event": event_type
    })

    out.write(frame)
    frame_count += 1

    if frame_idx % 50 == 0:
        elapsed = time.time() - start_time
        print(f"[PROGRESS] Frame {frame_idx}/{len(image_paths)} | FPS: {frame_count / elapsed:.1f}")

# =========================================================
# 8. FINAL METRICS
# =========================================================
out.release()
pd.options.display.float_format = '{:.3f}'.format
mh = mm.metrics.create()
summary = mh.compute(
    acc,
    metrics=['mota','motp','idf1','num_switches','precision','recall'],
    name='SingleTarget-OCSORT'
)

fps = frame_count / (time.time() - start_time)

print("\n=== FINAL RESULT ===")
print(summary)
print(f"FPS: {fps:.0f}")

# Catatan: motmetrics menghitung MOTP sebagai rata-rata distance (1 - IoU).
# Untuk melaporkan dalam satuan IoU: MOTP_IoU = 1 - MOTP_distance
motp_distance = summary.loc['SingleTarget-OCSORT', 'motp']
if not __import__('math').isnan(motp_distance):
    print(f"MOTP (sebagai IoU, untuk pelaporan): {1 - motp_distance:.3f}")


# =========================================================
# 9. SAVE RESULTS
# =========================================================
pd.DataFrame(id_switch_events).to_csv(CUSTOM_SWITCH_CSV, index=False)
pd.DataFrame(frame_event_log).to_csv(FRAME_EVENT_CSV, index=False)

events = acc.events

# Reset index supaya FrameId jadi kolom
events_df = events.reset_index()

# Filter SWITCH
switch = events_df[events_df['Type'] == 'SWITCH']

# Ambil kolom dengan aman
cols = []

if 'FrameId' in switch.columns:
    cols.append('FrameId')
if 'OId' in switch.columns:
    cols.append('OId')
if 'HId' in switch.columns:
    cols.append('HId')

if len(switch) > 0 and len(cols) == 3:
    df = switch[cols].copy()
    df.columns = ['frame', 'gt_id', 'pred_id']
else:
    df = pd.DataFrame(columns=['frame','gt_id','pred_id'])

df.to_csv(MOT_SWITCH_CSV, index=False)

with open(RESULT_TXT, "w") as f:
    f.write(summary.to_string())
    f.write("\n\n")
    motp_d = summary.loc['SingleTarget-OCSORT', 'motp']
    if not __import__('math').isnan(motp_d):
        f.write(f"MOTP (sebagai IoU): {1 - motp_d:.4f}\n")
    f.write(f"FPS: {fps:.0f}\n")

print("\n[INFO] Semua hasil tersimpan di:", OUT_DIR)