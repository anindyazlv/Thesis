import os
import cv2
import time
import glob
import argparse
import math
import numpy as np
import pandas as pd
import motmetrics as mm
from ultralytics import YOLO
from boxmot import BotSort
from common.iou import compute_iou
from common.visualizer import draw_box

# =========================================================
# 1. CONFIGURATION
# =========================================================
CONF_THRES = 0.7
IOU_THRES  = 0.5
MAX_AGE    = 50
FRAME_RATE = 30

DEFAULT_IMG_DIR = "/home/comvis/Anindya/Tesis/data_video/6_2"
DEFAULT_GT_FILE = "/home/comvis/Anindya/Tesis/data_video/labels_mot360/6_2/gt.txt"
DEFAULT_MODEL   = "/home/comvis/Anindya/Tesis/results/train_results/yolov10n-ghost-att2/weights/best.pt"
BASE_OUT        = "/home/comvis/Anindya/Tesis/video testing/CPUFIX_results_v10n-ghost-att2"

parser = argparse.ArgumentParser(description="BoT-SORT Evaluation (boxmot)")
parser.add_argument("--img_dir", type=str, default=DEFAULT_IMG_DIR)
parser.add_argument("--gt_file", type=str, default=DEFAULT_GT_FILE)
parser.add_argument("--model",   type=str, default=DEFAULT_MODEL)
args = parser.parse_args()

# =========================================================
# 2. OUTPUT SETUP
# =========================================================
VIDEO_NAME = os.path.basename(os.path.normpath(args.img_dir))
OUT_DIR = os.path.join(BASE_OUT, "botsort", VIDEO_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_VIDEO      = os.path.join(OUT_DIR, f"{VIDEO_NAME}_botsort.mp4")
RESULT_TXT        = os.path.join(OUT_DIR, f"{VIDEO_NAME}_metrics.txt")
CUSTOM_SWITCH_CSV = os.path.join(OUT_DIR, f"{VIDEO_NAME}_custom_switch.csv")
MOT_SWITCH_CSV    = os.path.join(OUT_DIR, f"{VIDEO_NAME}_mot_switch.csv")
FRAME_EVENT_CSV   = os.path.join(OUT_DIR, f"{VIDEO_NAME}_frame_event.csv")

# =========================================================
# 3. LOAD MODEL & TRACKER
# =========================================================
model = YOLO(args.model)
model.to("cpu") 
tracker = BotSort(
    reid_weights=None,
    device="cpu",
    half=False,
    track_high_thresh=CONF_THRES,
    track_buffer=MAX_AGE,
    match_thresh=IOU_THRES,
    min_hits=1,
    with_reid=False,
    gmc_method='sparseOptFlow',
)

# =========================================================
# 4. LOAD GT
# =========================================================
columns = ['frame','id','x','y','width','height','conf','x8','x9','x10']
gt_data = pd.read_csv(args.gt_file, header=None, names=columns).set_index('frame')

# =========================================================
# 5. VIDEO SETUP
# =========================================================
image_paths = sorted(glob.glob(os.path.join(args.img_dir, '*.jpg')))
if not image_paths:
    raise FileNotFoundError(f"Tidak ada gambar .jpg di: {args.img_dir}")

sample = cv2.imread(image_paths[0])
h, w = sample.shape[:2]
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), FRAME_RATE, (w, h))
acc = mm.MOTAccumulator(auto_id=True)

# =========================================================
# 6. TRACKING LOOP
# =========================================================
target_id            = None
last_target_box      = None
last_matched_pred_id = None

id_switch_events = []
frame_event_log  = []
start_time       = time.time()
frame_count      = 0

for idx, path in enumerate(image_paths):

    frame_idx = idx + 1
    frame     = cv2.imread(path)

    # ===== DETECTION =====
    results    = model(frame, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0 and box.conf[0] >= CONF_THRES:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                detections.append([x1, y1, x2, y2, float(box.conf[0]), 0])
    detections = np.array(detections) if detections else np.empty((0, 6))

    # ===== TRACKER UPDATE =====
    tracked = tracker.update(detections, frame)

    # ===== TARGET SELECTION =====
    target_box, target_tid = None, None

    if len(tracked) > 0:
        if target_id is None:
            target_id       = int(tracked[0][4])
            target_box      = tracked[0][:4].tolist()
            target_tid      = target_id
            last_target_box = target_box.copy()
        else:
            found = False
            for obj in tracked:
                if int(obj[4]) == target_id:
                    target_box      = obj[:4].tolist()
                    target_tid      = target_id
                    last_target_box = target_box.copy()
                    found           = True
                    break

            if not found and last_target_box is not None:
                best_iou, best = 0, None
                for obj in tracked:
                    iou = compute_iou(last_target_box, obj[:4])
                    if iou > best_iou:
                        best_iou, best = iou, obj

                if best is not None and best_iou >= IOU_THRES:
                    old_id  = target_id           # simpan sebelum diupdate
                    new_id  = int(best[4])

                    id_switch_events.append({
                        "frame":  frame_idx,
                        "old_id": old_id,         # FIX: old_id benar diambil sebelum target_id diupdate
                        "new_id": new_id,
                        "iou":    round(best_iou, 4)
                    })

                    target_id       = new_id
                    target_box      = best[:4].tolist()
                    target_tid      = new_id
                    last_target_box = target_box.copy()

    if target_box is not None:
        draw_box(frame, target_box, target_tid)

    # ===== EVALUATION =====
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

            gt_box    = [[gx, gy, gx + gw, gy + gh]]
            gt_id     = [int(gt_frame['id'])]
            gt_id_val = gt_id[0]

            if target_box is not None:
                pred_box    = [[target_box[0], target_box[1], target_box[2], target_box[3]]]
                pred_id     = [target_tid]
                pred_id_val = target_tid

                dist      = mm.distances.iou_matrix(gt_box, pred_box, max_iou=0.5)
                acc.update(gt_id, pred_id, dist)

                raw       = dist[0][0]
                iou_value = 1 - raw if not np.isnan(raw) else 0.0

                if last_matched_pred_id and pred_id_val != last_matched_pred_id:
                    event_type = "SWITCH"
                else:
                    event_type = "MATCH" if iou_value > 0 else "MISS"  # FIX: bedakan MATCH vs MISS saat IoU=0

                last_matched_pred_id = pred_id_val

            else:
                acc.update(gt_id, [], [])
                event_type           = "MISS"
                last_matched_pred_id = None

        else:
            # FIX: tangani frame tanpa GT (false positive)
            if target_box is not None:
                acc.update([], [target_tid], np.empty((0, 1)))
                event_type = "FP"
            else:
                acc.update([], [], [])

    except Exception as e:
        print(f"[ERROR] Frame {frame_idx}: {e}")

    frame_event_log.append({
        "frame":   frame_idx,
        "gt_id":   gt_id_val,
        "pred_id": pred_id_val,
        "iou":     round(iou_value, 4),
        "event":   event_type
    })

    out.write(frame)
    frame_count += 1

    if frame_idx % 50 == 0:
        elapsed = time.time() - start_time
        print(f"[PROGRESS] Frame {frame_idx}/{len(image_paths)} | FPS: {frame_count / elapsed:.1f}")

# =========================================================
# 7. FINAL METRICS
# =========================================================
out.release()
pd.options.display.float_format = '{:.3f}'.format
mh      = mm.metrics.create()
summary = mh.compute(
    acc,
    metrics=['mota', 'motp', 'idf1', 'num_switches', 'precision', 'recall'],
    name='SingleTarget-Botsort'
)

fps = frame_count / (time.time() - start_time)

print("\n=== FINAL RESULT ===")
print(summary)
print(f"FPS: {fps:.0f}")

# Catatan: motmetrics menghitung MOTP sebagai rata-rata distance (1 - IoU).
# Untuk melaporkan dalam satuan IoU: MOTP_IoU = 1 - MOTP_distance
motp_distance = summary.loc['SingleTarget-Botsort', 'motp']
if not math.isnan(motp_distance):
    print(f"MOTP (sebagai IoU, untuk pelaporan): {1 - motp_distance:.3f}")

# =========================================================
# 8. SAVE RESULTS
# =========================================================
pd.DataFrame(id_switch_events).to_csv(CUSTOM_SWITCH_CSV, index=False)
pd.DataFrame(frame_event_log).to_csv(FRAME_EVENT_CSV, index=False)

events_df = acc.events.reset_index()
switch    = events_df[events_df['Type'] == 'SWITCH']

cols = []
if 'FrameId' in switch.columns: cols.append('FrameId')
if 'OId'     in switch.columns: cols.append('OId')
if 'HId'     in switch.columns: cols.append('HId')

if len(switch) > 0 and len(cols) == 3:
    df         = switch[cols].copy()
    df.columns = ['frame', 'gt_id', 'pred_id']
else:
    df = pd.DataFrame(columns=['frame', 'gt_id', 'pred_id'])

df.to_csv(MOT_SWITCH_CSV, index=False)

with open(RESULT_TXT, "w") as f:
    f.write(summary.to_string())
    f.write("\n\n")
    if not math.isnan(motp_distance):
        f.write(f"MOTP (sebagai IoU): {1 - motp_distance:.4f}\n")
    f.write(f"FPS: {fps:.0f}\n")

print("\n[INFO] Semua hasil tersimpan di:", OUT_DIR)
