import sys
import os
from types import ModuleType

lib_path = r'C:\face\lib'
models_dir = r'C:\face\lib\face_recognition_models\models'
if lib_path not in sys.path: sys.path.insert(0, lib_path)

if 'face_recognition_models' not in sys.modules:
    mock_models = ModuleType('face_recognition_models')
    mock_models.__path__ = [os.path.join(lib_path, 'face_recognition_models')]
    mock_models.model_location = lambda: models_dir
    mock_models.pose_predictor_model_location = lambda: os.path.join(models_dir, "shape_predictor_68_face_landmarks.dat")
    mock_models.pose_predictor_five_point_model_location = lambda: os.path.join(models_dir, "shape_predictor_5_face_landmarks.dat")
    mock_models.face_recognition_model_location = lambda: os.path.join(models_dir, "dlib_face_recognition_resnet_model_v1.dat")
    mock_models.cnn_face_detector_model_location = lambda: os.path.join(models_dir, "mmod_human_face_detector.dat")
    sys.modules['face_recognition_models'] = mock_models

import cv2
import pickle
import numpy as np
import threading
import time
from PIL import Image, ImageDraw, ImageFont
import face_recognition
from mediapipe.python.solutions import face_detection as mp_face_detection

try:
    font_path = "C:/Windows/Fonts/arial.ttf"
    f_sm = ImageFont.truetype(font_path, 18)
    f_lg = ImageFont.truetype(font_path, 24)
    f_ui = ImageFont.truetype(font_path, 14)
except:
    f_sm = f_lg = f_ui = ImageFont.load_default()

def draw_unicode(img, text, pos, font, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

DB_PATH = r"C:\face\face_data.pkl"
def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict) and "users" in data:
                return data["users"], data.get("next_uid", 0)
    return {}, 0

known_users, next_db_uid = load_db()
def save_db():
    with open(DB_PATH, 'wb') as f:
        pickle.dump({"users": known_users, "next_uid": next_db_uid}, f)

mode = "SCAN"
input_buffer = ""; target_id = ""
tracked_faces = {}; next_session_id = 0
last_save_encoding = None
manual_trigger = False
fps = 0; frame_count = 0; start_time = time.time()

def recognition_worker(face_img, session_id, users_dict):
    global tracked_faces, last_save_encoding
    try:
        small = cv2.resize(face_img, (150, 150))
        encs = face_recognition.face_encodings(small, [(0, 150, 150, 0)], num_jitters=1)
        if encs:
            last_save_encoding = encs[0]
            name, uid_str = "Unknown", "NONE"
            if users_dict:
                uids = list(users_dict.keys())
                enc_list = [u["enc"] for u in users_dict.values()]
                matches = face_recognition.compare_faces(enc_list, encs[0], tolerance=0.6)
                if True in matches:
                    idx = matches.index(True)
                    name = users_dict[uids[idx]]["name"]
                    uid_str = str(uids[idx])
            if session_id in tracked_faces:
                tracked_faces[session_id].update({"name": name, "uid": uid_str})
    except: pass

face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)

cv2.namedWindow('Robot Master', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Robot Master', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    success, raw_frame = cap.read()
    if not success: break
    frame = cv2.flip(raw_frame, 1)
    h, w, _ = frame.shape

    frame_count += 1
    if time.time() - start_time > 1:
        fps = frame_count / (time.time() - start_time)
        frame_count = 0; start_time = time.time()

    ui_frame = np.zeros((h, w + 250, 3), dtype=np.uint8)
    ui_frame[:, :w] = frame
    
    is_master = False; is_face = False

    if mode == "SCAN":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)
        if results.detections:
            is_face = True
            for d in results.detections:
                b = d.location_data.relative_bounding_box
                bx, by, bw, bh = int(b.xmin*w), int(b.ymin*h), int(b.width*w), int(b.height*h)
                bx, by = max(0, bx), max(0, by)
                center = (bx + bw//2, by + bh//2)
                
                sid = None; min_dist = 80 
                for s, data in tracked_faces.items():
                    dist = np.linalg.norm(np.array(center)-np.array(data["center"]))
                    if dist < min_dist: sid = s; min_dist = dist
                
                if sid is None or manual_trigger:
                    if sid is None: sid = next_session_id; next_session_id += 1
                    tracked_faces[sid] = {"center": center, "name": "Scanning...", "uid": "NONE", "last_seen": time.time()}
                    crop = rgb[by:by+bh, bx:bx+bw]
                    if crop.size > 0:
                        threading.Thread(target=recognition_worker, args=(crop, sid, known_users), daemon=True).start()
                else:
                    tracked_faces[sid].update({"center": center, "last_seen": time.time()})
                    if str(tracked_faces[sid]["uid"]) == "0": is_master = True

                name = tracked_faces[sid]["name"]
                color = (0, 255, 0) if name not in ["Unknown", "Scanning..."] else (0, 0, 255)
                cv2.rectangle(ui_frame, (bx, by), (bx+bw, by+bh), color, 2)
                ui_frame = draw_unicode(ui_frame, name, (bx, by-30), f_sm, color)
            manual_trigger = False 
        else: manual_trigger = False
        tracked_faces = {s: d for s, d in tracked_faces.items() if time.time() - d["last_seen"] < 0.6}

    msg, col = ("ADMIN CONTROL: GRANTED", (0, 255, 0)) if is_master else \
               ("ACCESS RESTRICTED", (0, 0, 255)) if is_face else \
               ("SYSTEM: STANDBY", (150, 150, 150))
    
    cv2.rectangle(ui_frame, (0, 0), (w, 60), (0, 0, 0), -1)
    t_sz = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0]
    cv2.putText(ui_frame, msg, ((w - t_sz[0]) // 2, 42), cv2.FONT_HERSHEY_DUPLEX, 1.0, col, 2, cv2.LINE_AA)

    ui_frame = draw_unicode(ui_frame, "Face DB Server", (w+15, 15), f_lg, (0, 255, 0))
    y = 65
    for uid, info in sorted(known_users.items()):
        if y > h - 180: break
        ui_frame = draw_unicode(ui_frame, f"ID {uid}: {info['name']}", (w+15, y), f_sm); y += 25

    cv2.rectangle(ui_frame, (w+5, h-160), (w+245, h-5), (25, 25, 25), -1)
    ui_frame = draw_unicode(ui_frame, f"FPS: {fps:.1f}", (w+15, h-150), f_ui, (0, 255, 255))
    ly = h-120
    for txt in ["[SPACE] MANUAL SCAN", "[S] SAVE FACE", "[E] EDIT ID", "[D] DELETE ID", "[Q] QUIT"]:
        ui_frame = draw_unicode(ui_frame, txt, (w+15, ly), f_ui, (150, 150, 150)); ly += 18

    if mode != "SCAN":
        overlay = ui_frame.copy(); cv2.rectangle(overlay, (w//2-220, h//2-90), (w//2+220, h//2+90), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.85, ui_frame, 0.15, 0, ui_frame)
        prompt = "Enter Name:" if mode == "SAVE" else "Enter ID to Edit:" if target_id == "" else "New Name:"
        ui_frame = draw_unicode(ui_frame, f"{mode} MODE", (w//2-200, h//2-70), f_lg, (0, 255, 0))
        ui_frame = draw_unicode(ui_frame, prompt, (w//2-200, h//2-30), f_sm, (180, 180, 180))
        cv2.rectangle(ui_frame, (w//2-200, h//2+10), (w//2+200, h//2+50), (40, 40, 40), -1)
        ui_frame = draw_unicode(ui_frame, input_buffer + "|", (w//2-190, h//2+20), f_sm)

    cv2.imshow('Robot Master', ui_frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'): break 
    
    if mode == "SCAN":
        if key == ord('s') and last_save_encoding is not None: mode = "SAVE"; input_buffer = ""
        elif key == ord('e'): mode = "EDIT"; input_buffer = ""; target_id = ""
        elif key == ord('d'): mode = "DELETE"; input_buffer = ""
        elif key == 32: manual_trigger = True
    else:
        if key == 27: mode = "SCAN" # ESC to cancel
        elif key == 13: # ENTER to confirm
            try:
                if mode == "SAVE" and input_buffer:
                    known_users[next_db_uid] = {"name": input_buffer, "enc": last_save_encoding}
                    next_db_uid += 1; save_db()
                elif mode == "EDIT":
                    if target_id == "": 
                        if int(input_buffer) in known_users: target_id = input_buffer; input_buffer = ""
                        else: mode = "SCAN"
                        continue
                    else: 
                        known_users[int(target_id)]["name"] = input_buffer; save_db(); mode = "SCAN"
                elif mode == "DELETE":
                    if int(input_buffer) in known_users: del known_users[int(input_buffer)]; save_db()
            except: pass
            mode = "SCAN"; input_buffer = ""; target_id = ""
        elif key == 8: input_buffer = input_buffer[:-1] # Backspace
        elif key != -1:
            try: 
                char = chr(key)
                if char.isprintable(): input_buffer += char
            except: pass

cap.release(); cv2.destroyAllWindows()