import cv2
import mediapipe as mp
import random
import time
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# MediaPipe Pose 初期化
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# スコア履歴
score_file = "scores.txt"
scores = []
if os.path.exists(score_file):
    with open(score_file, "r") as f:
        for line in f:
            try:
                scores.append(int(line.strip()))
            except:
                pass

# ゲーム設定
game_time = 30
score = 0
target_radius = 50
frame_counter = 0
new_target_frame_interval = 25
hit_flag = False
target_type = 0
target_x = 100
target_y = 100

# 背景・スタート画像
bg = cv2.imread("background.jpg")
start_img = cv2.imread("start_screen.jpg")
if bg is None or start_img is None:
    print("背景またはスタート画面画像が見つかりません。")
    exit()

# フォント設定
font_path = "Creepster-Regular.ttf"
font_title = ImageFont.truetype(font_path, 100)
font_sub = ImageFont.truetype(font_path, 50)
font_text = ImageFont.truetype(font_path, 40)
font_small = ImageFont.truetype(font_path, 35)

# フルスクリーン
cv2.namedWindow("Pose Game", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Pose Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

state = "start"
start_time = None

# 中央揃え関数
def draw_centered_text(draw, text, font, y, color, image_width, shadow=False):
    text_width = draw.textlength(text, font=font)
    x = (image_width - text_width) // 2
    if shadow:
        draw.text((x+2, y+2), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=color)

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    enable_segmentation=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
) as pose:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        bg_resized = cv2.resize(bg, (frame.shape[1], frame.shape[0]))
        display_frame = bg_resized.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # === スタート画面 ===
        if state == "start":
            start_resized = cv2.resize(start_img, (frame.shape[1], frame.shape[0]))
            img_pil = Image.fromarray(cv2.cvtColor(start_resized, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            # タイトルとテキスト位置調整（もう少し上＆明るめに）
            draw_centered_text(draw, "GHOST HUNTER", font_title, 180, (255, 255, 255), frame.shape[1], shadow=True)
            draw_centered_text(draw, "Press SPACE to Start", font_sub, 350, (255, 255, 255), frame.shape[1], shadow=True)

            display_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            cv2.imshow("Pose Game", display_frame)

            key = cv2.waitKey(5) & 0xFF
            if key == 32:
                score = 0
                start_time = time.time()
                state = "play"
            elif key == 27:
                break
            continue

        # === ゲーム中 ===
        if state == "play":
            frame_counter += 1
            if frame_counter % new_target_frame_interval == 0:
                h, w, _ = display_frame.shape
                target_type = random.choice([0, 1])
                target_x = random.randint(target_radius, w - target_radius - 1)
                target_y = random.randint(target_radius, h - target_radius - 1)
                frame_counter = 0

            right_wrist_x = right_wrist_y = None
            left_wrist_x = left_wrist_y = None

            if results.pose_landmarks:
                right_wrist = results.pose_landmarks.landmark[15]
                left_wrist = results.pose_landmarks.landmark[16]

                right_wrist_x = int(right_wrist.x * display_frame.shape[1])
                right_wrist_y = int(right_wrist.y * display_frame.shape[0])
                left_wrist_x = int(left_wrist.x * display_frame.shape[1])
                left_wrist_y = int(left_wrist.y * display_frame.shape[0])

                cv2.circle(display_frame, (right_wrist_x, right_wrist_y), 10, (0, 255, 0), -1)
                cv2.circle(display_frame, (left_wrist_x, left_wrist_y), 10, (0, 255, 0), -1)

                distance_right = ((right_wrist_x - target_x)**2 + (right_wrist_y - target_y)**2)**0.5
                distance_left = ((left_wrist_x - target_x)**2 + (left_wrist_y - target_y)**2)**0.5

                if (distance_right < target_radius or distance_left < target_radius) and not hit_flag:
                    score += (1 if target_type == 0 else -1)
                    hit_flag = True
                    target_x = -100
                    frame_counter = 0
                elif distance_right >= target_radius and distance_left >= target_radius:
                    hit_flag = False

            color = (0, 0, 255) if target_type == 0 else (255, 0, 0)
            cv2.circle(display_frame, (target_x, target_y), target_radius, color, -1)

            elapsed = int(time.time() - start_time)
            remaining = max(0, game_time - elapsed)

            img_pil = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            draw.text((30, 30), f"TIME: {remaining}", font=font_text, fill=(255, 255, 255))
            draw.text((400, 30), f"SCORE: {score}", font=font_text, fill=(255, 255, 255))
            display_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            if remaining <= 0:
                scores.append(score)
                with open(score_file, "w") as f:
                    for s in scores:
                        f.write(str(s) + "\n")
                state = "result"

        # === リザルト画面 ===
        elif state == "result":
            display_frame[:] = (0, 0, 0)
            img_pil = Image.fromarray(display_frame)
            draw = ImageDraw.Draw(img_pil)

            base_y = 50  # ← 全体をさらに上に
            draw_centered_text(draw, "GAME OVER", font_title, base_y, (255, 0, 0), frame.shape[1])
            draw_centered_text(draw, f"YOUR SCORE: {score}", font_sub, base_y + 100, (255, 255, 255), frame.shape[1])

            # ランキング（上位3位のみ）
            ranking = sorted(scores, reverse=True)[:3]
            draw_centered_text(draw, "RANKING", font_sub, base_y + 200, (0, 255, 0), frame.shape[1])
            for i, s in enumerate(ranking):
                draw_centered_text(draw, f"{i+1}. {s}", font_small, base_y + 250 + i * 40, (255, 255, 255), frame.shape[1])

            draw_centered_text(draw, "Press SPACE to Restart or ESC to Quit", font_small, base_y + 430, (200, 200, 200), frame.shape[1])

            display_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            cv2.imshow("Pose Game", display_frame)

            key = cv2.waitKey(5) & 0xFF
            if key == 32:
                score = 0
                start_time = time.time()
                state = "play"
            elif key == 27:
                break
            continue

        cv2.imshow("Pose Game", display_frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
