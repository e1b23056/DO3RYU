import cv2
import mediapipe as mp
import random
import time
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math
import pygame

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
game_time = 40
score = 0
combo = 0
last_hit_time = 0
target_radius = 50
effect_size = target_radius * 2
frame_counter = 0
new_target_frame_interval = 25
hit_flag = False
target_type = 0
target_x = 100
target_y = 100

target_amp_x = 150  # X方向の動きの振幅（ピクセル）
target_amp_y = 100   # Y方向の動きの振幅（ピクセル）
target_speed = 1.0  # 動きの速さ（秒間）
target_center_x = 100 # 動きの中心X座標
target_center_y = 100 # 動きの中心Y座標

targets = []  # 各お化けのデータを格納
max_targets = 2  # 同時に表示する最大数
hit_effects = []  # 魂抜けエフェクトのデータを格納

pygame.mixer.init(frequency = 44100)    # 初期設定
bgm = "music/bgm.mp3"
end_BGM = "music/endbgm.mp3"
se_hit = pygame.mixer.Sound("music/se.mp3")  # ヒット音の読み込み

def generate_new_target(prev_x, prev_y, radius, amp_x, amp_y, w, h, min_dist=120):#的の位置を離れるような設定
    while True:
        new_x = random.randint(radius + amp_x, w - radius - amp_x - 1)
        new_y = random.randint(radius + amp_y, h - radius - amp_y - 1)
        distance = ((new_x - prev_x) ** 2 + (new_y - prev_y) ** 2) ** 0.5
        if distance > min_dist:
            return new_x, new_y

# 背景・スタート画像
bg = cv2.imread("images/background.jpg")
start_img = cv2.imread("images/start_screen.jpg")
if bg is None or start_img is None:
    print("背景またはスタート画面画像が見つかりません。")
    exit()

# 的画像の読み込み
# 的画像の読み込み
target_img_normal = cv2.imread("images/obake_green.png", cv2.IMREAD_UNCHANGED) # 普通用
target_img_right = cv2.imread("images/obake_red.png", cv2.IMREAD_UNCHANGED)   # 右用
target_img_left = cv2.imread("images/obake_blue.png", cv2.IMREAD_UNCHANGED) # 左用
soul_effect_blue = cv2.imread("images/soul_blue.png", cv2.IMREAD_UNCHANGED)
soul_effect_red = cv2.imread("images/soul_red.png", cv2.IMREAD_UNCHANGED)
soul_effect_green = cv2.imread("images/soul_green.png", cv2.IMREAD_UNCHANGED)

if target_img_normal is None or target_img_right is None or target_img_left is None:
    print("的の画像が見つかりません。")
    exit()

if soul_effect_blue is None or soul_effect_red is None or soul_effect_green is None:
    print("魂抜けエフェクト画像が見つかりません。")
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
        current_time = time.time()
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
                h, w, _ = display_frame.shape
                target_center_x = target_x = random.randint(target_radius + target_amp_x, w - target_radius - target_amp_x - 1)
                target_center_y = target_y = random.randint(target_radius + target_amp_y, h - target_radius - target_amp_y - 1)
                state = "ready"
            elif key == 27:
                break
            continue


        # === 準備状態（カウントダウン） ===
        if state == "ready":
            pygame.mixer.music.load(bgm)     # 音楽ファイルの読み込み
            pygame.mixer.music.play(-1)
            countdown_start = time.time()
            countdown_duration = 5  # 5秒
            while True:
                current_time = time.time()
                elapsed = current_time - countdown_start
                remaining_sec = int(countdown_duration - elapsed) + 1

                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                bg_resized = cv2.resize(bg, (frame.shape[1], frame.shape[0]))
                display_frame = bg_resized.copy()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                # === 手の描画 ===
                if results.pose_landmarks:
                    right_wrist = results.pose_landmarks.landmark[15]
                    left_wrist = results.pose_landmarks.landmark[16]

                    rwx = int(right_wrist.x * display_frame.shape[1])
                    rwy = int(right_wrist.y * display_frame.shape[0])
                    lwx = int(left_wrist.x * display_frame.shape[1])
                    lwy = int(left_wrist.y * display_frame.shape[0])

                    # 右手：青，左手：赤
                    cv2.circle(display_frame, (rwx, rwy), 10, (0, 0, 255), -1)
                    cv2.circle(display_frame, (lwx, lwy), 10, (255, 0, 0), -1)

                # === カウントダウン数字描画 ===
                img_pil = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                if remaining_sec > 0:
                    draw_centered_text(draw, str(remaining_sec), font_title, 250, (255, 255, 255), frame.shape[1], shadow=True)
                else:
                    draw_centered_text(draw, "GO!", font_title, 250, (255, 100, 100), frame.shape[1], shadow=True)
                display_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                cv2.imshow("Pose Game", display_frame)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESCで中断
                    state = "start"
                    break

                # 5秒経過したらplayに移行
                if elapsed >= countdown_duration:
                    state = "play"
                    start_time = time.time()
                    pygame.mixer.music.play(-1)  # BGM開始
                    break
            continue

        # === ゲーム中 ===
        if state == "play":
            current_time = time.time()
            frame_counter += 1

            

            # --- お化けリストが空なら初期生成 ---
            if len(targets) == 0:
                h, w, _ = display_frame.shape
                for _ in range(max_targets):
                    tx, ty = generate_new_target(
                        random.randint(100, w-100),
                        random.randint(100, h-100),
                        target_radius, target_amp_x, target_amp_y, w, h
                    )
                    #15秒で的の出現変化
                    if int(time.time() - start_time) < 15:
                        ttype = 0 
                    else:
                        ttype = random.choice([0, 1, 2])
                    tcenterx, tcentery = tx, ty
                    targets.append({
                        "x": tx, "y": ty,
                        "center_x": tcenterx, "center_y": tcentery,
                        "type": ttype,
                        "hit": False,
                        "start_time": current_time
                    })

            # --- 各お化けを更新＆描画 ---
            for target in targets:
                # 動きの更新
                t = current_time * target_speed
                target["x"] = int(target["center_x"] + target_amp_x * math.cos(t + target["center_x"]/100))
                target["y"] = int(target["center_y"] + target_amp_y * math.sin(t + target["center_y"]/100))

                # 当たり判定（Pose）
                if results.pose_landmarks:
                    right_wrist = results.pose_landmarks.landmark[15]
                    left_wrist = results.pose_landmarks.landmark[16]

                    rwx = int(right_wrist.x * display_frame.shape[1])
                    rwy = int(right_wrist.y * display_frame.shape[0])
                    lwx = int(left_wrist.x * display_frame.shape[1])
                    lwy = int(left_wrist.y * display_frame.shape[0])

                    dist_r = ((rwx - target["x"])**2 + (rwy - target["y"])**2)**0.5
                    dist_l = ((lwx - target["x"])**2 + (lwy - target["y"])**2)**0.5

                    #右手首を描画 (青色の点)
                    cv2.circle(display_frame, (rwx, rwy), 10, (0, 0, 255), -1)
                    #左手首を描画 (赤色の点)
                    cv2.circle(display_frame, (lwx, lwy), 10, (255, 0, 0), -1)

                    #当たり判定
                    right_hit = (dist_r < target_radius)
                    left_hit = (dist_l < target_radius)

                    if not target["hit"]:
                        # === 当たり判定とスコア・コンボ処理 ===
                        now = time.time()

                        def apply_combo(base_point):
                            global combo, last_hit_time
                            # 最初のヒットならそのまま
                            if last_hit_time == 0:
                                combo = 1
                            # 2秒以内ならコンボ継続
                            elif now - last_hit_time <= 2:
                                combo += 1
                            # 2秒以上経過ならリセット
                            else:
                                combo = 0
                            last_hit_time = now

                            # 10コンボ以上なら+1点ボーナス
                            bonus = 1 if combo >= 10 else 0
                            return base_point + bonus

                        # コンボ対応スコア加算処理
                        if target["type"] == 0:
                            if right_hit or left_hit:
                                score += apply_combo(1)
                                se_hit.play()  # ヒット音再生
                                target["hit"] = True
                                hit_effects.append({"x": target["x"], "y": target["y"], "time": time.time()})
                                target["x"] = -9999
                                effect = 0
                        elif target["type"] == 1:
                            if right_hit:
                                score += apply_combo(1)
                                se_hit.play()  # ヒット音再生
                                target["hit"] = True
                                hit_effects.append({"x": target["x"], "y": target["y"], "time": time.time()})
                                target["x"] = -9999
                                effect = 1
                        elif target["type"] == 2:
                            if left_hit:
                                score += apply_combo(1)
                                se_hit.play()  # ヒット音再生
                                target["hit"] = True
                                hit_effects.append({"x": target["x"], "y": target["y"], "time": time.time()})
                                target["x"] = -9999
                                effect = 2

                # --- 的画像の描画 ---
                target_size = target_radius * 2
                if target["type"] == 0:
                    target_img = target_img_normal
                elif target["type"] == 1:
                    target_img = target_img_right
                else:
                    target_img = target_img_left
            
                target_img = cv2.resize(target_img, (target_size, target_size))

                x1, y1 = target["x"] - target_radius, target["y"] - target_radius
                x2, y2 = target["x"] + target_radius, target["y"] + target_radius

                x1_disp, y1_disp = max(0, x1), max(0, y1)
                x2_disp, y2_disp = min(display_frame.shape[1], x2), min(display_frame.shape[0], y2)

                if x2_disp > x1_disp and y2_disp > y1_disp:
                    crop = target_img[0:(y2_disp - y1_disp), 0:(x2_disp - x1_disp)]
                    if crop.shape[2] == 4:
                        alpha = crop[:, :, 3] / 255.0
                        for c in range(3):
                            display_frame[y1_disp:y2_disp, x1_disp:x2_disp, c] = (
                                alpha * crop[:, :, c] +
                                (1 - alpha) * display_frame[y1_disp:y2_disp, x1_disp:x2_disp, c]
                            )

            # --- 古いお化けを消し、新しいお化けを補充 ---
            targets = [t for t in targets if not t["hit"]]  # 当たったものを削除
            while len(targets) < max_targets:
                h, w, _ = display_frame.shape
                tx, ty = generate_new_target(
                    random.randint(100, w-100),
                    random.randint(100, h-100),
                    target_radius, target_amp_x, target_amp_y, w, h
                )
                #15秒で的の出現変化
                if int(time.time() - start_time) < 15:
                    ttype = 0 
                else:
                    ttype = random.choice([0, 1, 2])
                targets.append({
                    "x": tx, "y": ty,
                    "center_x": tx, "center_y": ty,
                    "type": ttype,
                    "hit": False,
                    "start_time": current_time
                })

            # --- 残り時間・スコア描画 ---
            elapsed = int(time.time() - start_time)
            remaining = max(0, game_time - elapsed)

            if combo > 0 and time.time() - last_hit_time > 2:
                combo = 0

            img_pil = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            draw.text((30, 30), f"TIME: {remaining}", font=font_text, fill=(255, 255, 255))
            draw.text((400, 30), f"SCORE: {score}", font=font_text, fill=(255, 255, 255))
            if combo > 0:
                draw.text((30, 80), f"COMBO: {combo}", font=font_text, fill=(255, 200, 0))
                if combo >= 10:
                    draw.text((30, 120), "BONUS ACTIVE!", font=font_small, fill=(255, 100, 100))
            display_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # --- 終了判定 ---
            if remaining <= 0:
                pygame.mixer.music.stop() 
                scores.append(score)
                with open(score_file, "w") as f:
                    for s in scores:
                        f.write(str(s) + "\n")
                state = "result"
                pygame.mixer.music.load(end_BGM)     # 音楽ファイルの読み込み
                pygame.mixer.music.play(-1)

            current_time = time.time()
            hit_effects = [e for e in hit_effects if current_time - e["time"] < 0.5]  # 0.5秒間表示

            for e in hit_effects:
                alpha = max(0, 1 - (current_time - e["time"]) / 0.5)  # 時間経過で透明化
                if effect == 0:
                    soul = soul_effect_green
                elif effect == 1:
                    soul = soul_effect_red
                else:
                    soul = soul_effect_blue

                target_img = cv2.resize(soul, (effect_size, effect_size))

                x1, y1 = e["x"] - target_radius, e["y"] - target_radius
                x2, y2 = e["x"] + target_radius, e["y"] + target_radius

                x1_disp, y1_disp = max(0, x1), max(0, y1)
                x2_disp, y2_disp = min(display_frame.shape[1], x2), min(display_frame.shape[0], y2)

                if x2_disp > x1_disp and y2_disp > y1_disp:
                    crop = target_img[0:(y2_disp - y1_disp), 0:(x2_disp - x1_disp)]
                    if crop.shape[2] == 4:
                        alpha_s = crop[:, :, 3] / 255.0
                        for c in range(3):
                            display_frame[y1_disp:y2_disp, x1_disp:x2_disp, c] = (
                                alpha_s * crop[:, :, c] + (1 - alpha_s) * display_frame[y1_disp:y2_disp, x1_disp:x2_disp, c]
                            )

        # === リザルト画面 ===
        elif state == "result":
            targets.clear()
            hit_effects.clear()
            result_img = cv2.imread("images/result_screen.jpg")
            if result_img is None:
                display_frame[:] = (0, 0, 0)
            else:
                result_resized = cv2.resize(result_img, (frame.shape[1], frame.shape[0]))
                 # ▼ここで明度を下げる（0.0〜1.0 で暗く、1.0 で元のまま）
                DIM_ALPHA = 0.55  # 明るさ係数（数字を小さくすると暗くなる）
                display_frame = cv2.convertScaleAbs(result_resized, alpha=DIM_ALPHA, beta=0)

                # さらに軽く暗幕を足してコントラスト維持（任意）
                overlay = np.zeros_like(display_frame)
                display_frame = cv2.addWeighted(display_frame, 1.0, overlay, 0.25, 0)

            # --- お化けの飾りを右下・左下に配置 ---
            obake_size = 180  # お化けの表示サイズ
            obake_right = cv2.resize(target_img_right, (obake_size, obake_size))
            obake_left = cv2.resize(target_img_left, (obake_size, obake_size))

            h, w, _ = display_frame.shape
            # 左下
            y1_l = h - obake_size - 20
            x1_l = 20
            # 右下
            y1_r = h - obake_size - 20
            x1_r = w - obake_size - 20

            # 透過対応描画関数
            def overlay_image(bg, fg, x, y):
                if fg.shape[2] == 4:  # αチャンネルあり
                    alpha = fg[:, :, 3] / 255.0
                    for c in range(3):
                        bg[y:y+fg.shape[0], x:x+fg.shape[1], c] = (
                            alpha * fg[:, :, c] + (1 - alpha) * bg[y:y+fg.shape[0], x:x+fg.shape[1], c]
                        )

            overlay_image(display_frame, obake_left, x1_l, y1_l)
            overlay_image(display_frame, obake_right, x1_r, y1_r)

            # --- テキスト描画 ---
            img_pil = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            # フォントを少し小さくして収まり良く
            font_title_small = ImageFont.truetype(font_path, 70)

            base_y = 80  # 文字の縦位置調整
            draw_centered_text(draw, "THANK YOU FOR PLAYING", font_title_small, base_y, (255, 0, 0), frame.shape[1])
            draw_centered_text(draw, f"YOUR SCORE: {score}", font_sub, base_y + 100, (255, 255, 255), frame.shape[1])

            # ランキング（上位3位のみ）
            ranking = sorted(scores, reverse=True)[:3]
            draw_centered_text(draw, "RANKING", font_sub, base_y + 220, (0, 255, 0), frame.shape[1])
            for i, s in enumerate(ranking):
                draw_centered_text(draw, f"{i+1}. {s}", font_small, base_y + 270 + i * 40, (255, 255, 255), frame.shape[1])

            draw_centered_text(draw, "Press SPACE to Restart or ESC to Quit", font_small, base_y + 460, (200, 200, 200), frame.shape[1])

            display_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            cv2.imshow("Pose Game", display_frame)

            key = cv2.waitKey(5) & 0xFF
            if key == 32:
                score = 0
                start_time = time.time()
                state = "start"
                pygame.mixer.music.stop()
            elif key == 27:
                break
            continue

        cv2.imshow("Pose Game", display_frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
