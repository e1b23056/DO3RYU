import cv2
import mediapipe as mp
import random
import time
import os

# MediaPipe Pose 初期化
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# スコア履歴をファイルから読み込み
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
game_time = 30   # ゲーム時間（秒）
score = 0

# 的の初期設定
target_radius = 50
frame_counter = 0
new_target_frame_interval = 25  # 25フレームごとに的を移動 (50→25)
hit_flag = False  # 的に当たった直後かどうかの判定
target_type = 0 #的の種別(0:加点,1:減点)
target_x = 100
target_y = 100

# 背景画像の読み込み
bg = cv2.imread("background.jpg")

if bg is None:
    print("背景画像が見つかりません。")
    exit()

# === ここでフルスクリーン設定 ===
cv2.namedWindow("Pose Game", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Pose Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

state = "start"
start_time = None

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

        #反転（こっちのがやりやすそう）
        frame = cv2.flip(frame, 1)

        # 背景をカメラサイズにリサイズ
        bg_resized = cv2.resize(bg, (frame.shape[1], frame.shape[0]))
        display_frame = bg_resized.copy()

        # カメラ映像で骨格検出（背景には描画しない）
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # === スタート画面 ===
        if state == "start":
            # スタート画像の読み込み（最初に一度だけ）
            if 'start_img' not in locals():
                start_img = cv2.imread("start_screen.jpg")
                if start_img is None:
                    print("スタート画面画像が見つかりません。")
                    exit()

            # 画面サイズに合わせてリサイズ
            start_resized = cv2.resize(start_img, (frame.shape[1], frame.shape[0]))

            # 表示
            display_frame = start_resized.copy()
            cv2.putText(display_frame, "Press SPACE to Start", (150, 600),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            cv2.imshow("Pose Game", display_frame)
            key = cv2.waitKey(5) & 0xFF
            if key == 32:  # SPACE
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
                target_type = random.choice([0, 1]) #的の抽選
                target_x = random.randint(target_radius, w - target_radius - 1)
                target_y = random.randint(target_radius, h - target_radius - 1)
                frame_counter = 0

            # Pose骨格の描画 ＋ 手首座標の取得
            right_wrist_x = right_wrist_y = None
            left_wrist_x = left_wrist_y = None

            #カメラで人が認識できたら
            if results.pose_landmarks: 
                # 手首の座標を取得
                right_wrist = results.pose_landmarks.landmark[15]
                left_wrist = results.pose_landmarks.landmark[16]

                right_wrist_x = int(right_wrist.x * display_frame.shape[1])
                right_wrist_y = int(right_wrist.y * display_frame.shape[0])
                left_wrist_x = int(left_wrist.x * display_frame.shape[1])
                left_wrist_y = int(left_wrist.y * display_frame.shape[0])

                #右手首を描画 (緑色の点)
                cv2.circle(display_frame, (right_wrist_x, right_wrist_y), 10, (0, 255, 0), -1)
                #左手首を描画 (緑色の点)
                cv2.circle(display_frame, (left_wrist_x, left_wrist_y), 10, (0, 255, 0), -1)

                
                # --- 的との当たり判定 ---
                distance_right = ((right_wrist_x - target_x) ** 2 + (right_wrist_y - target_y) ** 2) ** 0.5
                distance_left = ((left_wrist_x - target_x) ** 2 + (left_wrist_y - target_y) ** 2) ** 0.5

                if (distance_right < target_radius or distance_left < target_radius) and not hit_flag:
                    if target_type == 0:
                        score += 1
                    elif target_type == 1:
                        score -= 1
                    hit_flag = True
                    #的を外に
                    target_x = -100
                    frame_counter = 0
                elif distance_right >= target_radius and distance_left >= target_radius:
                    hit_flag = False

            # 的を描画
            if target_type == 0:
                cv2.circle(display_frame, (target_x, target_y), target_radius, (0, 0, 255), -1) # 赤 = 加点
            elif target_type == 1:
                cv2.circle(display_frame, (target_x, target_y), target_radius, (255, 0, 0), -1) # 青 = 減点 


            # TIMEとSCORE表示
            elapsed = int(time.time() - start_time)
            remaining = max(0, game_time - elapsed)
            cv2.putText(display_frame, f"TIME:{remaining}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(display_frame, f"SCORE:{score}", (400, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            if remaining <= 0:
                # スコアを保存
                scores.append(score)
                with open(score_file, "w") as f:
                    for s in scores:
                        f.write(str(s) + "\n")
                state = "result"

        # === リザルト画面 ===
        elif state == "result":
            display_frame[:] = (0, 0, 0)
            cv2.putText(display_frame, "GAME OVER", (200, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.putText(display_frame, f"Your Score: {score}", (200, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # ランキング表示
            ranking = sorted(scores, reverse=True)[:5]
            cv2.putText(display_frame, "RANKING:", (200, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            for i, s in enumerate(ranking):
                cv2.putText(display_frame, f"{i+1}. {s}", (220, 320 + i*40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.putText(display_frame, "Press SPACE to Restart or ESC to Quit", (50, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            cv2.imshow("Pose Game", display_frame)
            key = cv2.waitKey(5) & 0xFF
            if key == 32:  # リスタート
                score = 0
                start_time = time.time()
                state = "play"
            elif key == 27:  # 終了
                break
            continue

        cv2.imshow("Pose Game", display_frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()