import cv2
import mediapipe as mp
import random
import time

# MediaPipe Pose 初期化
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ゲーム設定
start_time = time.time()
game_time = 30   # ゲーム時間（秒）
score = 0

# 的の初期設定
target_radius = 50
frame_counter = 0
new_target_frame_interval = 25  # 25フレームごとに的を移動 (50→25)
hit_flag = False  # 的に当たった直後かどうかの判定

# 加点的（赤）
target_plus_x = 100
target_plus_y = 100
hit_plus_flag = False

# 減点的（青）
target_minus_x = 300
target_minus_y = 200
hit_minus_flag = False

# 背景画像の読み込み
bg = cv2.imread("background.jpg")

if bg is None:
    print("背景画像が見つかりません。")
    exit()

# === ここでフルスクリーン設定 ===
cv2.namedWindow("Pose Game", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Pose Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


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

        frame_counter += 1
        if frame_counter % new_target_frame_interval == 0:
            h, w, _ = display_frame.shape
            target_plus_x = random.randint(target_radius, w - target_radius - 1)
            target_plus_y = random.randint(target_radius, h - target_radius - 1)
            target_minus_x = random.randint(target_radius, w - target_radius - 1)
            target_minus_y = random.randint(target_radius, h - target_radius - 1)
            frame_counter = 0
        
        if target_plus_x < 0:
            # 加点的の位置
            target_plus_x = random.randint(target_radius, w - target_radius - 1)
            target_plus_y = random.randint(target_radius, h - target_radius - 1)
            
        if target_minus_x < 0:
            # 減点的の位置 ← 追加
            target_minus_x = random.randint(target_radius, w - target_radius - 1)
            target_minus_y = random.randint(target_radius, h - target_radius - 1)

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

            # --- 加点的との当たり判定 ---
            distance_right_plus = ((right_wrist_x - target_plus_x) ** 2 + (right_wrist_y - target_plus_y) ** 2) ** 0.5
            distance_left_plus = ((left_wrist_x - target_plus_x) ** 2 + (left_wrist_y - target_plus_y) ** 2) ** 0.5

            if (distance_right_plus < target_radius or distance_left_plus < target_radius) and not hit_plus_flag:
                score += 1
                hit_plus_flag = True
                #的を外に
                target_plus_x = -100
                frame_counter = 0
            elif distance_right_plus >= target_radius and distance_left_plus >= target_radius:
                hit_plus_flag = False

            # --- 減点的との当たり判定 ---
            distance_right_minus = ((right_wrist_x - target_minus_x) ** 2 + (right_wrist_y - target_minus_y) ** 2) ** 0.5
            distance_left_minus = ((left_wrist_x - target_minus_x) ** 2 + (left_wrist_y - target_minus_y) ** 2) ** 0.5

            if (distance_right_minus < target_radius or distance_left_minus < target_radius) and not hit_minus_flag:
                score -= 1
                hit_minus_flag = True
                #的を外に
                target_minus_x = -100
                frame_counter = 0
            elif distance_right_minus >= target_radius and distance_left_minus >= target_radius:
                hit_minus_flag = False

        # 的を描画
        cv2.circle(display_frame, (target_plus_x, target_plus_y), target_radius, (0, 0, 255), -1)   # 赤 = 加点
        cv2.circle(display_frame, (target_minus_x, target_minus_y), target_radius, (255, 0, 0), -1) # 青 = 減点 ← 追加

        # TIMEとSCORE表示
        elapsed = int(time.time() - start_time)
        remaining = max(0, game_time - elapsed)
        cv2.putText(display_frame, f"TIME:{remaining}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(display_frame, f"SCORE:{score}", (400, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # 画面表示
        cv2.imshow("Pose Game", display_frame)

        # TIMEが0になったら終了
        if remaining <= 0:
            print("Game Over! Final Score:", score)
            break

        if cv2.waitKey(5) & 0xFF == 27:  # ESCで終了
            break

cap.release()
cv2.destroyAllWindows()
