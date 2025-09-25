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
target_center_x = 100
target_center_y = 100
target_radius = 50
frame_counter = 0
new_target_frame_interval = 50  # 50フレームごとに的を移動
hit_flag = False  # 的に当たった直後かどうかの判定

# 背景画像の読み込み
bg = cv2.imread(r"C:\oit\home\py24\zemi\background.jpg")

if bg is None:
    print("背景画像が見つかりません。")
    exit()

# === ここでフルスクリーン設定 ===
cv2.namedWindow("Pose Game", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Pose Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# === ここでフルスクリーン設定 ===
cv2.namedWindow("Pose Game", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Pose Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 背景をカメラサイズにリサイズ
        bg_resized = cv2.resize(bg, (frame.shape[1], frame.shape[0]))
        display_frame = bg_resized.copy()

        # カメラ映像で骨格検出（背景には描画しない）
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        frame_counter += 1
        if frame_counter % new_target_frame_interval == 0:
            # 新しい的の座標をランダム生成
            h, w, _ = display_frame.shape
            target_center_x = random.randint(target_radius, w - target_radius - 1)
            target_center_y = random.randint(target_radius, h - target_radius - 1)

        # Pose骨格の描画 ＋ 手首座標の取得
        right_wrist_x = right_wrist_y = None
        left_wrist_x = left_wrist_y = None

        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                display_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # 手首の座標を取得
            right_wrist = results.pose_landmarks.landmark[15]
            left_wrist = results.pose_landmarks.landmark[16]

            right_wrist_x = int(right_wrist.x * display_frame.shape[1])
            right_wrist_y = int(right_wrist.y * display_frame.shape[0])
            left_wrist_x = int(left_wrist.x * display_frame.shape[1])
            left_wrist_y = int(left_wrist.y * display_frame.shape[0])

        # 当たり判定（両手と的の距離）
        if right_wrist_x is not None and left_wrist_x is not None:
            distance_right = ((right_wrist_x - target_center_x) ** 2 + (right_wrist_y - target_center_y) ** 2) ** 0.5
            distance_left = ((left_wrist_x - target_center_x) ** 2 + (left_wrist_y - target_center_y) ** 2) ** 0.5

            if (distance_right < target_radius or distance_left < target_radius) and not hit_flag:
                score += 1
                hit_flag = True
            elif distance_right >= target_radius and distance_left >= target_radius:
                hit_flag = False

        # 的を描画
        cv2.circle(display_frame, (target_center_x, target_center_y), target_radius, (0, 0, 255), -1)

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
