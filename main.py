"""
main.py — тест трекера на веб-камере компьютера.

Управление:
  c — захватить центр кадра как точку удержания
  r — сброс
  q — выход

MAVLink отключён. Вместо реальной высоты используется
константа SIMULATED_ALTITUDE_M.
Команды vx, vy только выводятся в HUD (никуда не отправляются).
"""

import cv2
import time
import numpy as np

from optical_flow_tracker import OpticalFlowTracker
from alpha_beta_filter    import AlphaBetaFilter
from pid                  import PID
from visualizer           import Visualizer




CAMERA_INDEX          = 0       # индекс камеры
SIMULATED_ALTITUDE_M  = 5.0     # [м] — заменить на реальную высоту с MAVLink
FOV_H_DEG             = 90.0    # горизонтальный угол обзора камеры [°]
F_CUT_HZ              = 1.5     # частота среза α-β фильтра [Гц]
PID_KP                = 0.6
PID_KI                = 0.05
PID_KD                = 0.15
MAX_SPEED_MS          = 1.0     # ограничение команды скорости [м/с]


def pixels_to_meters(err_px_x, err_px_y, altitude, fx, fy):
    """Перевод ошибки в пикселях → метры через высоту и FOV."""
    err_m_x = (err_px_x / fx) * altitude
    err_m_y = (err_px_y / fy) * altitude
    return err_m_x, err_m_y


def compute_fx_fy(frame_w, frame_h, fov_h_deg):
    """Фокусное расстояние в пикселях из угла обзора (pinhole модель)."""
    fov_h = np.radians(fov_h_deg)
    fx    = (frame_w / 2) / np.tan(fov_h / 2)
    fy    = fx   # квадратные пиксели
    return fx, fy


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Камера не найдена")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Камера: {frame_w}x{frame_h}")

    fx, fy = compute_fx_fy(frame_w, frame_h, FOV_H_DEG)
    dt     = 1 / 30   # начальное значение, уточняется по реальному времени

    tracker   = OpticalFlowTracker(frame_w, frame_h)
    ab_filter = AlphaBetaFilter(f_cut=F_CUT_HZ, dt=dt)
    pid_x     = PID(Kp=PID_KP, Ki=PID_KI, Kd=PID_KD, dt=dt)
    pid_y     = PID(Kp=PID_KP, Ki=PID_KI, Kd=PID_KD, dt=dt)
    vis       = Visualizer(frame_w, frame_h)

    prev_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now       = time.perf_counter()
        dt        = max(now - prev_time, 1e-3)
        prev_time = now
        ab_filter.dt = dt
        pid_x.dt     = dt
        pid_y.dt     = dt

        result = tracker.update(frame)

        if result is not None:
            cx_raw, cy_raw, flow_pts = result

            err_px_x = cx_raw - frame_w // 2
            err_px_y = cy_raw - frame_h // 2

            err_m_x, err_m_y = pixels_to_meters(
                err_px_x, err_px_y, SIMULATED_ALTITUDE_M, fx, fy
            )

            fx_f, fy_f, vx_est, vy_est = ab_filter.update(err_m_x, err_m_y)

            vx_cmd = float(np.clip(-pid_x.update(fx_f), -MAX_SPEED_MS, MAX_SPEED_MS))
            vy_cmd = float(np.clip(-pid_y.update(fy_f), -MAX_SPEED_MS, MAX_SPEED_MS))

            cx_filt = int(frame_w // 2 + fx_f / SIMULATED_ALTITUDE_M * fx)
            cy_filt = int(frame_h // 2 + fy_f / SIMULATED_ALTITUDE_M * fy)

            vis.draw(frame,
                     cx_raw, cy_raw,
                     cx_filt, cy_filt,
                     flow_pts,
                     vx_est, vy_est,
                     fx_f, fy_f)

            vis.draw_status(frame,
                            f"TRACKING  vx_cmd={vx_cmd:+.3f}  vy_cmd={vy_cmd:+.3f} m/s",
                            color=(0, 220, 0))

        elif tracker.tracking:
            px, py = ab_filter.predict_only()
            cx_pred = int(frame_w // 2 + px / SIMULATED_ALTITUDE_M * fx)
            cy_pred = int(frame_h // 2 + py / SIMULATED_ALTITUDE_M * fy)
            vis.draw_predict(frame, cx_pred, cy_pred)
            vis.draw_status(frame, "PREDICT (маркер не найден)", color=(0, 180, 220))

        else:
            vis.draw_status(frame, "Нажмите 'c' для захвата")

        cv2.imshow("Drone Tracker (webcam test)", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('c'):
            tracker.capture(frame)
            ab_filter.reset()
            pid_x.reset()
            pid_y.reset()
            vis.reset_trail()

        elif key == ord('r'):
            tracker.reset()
            ab_filter.reset()
            pid_x.reset()
            pid_y.reset()
            vis.reset_trail()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
