import cv2
import numpy as np


class Visualizer:
    """
    Вся отрисовка на кадре: трек, точки, векторы, HUD.
    """

    def __init__(self, frame_w: int, frame_h: int, roi_size: int = 120):
        self.frame_w  = frame_w
        self.frame_h  = frame_h
        self.roi_size = roi_size

        self._trail: list = []
        self._max_trail   = 50


    def update_trail(self, point: tuple):
        self._trail.append(point)
        if len(self._trail) > self._max_trail:
            self._trail.pop(0)

    def reset_trail(self):
        self._trail.clear()


    def draw(self, frame, cx_raw, cy_raw, cx_filt, cy_filt,
             flow_points, vx, vy, err_m_x, err_m_y):
        """Рисует всё на кадре."""
        self._draw_center_cross(frame)
        self._draw_flow_points(frame, flow_points)
        self._draw_roi(frame, cx_raw, cy_raw)
        self._draw_raw_position(frame, cx_raw, cy_raw)
        self._draw_filtered_position(frame, cx_filt, cy_filt)
        self._draw_trail(frame)
        self._draw_velocity_vector(frame, cx_filt, cy_filt, vx, vy)
        self._draw_hud(frame, cx_filt, cy_filt, vx, vy, err_m_x, err_m_y)

    def draw_predict(self, frame, cx_pred, cy_pred):
        """Рисует предсказанную позицию когда маркер не найден."""
        cv2.circle(frame, (int(cx_pred), int(cy_pred)), 10, (128, 128, 0), 1)
        cv2.putText(frame, "PREDICT", (int(cx_pred) + 12, int(cy_pred)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 0), 1)

    def draw_status(self, frame, text: str, color=(100, 100, 100)):
        cv2.putText(frame, text, (10, self.frame_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    def _draw_center_cross(self, frame):
        cx, cy = self.frame_w // 2, self.frame_h // 2
        cv2.drawMarker(frame, (cx, cy), (150, 150, 150),
                       cv2.MARKER_CROSS, 20, 1)

    def _draw_flow_points(self, frame, pts):
        if pts is None:
            return
        for p in pts:
            cv2.circle(frame, (int(p[0]), int(p[1])), 2, (255, 100, 0), -1)

    def _draw_roi(self, frame, cx, cy):
        half = self.roi_size // 2
        cv2.rectangle(frame,
                      (cx - half, cy - half),
                      (cx + half, cy + half),
                      (100, 100, 255), 1)

    def _draw_raw_position(self, frame, cx, cy):
        """Красная точка — сырой optical flow."""
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    def _draw_filtered_position(self, frame, cx, cy):
        """Жёлтая окружность — позиция после α-β фильтра."""
        pos = (int(cx), int(cy))
        cv2.circle(frame, pos, 8, (0, 220, 220), 2)
        self.update_trail(pos)

    def _draw_trail(self, frame):
        for i in range(1, len(self._trail)):
            cv2.line(frame, self._trail[i - 1], self._trail[i],
                     (0, 255, 0), 2)

    def _draw_velocity_vector(self, frame, cx, cy, vx, vy):
        scale = 30
        start = (int(cx), int(cy))
        end   = (int(cx + vx * scale), int(cy + vy * scale))
        cv2.arrowedLine(frame, start, end, (0, 0, 255), 2, tipLength=0.3)

    def _draw_hud(self, frame, cx, cy, vx, vy, err_m_x, err_m_y):
        lines = [
            f"Err px : ({cx - self.frame_w//2:+.0f}, {cy - self.frame_h//2:+.0f})",
            f"Err  m : ({err_m_x:+.3f}, {err_m_y:+.3f})",
            f"Cmd v  : vx={vx:+.3f}  vy={vy:+.3f}  m/s",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
