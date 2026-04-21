import cv2
import numpy as np


class OpticalFlowTracker:
    """
    Отслеживает маркер через оптический поток (Lucas-Kanade).

    Возможности:
      - Захват точек goodFeaturesToTrack в ROI вокруг маркера
      - Отбор выбросов по расстоянию от центра облака
      - Проверка шаблона (защита от ухода на другой объект)
      - Адаптивное обновление шаблона
      - Motion gating — отклонение резких скачков позиции
    """

    def __init__(self,
                 frame_w:    int   = 640,
                 frame_h:    int   = 480,
                 roi_size:   int   = 120,
                 min_points: int   = 10,
                 max_points: int   = 80,
                 max_jump:   int   = 60,
                 adapt_rate: float = 0.1):

        self.frame_w    = frame_w
        self.frame_h    = frame_h
        self.roi_size   = roi_size
        self.min_points = min_points
        self.max_points = max_points
        self.max_jump   = max_jump
        self.adapt_rate = adapt_rate

        self.points      = None
        self.prev_gray   = None
        self.template    = None
        self.prev_center = None
        self.tracking    = False


    def capture(self, frame) -> bool:
        """
        Захват маркера в центре кадра.
        Вызывается по нажатию 'c'.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cx   = self.frame_w // 2
        cy   = self.frame_h // 2

        points = self._generate_points(gray, cx, cy)
        if points is None:
            print("[Tracker] Нет точек для захвата")
            return False

        self.points      = points
        self.prev_gray   = gray
        self.prev_center = (cx, cy)
        self.template    = self._crop(gray, cx, cy)
        self.tracking    = True
        print("[Tracker] Трекинг запущен")
        return True

    def update(self, frame):
        """
        Обновление на новом кадре.
        Возвращает (cx, cy, points) или None если маркер потерян.
        """
        if not self.tracking or self.points is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.points, None
        )
        if next_pts is None:
            self._lose()
            return None

        good = next_pts[status == 1]
        if len(good) == 0:
            self._lose()
            return None

        raw_cx = int(np.mean(good[:, 0]))
        raw_cy = int(np.mean(good[:, 1]))
        good   = self._reject_outliers(good, raw_cx, raw_cy)

        if len(good) < self.min_points:
            new_pts = self._generate_points(gray, raw_cx, raw_cy)
            if new_pts is None:
                self._lose()
                return None
            self.points = new_pts
        else:
            self.points = good.reshape(-1, 1, 2)

        self.prev_gray = gray

        cx = int(np.mean(self.points[:, 0, 0]))
        cy = int(np.mean(self.points[:, 0, 1]))

        # motion gating
        if self.prev_center is not None:
            dist = np.linalg.norm(np.array([cx, cy]) - np.array(self.prev_center))
            if dist > self.max_jump:
                return None

        if not self._check_template(gray, cx, cy):
            print("[Tracker] Объект изменился")
            return None

        self.prev_center = (cx, cy)
        return cx, cy, self.points.reshape(-1, 2)

    def reset(self):
        self.points      = None
        self.prev_gray   = None
        self.template    = None
        self.prev_center = None
        self.tracking    = False
        print("[Tracker] Сброс")


    def _generate_points(self, gray, cx, cy):
        half = self.roi_size // 2
        x1 = max(cx - half, 0);  x2 = min(cx + half, self.frame_w)
        y1 = max(cy - half, 0);  y2 = min(cy + half, self.frame_h)
        roi = gray[y1:y2, x1:x2]

        pts = cv2.goodFeaturesToTrack(
            roi, maxCorners=self.max_points,
            qualityLevel=0.01, minDistance=5
        )
        if pts is None:
            return None

        pts[:, 0, 0] += x1
        pts[:, 0, 1] += y1
        return pts

    def _reject_outliers(self, pts, cx, cy, max_dist: int = 50):
        filtered = [p for p in pts
                    if np.linalg.norm(p - np.array([cx, cy])) < max_dist]
        return np.array(filtered) if filtered else pts

    def _crop(self, gray, cx, cy):
        half = self.roi_size // 2
        return gray[
            max(cy - half, 0):min(cy + half, self.frame_h),
            max(cx - half, 0):min(cx + half, self.frame_w)
        ].copy()

    def _check_template(self, gray, cx, cy) -> bool:
        if self.template is None:
            return True
        current = self._crop(gray, cx, cy)
        if current.shape != self.template.shape:
            return False
        score = np.mean(cv2.absdiff(self.template, current))
        if score < 40:
            self.template = cv2.addWeighted(
                self.template, 1 - self.adapt_rate,
                current,       self.adapt_rate, 0
            )
            return True
        return False

    def _lose(self):
        self.tracking = False
