import numpy as np


class AlphaBetaFilter:
    """
    Альфа-бета фильтр для 2D позиции.

    Вектор состояния: [x, vx, y, vy]

    Шаг 1 — Предсказание по кинематике:
        x_pred = x + vx * dt

    Шаг 2 — Коррекция по измерению:
        x  = x_pred + alpha * (x_meas - x_pred)
        vx = vx     + (beta / dt) * (x_meas - x_pred)

    Коэффициенты выбираются через частоту среза f_cut:
        объект не может двигаться быстрее f_cut Гц.
    """

    def __init__(self, f_cut: float = 1.5, dt: float = 0.033):
        """
        f_cut — частота среза [Гц].
                Меньше → сильнее фильтрация шума, медленнее отклик.
                Рекомендуется 1.0–2.0 Гц для удержания точки.
        dt    — период кадра [с].
        """
        self.dt = dt
        self.alpha, self.beta = self._from_frequency(f_cut, dt)

        self.x  = 0.0
        self.y  = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.initialized = False

    def update(self, x_meas: float, y_meas: float):
        """
        Обновление по измерению.
        Возвращает (x, y, vx, vy) — сглаженная позиция и оценка скорости.
        """
        if not self.initialized:
            self.x, self.y   = x_meas, y_meas
            self.initialized = True
            return self.x, self.y, 0.0, 0.0

        x_pred = self.x + self.vx * self.dt
        y_pred = self.y + self.vy * self.dt

        rx = x_meas - x_pred
        ry = y_meas - y_pred

        self.x  = x_pred + self.alpha * rx
        self.vx = self.vx + (self.beta / self.dt) * rx
        self.y  = y_pred + self.alpha * ry
        self.vy = self.vy + (self.beta / self.dt) * ry

        return self.x, self.y, self.vx, self.vy

    def predict_only(self):
        """
        Шаг без измерения — маркер временно не найден.
        Фильтр продолжает предсказывать позицию по инерции.
        """
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt
        return self.x, self.y

    def reset(self):
        self.x  = self.y  = 0.0
        self.vx = self.vy = 0.0
        self.initialized  = False

    @staticmethod
    def _from_frequency(f_cut: float, dt: float):
        wc    = 2 * np.pi * f_cut
        alpha = 1 - np.exp(-wc * dt)
        beta  = alpha ** 2 / (2 - alpha)
        return alpha, beta
