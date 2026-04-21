class PID:
    """
    PID-регулятор для одной оси.
    Используется отдельно для X и Y.
    """

    def __init__(self, Kp=0.6, Ki=0.05, Kd=0.15, dt=0.033):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt

        self._integral   = 0.0
        self._prev_error = 0.0

    def update(self, error: float) -> float:
        self._integral  += error * self.dt
        derivative       = (error - self._prev_error) / self.dt
        output           = self.Kp * error + self.Ki * self._integral + self.Kd * derivative
        self._prev_error = error
        return output

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0
