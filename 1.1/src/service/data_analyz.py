class DataAnalyzer:
    def __init__(self, data: list[tuple[float, float]]):
        self.points: list[tuple[float, float]] = data
        self.len: int = len(self.points)
        self.x_col: int = 0
        self.y_col: int = 1
        self._column_selection()

    def _column_selection(self) -> None:
        while True:
            try:
                choice: int = int(input(f"Установить первый столбец как X? 1 - Подтвердить, 0 - Выбрать начальный столбец как Y. (по умолчанию X -  {self.x_col})\n"))
                if choice in (0, 1):
                    self.x_col = int(choice)
                    self.y_col = 1 - self.x_col
                    break
            except ValueError:
                print("Номер столбца должен быть целым числом")

    @property
    def x_values(self) -> list[float]:
        return [point[self.x_col] for point in self.points]

    @property
    def y_values(self) -> list[float]:
        return [point[self.y_col] for point in self.points]

    @property
    def mean_values(self) -> tuple[float, float]:
        sum_x = sum(self.x_values)
        sum_y = sum(self.y_values)
        return sum_x / self.len, sum_y / self.len

    @property
    def min_values(self) -> tuple[float, float]:
        min_x = min(self.x_values)
        min_y = min(self.y_values)
        return min_x, min_y

    @property
    def max_values(self) -> tuple[float, float]:
        max_x = max(self.x_values)
        max_y = max(self.y_values)
        return max_x, max_y

    @property
    def linear_regression(self) -> tuple[float, float]:
        x = self.x_values
        y = self.y_values
        x_mean, y_mean = self.mean_values

        cov = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        ss_x = sum((xi - x_mean)**2 for xi in x)

        b_1 = cov / ss_x
        b_0 = y_mean - b_1 * x_mean

        return b_0, b_1

    def print_statistics(self) -> None:
        if not self.points:
            print('Нет данных для анализа!')
            return

        av_x, av_y = self.mean_values
        min_x, min_y = self.min_values
        max_x, max_y = self.max_values

        print("\nСтатистика данных:")
        print(f"Количество точек: {self.len}")
        print(f"X: min={min_x:.2f}, max={max_x:.2f}, avg={av_x:.2f}")
        print(f"Y: min={min_y:.2f}, max={max_y:.2f}, avg={av_y:.2f}")
