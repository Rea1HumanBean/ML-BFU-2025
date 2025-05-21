import os

from service.data_parser import get_points
from service.data_analyz import DataAnalyzer
from service.image_points import plot_comparison


data: list[tuple[float, float]] = get_points(file_path=os.path.join("..", "data", "scores.csv"))
points: DataAnalyzer = DataAnalyzer(data)


points.print_statistics()
plot_comparison(points)
