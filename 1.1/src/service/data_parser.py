def get_points(file_path: str) -> list[tuple[float, float]]:
    data: list[tuple[float, float]] = []
    with open(file_path, "r") as f:
        for line in f:
            hours, score = line.strip().split(",")
            try:
                data.append((float(hours), float(score)))
            except ValueError:
                continue

    return data
