from typing import List, Tuple
import random
import math
import numpy as np

pi = math.pi

def generate_random(amount: int):
    noise_points = []
    for i in range(amount):
        noise_points.append([get_random_value(), get_random_value()])
    return noise_points

def add_gaussian_noise(dataset: Tuple[np.ndarray, np.ndarray], n_samples: int, n_classes: int, x_range: Tuple[float, float] = (-10, 10), y_range: Tuple[float, float] = (-10, 10)) -> None:
    dataset_X, dataset_y = dataset
    for i in range(n_samples):
        pt = get_random_point(x_range, y_range)
        dataset_X = np.append(dataset_X, [pt], axis=0)
        # Add noise points as new class
        dataset_y = np.append(dataset_y, [n_classes], axis=0)
    return (dataset_X, dataset_y)

def generate_horizontal_line_equal_dist(amount: int, y: int = 0, x_range: Tuple[float, float] = (-10, 10)):
    noise_points = []
    x_min, x_max = x_range
    interval_distance = (x_max - x_min) / (amount - 1)
    for i in range(amount):
        noise_points.append([i * interval_distance + x_min, y])
    return noise_points

def add_horizontal_line_noise(dataset: Tuple[np.ndarray, np.ndarray], n_samples: int, n_classes: int, x_range: Tuple[float, float] = (-10, 10)) -> Tuple[np.ndarray, np.ndarray]:
    dataset_x, dataset_y = dataset
    x_min, x_max = x_range
    interval_distance = (x_max - x_min) / (n_samples - 1)
    for i in range(n_samples):
        point = (i * interval_distance + x_min, 0)
        dataset_x = np.append(dataset_x, [point], axis=0)
        dataset_y = np.append(dataset_y, [n_classes], axis=0)
    return (dataset_x, dataset_y)


def generate_vertical_line_equal_dist(amount: int, x: int = 0):
    noise_points = []
    interval_distance = 2 / (amount - 1)
    for i in range(amount):
        noise_points.append([x, i * interval_distance - 1])
    return noise_points


def add_circle_noise(dataset: Tuple[np.ndarray, np.ndarray], n_samples: int, n_classes: int, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    dataset_x, dataset_y = dataset
    point_in_circle = generate_points_in_a_circle(n_samples, radius)
    for point in point_in_circle:
        dataset_x = np.append(dataset_x, [point], axis=0)
        dataset_y = np.append(dataset_y, [n_classes], axis=0)
    return (dataset_x, dataset_y)


def generate_points_in_a_circle(amount: int, radius: float = 5):
    return [(math.cos(2 * pi / amount * x) * radius, math.sin(2 * pi / amount * x) * radius) for x in range(0, amount + 1)]

def get_random_point(x_range: Tuple[float, float], y_range: Tuple[float, float]) -> List[float]:
    return [get_random_value(x_range), get_random_value(y_range)]

def get_random_value(range: Tuple[float, float] = (-10, 10)) -> float:
    return range[0] + random.random() * (range[1] - range[0])
