from typing import List, Tuple
import random
import math
pi = math.pi

def generate_random(amount: int):
    noise_points = []
    for i in range(amount):
        noise_points.append([get_random_value(), get_random_value()])
    return noise_points


def generate_horizontal_line_equal_dist(amount: int, y: int = 0):
    noise_points = []
    interval_distance = 20 / (amount - 1)
    for i in range(amount):
        noise_points.append([i * interval_distance - 10, y])
    return noise_points


def generate_vertical_line_equal_dist(amount: int, x: int = 0):
    noise_points = []
    interval_distance = 2 / (amount - 1)
    for i in range(amount):
        noise_points.append([x, i * interval_distance - 1])
    return noise_points


def generate_points_in_a_circle(amount: int, radius = 5):
    return [(math.cos(2 * pi / amount * x) * radius, math.sin(2 * pi / amount * x) * radius) for x in range(0, amount + 1)]


def get_random_value() -> float:
    return random.random() * 20 - 10
