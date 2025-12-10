import numpy as np
from typing import Tuple

Pos = Tuple[int, int]

W = 0
H = 0

def init(w,h):
    global H,W
    H= h
    W= w
    

def clip_pos(pos) -> Pos:
    x, y = pos
    x = max(0, min(W-1, x))
    y = max(0, min(H-1, y))
    return x, y


def midpoint(pos1, pos2) -> Pos:
    x1, y1 = pos1
    x2, y2 = pos2
    return (x1+x2)//2, (y1+y2)//2


def opposite(origin, pos) -> Pos:
    x1, y1 = origin
    x2, y2 = pos
    return 2*x1-x2, 2*y1-y2


def distance(pos1, pos2) -> int:
    x1, y1 = pos1
    x2, y2 = pos2
    dx = x2 - x1
    dy = y2 - y1
    return np.hypot(dx, dy)


def square(pos, size, w=None, h=None):
    w = W if w is None else w
    h = H if h is None else h
    x, y = pos
    radius = size #(size-1)//2
    x_min = max(0, x-radius)
    x_max = min(w-1, x_min+2*radius)
    y_min = max(0, y-radius)
    y_max = min(h-1, y_min+2*radius)
    return x_min, x_max, y_min, y_max


def random_pos(pos=None, radius=None) -> Pos:
    if pos is not None and radius is not None:
        x_min, x_max, y_min, y_max = square(pos, radius)
    else:
        x_min, x_max, y_min, y_max = 0, W, 0, H
    x = np.random.randint(x_min, x_max)
    y = np.random.randint(y_min, y_max)
    return x, y