import numpy as np


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.empty(capacity, dtype=object)
        self.size = 0
        self.curr_point = 0

    def add(self, data):
        self.data[self.curr_point] = data
        self.update(self.curr_point, max(self.tree[self.capacity - 1:self.capacity + self.size]))
        self.curr_point += 1
        if self.curr_point >= self.capacity:
            self.curr_point = 0
        if self.size < self.capacity:
            self.size += 1

    def update(self, point, weight):
        if abs(weight) < 1e-4:
            weight = 1e-4
        idx = point + self.capacity - 1
        change = weight - self.tree[idx]
        self.tree[idx] = weight
        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            parent = (parent - 1) // 2

    def get_total(self):
        return self.tree[0]

    def get_min(self):
        return min(self.tree[self.capacity - 1:self.capacity + self.size - 1])

    def sample(self, v):
        idx = 0
        while idx < self.capacity - 1:
            l_idx = idx * 2 + 1
            r_idx = l_idx + 1
            if self.tree[l_idx] >= v:
                idx = l_idx
            else:
                idx = r_idx
                v = v - self.tree[l_idx]

        point = idx - (self.capacity - 1)
        return point, self.data[point], self.tree[idx] / self.get_total()
