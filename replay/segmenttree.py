def Min(x, y):
    if x < y:
        return x
    return y


def Max(x, y):
    if x < y:
        return y
    return x


class SegmentTree:
    def __init__(self, siz):
        self.siz = siz
        self.sum = [0] * siz * 4
        self.min = [0] * siz * 4
        self.max = [0] * siz * 4
        self.num = [0] * siz
        self.data = [None] * siz
        self.cur = self.count = 0
        self.build(1, 0, siz - 1)

    def get_max(self):
        return self.max[1]

    def get_min(self):
        return self.min[1]

    def get_total(self):
        return self.sum[1]

    def update(self, k):
        self.sum[k] = self.sum[k << 1] + self.sum[(k << 1) | 1]
        self.min[k] = Min(self.min[k << 1], self.min[(k << 1) | 1])
        self.max[k] = Max(self.max[k << 1], self.max[(k << 1) | 1])

    def build(self, k, l, r):
        if l == r:
            self.sum[k] = 0
            self.min[k] = 1e9
            self.max[k] = 0
            self.num[l] = k
            return
        mid = (l + r) >> 1
        self.build(k << 1, l, mid)
        self.build((k << 1) | 1, mid + 1, r)
        self.update(k)

    def modify(self, x, w):
        cur = self.num[x]
        self.min[cur] = self.sum[cur] = self.max[cur] = w
        cur >>= 1
        while cur > 0:
            self.update(cur)
            cur >>= 1
        # print(f'modify({k}, {l}, {r}, {x}, {w})')
        # print(f'update({k})')
        # print(self.sum[k])

    def query(self, value):
        cur = 1
        l = 0
        r = self.siz - 1
        while l != r:
            mid = (l + r) >> 1
            lc = cur << 1
            rc = lc + 1
            if value <= self.sum[lc]:
                cur = lc
                r = mid
            else:
                value -= self.sum[lc]
                cur = rc
                l = mid + 1
        return l, self.sum[cur] / self.get_total()

    def ins(self, data):
        self.data[self.cur] = data
        # print(self.get_max())
        if self.count == 0:
            self.modify(self.cur, 1)
        else:
            self.modify(self.cur, self.get_max())
        if self.count < self.siz:
            self.count += 1
        # assert self.count == self.get_total(), "insertion failed"
        self.cur = (self.cur + 1) % self.siz

    def recover(self, x, w):
        self.modify(x, w)

    def sample(self, value):
        # print(f'value : {value}, sum : {self.get_total()}')
        idx, prob = self.query(value)
        return idx, self.data[idx], prob
