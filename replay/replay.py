from sumtree import SumTree
import math
import numpy as np
import random
import torch


class Replay:
    def __init__(self, max_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_size = 2**math.floor(math.log2(max_size))
        self.memory = SumTree(self.max_size)

    def __len__(self):
        return self.memory.size

    def add(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def transform(self, lazy_frame):
        state = torch.from_numpy(lazy_frame.__array__()[None] / 255).float()
        return state.to(self.device)

    def update(self, idxs, errors, a):
        for _ in range(len(idxs)):
            self.memory.update(idxs[_], (abs(errors[_]) + 1e-4)**a)

    def sample(self, batch_size, beta):
        s = self.memory.get_total()
        # print(s)
        interval = s / batch_size
        idxs, states, actions, rewards, next_states, dones, probs = [], [], [], [], [], [], []
        for _ in range(batch_size):
            lft = _ * interval
            rgt = (_ + 1) * interval
            value = random.uniform(lft, rgt)
            idx, data, prob = self.memory.sample(value)
            # print(data)
            frame, action, reward, next_frame, done = data
            state = self.transform(frame)
            next_state = self.transform(next_frame)
            state = torch.squeeze(state, 0)
            next_state = torch.squeeze(next_state, 0)
            idxs.append(idx)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            probs.append(prob)

        max_ratio = (self.memory.get_min() / s)**-beta
        ratios = [p**-beta / max_ratio for p in probs]
        return (torch.tensor(idxs).to(self.device), torch.stack(states).to(self.device),
                torch.tensor(actions).to(self.device), torch.tensor(rewards).to(self.device),
                torch.stack(next_states).to(self.device), torch.tensor(dones).to(self.device),
                torch.tensor(ratios).to(self.device))
