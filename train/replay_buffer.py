"""
replay_buffer.py

Prioritized Experience Replay (PER), Schaul et al. 2016, implemented with a
sum-tree for O(log n) sampling/update — not the tutorial-style "sort a list"
version. Supports importance-sampling weight correction with annealed beta.
"""
import numpy as np


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity=50_000, alpha=0.6, beta_start=0.4,
                 beta_frames=100_000, eps=1e-5):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps
        self.frame = 1
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        # new transitions get max priority so they're sampled at least once
        self.tree.add(self.max_priority ** self.alpha, transition)

    def _beta(self):
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def sample(self, batch_size: int):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            while data == 0:  # safety guard against empty slots early on
                s = np.random.uniform(a, b)
                idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        self.frame += 1
        beta = self._beta()
        sampling_probs = np.array(priorities) / self.tree.total()
        n = self.tree.n_entries
        is_weights = np.power(n * sampling_probs + 1e-10, -beta)
        is_weights /= is_weights.max()

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            idxs,
            is_weights.astype(np.float32),
        )

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            p = (abs(err) + self.eps) ** self.alpha
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return self.tree.n_entries


if __name__ == "__main__":
    buf = PrioritizedReplayBuffer(capacity=1000)
    for i in range(50):
        s = np.random.randn(8)
        ns = np.random.randn(8)
        buf.push(s, np.random.randint(5), np.random.randn(), ns, False)
    states, actions, rewards, next_states, dones, idxs, w = buf.sample(16)
    print("Sampled batch shapes:", states.shape, actions.shape, rewards.shape, w.shape)
    buf.update_priorities(idxs, np.random.rand(16))
    print("PER buffer smoke test OK. len(buf)=", len(buf))
