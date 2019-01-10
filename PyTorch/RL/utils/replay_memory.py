import numpy as np

class ReplayMemory:
    def __init__(self, capacity, alpha, beta, epsilon=1e-5):
        self.experiences = SumTree(capacity)

        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

        self.beta_increment = 0.001

    def __len__(self):
        return self.experiences.n_entries

    def _get_priority(self, error):
        return (error + self.epsilon) ** self.alpha

    def add(self, experience, error):
        priority = self._get_priority(error)
        self.experiences.add(experience, priority)

    def sample(self, size):
        batch = []
        idxs = []
        priorities = []

        segment = self.experiences.total() / size

        self.beta = np.min([1, self.beta + self.beta_increment])

        for i in range(size):
            low = segment * i
            high = segment * (i + 1)

            sample = np.random.uniform(low, high)

            idx, experience, priority = self.experiences.get(sample)

            batch.append(experience)
            idxs.append(idx)
            priorities.append(priority)

        probabilities = priorities / self.experiences.total()
        is_weights = np.power(self.experiences.n_entries * probabilities,
                              -self.beta)
        is_weights /= is_weights.max()

        return batch, idxs, is_weights

    def update_weights(self, idxs, errors):
        self.beta += 0.001
        self.beta = np.min([self.beta, 1])

        for idx, error in zip(idxs, errors):
            priority = self._get_priority(error)
            self.experiences.update(idx, priority)

    def size(self):
        return self.experiences.n_entries

class SumTree():
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
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

    # store priority and sample
    def add(self, data, p):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.data[dataIdx], self.tree[idx])

class ISMemory():
    def __init__(self, capacity, alpha, beta, epsilon=1e-5):
        self.experiences = SumTree(capacity)

        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

        self.beta_increment = 0.001

    def __len__(self):
        return self.experiences.n_entries

    def _get_priority(self, error):
        return (error + self.epsilon) ** self.alpha

    def add(self, experience, error):
        priority = self._get_priority(error)
        self.experiences.add(experience, priority)

    def sample(self, size):
        batch = []
        idxs = []
        priorities = []

        segment = self.experiences.total() / size

        self.beta = np.min([1, self.beta + self.beta_increment])

        for i in range(size):
            low = segment * i
            high = segment * (i + 1)

            sample = np.random.uniform(low, high)

            idx, experience, priority = self.experiences.get(sample)

            batch.append(experience)
            idxs.append(idx)
            priorities.append(priority)

        probabilities = priorities / self.experiences.total()
        is_weights = np.power(self.experiences.n_entries * probabilities,
                              -self.beta)
        is_weights /= is_weights.max()

        return batch, idxs, is_weights

    def update_weights(self, idxs, errors):
        self.beta += 0.001
        self.beta = np.min([self.beta, 1])

        for idx, error in zip(idxs, errors):
            priority = self._get_priority(error)
            self.experiences.update(idx, priority)

    def size(self):
        return self.experiences.n_entries

class SumTree():
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
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

    # store priority and sample
    def add(self, data, p):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.data[dataIdx], self.tree[idx])
