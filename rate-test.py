import redis
import random
from collections import OrderedDict, Counter, deque
import numpy as np

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Simulate data for 1000 movies
keys = [f"movie_{i}" for i in range(1, 1001)]  # 1000 movies
values = [f"info_{i}" for i in range(1, 1001)]
test_data = dict(zip(keys, values))

# Load test data into Redis
for key, value in test_data.items():
    r.set(key, value)

# Generate 10000 random access requests
requests = random.choices(keys, k=10000)


# Cache Strategy Classes

class LRUCache:
    """ Least Recently Used (LRU) cache strategy """

    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return True
        return False

    def put(self, key):
        self.cache[key] = None
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class LFUCache:
    """ Least Frequently Used (LFU) cache strategy """

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # Store key-value pairs
        self.freq = Counter()  # Track access frequency

    def get(self, key):
        if key in self.cache:
            self.freq[key] += 1
            return True
        return False

    def put(self, key):
        if len(self.cache) >= self.capacity:
            # Find and evict the least frequently used key
            lfu_key = self.freq.most_common()[:-2:-1][0][0]
            self.cache.pop(lfu_key)
            self.freq.pop(lfu_key)
        self.cache[key] = None
        self.freq[key] = 1


class RandomCache:
    """ Random replacement cache strategy """

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = set()

    def get(self, key):
        return key in self.cache

    def put(self, key):
        if len(self.cache) >= self.capacity:
            self.cache.remove(random.choice(list(self.cache)))
        self.cache.add(key)


class FIFOCache:
    """ First-In-First-Out cache strategy """

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = set()
        self.order = deque()

    def get(self, key):
        return key in self.cache

    def put(self, key):
        if len(self.cache) >= self.capacity:
            oldest = self.order.popleft()
            self.cache.remove(oldest)
        self.cache.add(key)
        self.order.append(key)


class MRUCache:
    """ Most Recently Used cache strategy """

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}

    def get(self, key):
        if key in self.cache:
            return True
        return False

    def put(self, key):
        self.cache[key] = None
        if len(self.cache) > self.capacity:
            # Delete the most recently used element
            last = next(reversed(self.cache))
            del self.cache[last]


class SHiPCache:
    """ SHiP (Signature History based Predictor) cache strategy """

    def __init__(self, capacity):
        self.capacity = capacity
        self.main_cache = set()
        self.history = set()

    def get(self, key):
        if key in self.main_cache:
            return True
        if key in self.history:
            self.history.remove(key)
            self.main_cache.add(key)
            return True
        return False

    def put(self, key):
        if len(self.main_cache) + len(self.history) >= self.capacity:
            if len(self.history) > 0:
                self.history.pop()
            else:
                self.main_cache.pop()
        self.history.add(key)


# Define a function to test different cache replacement strategies
def test_cache_strategy(cache_class, capacity, requests):
    cache = cache_class(capacity)
    hits, misses = 0, 0
    for req in requests:
        if cache.get(req):
            hits += 1
        else:
            misses += 1
            cache.put(req)
    return hits / len(requests)


# Test Cache Strategies

# Test FIFO Strategy
fifo_hit_rate = test_cache_strategy(FIFOCache, 100, requests)
print(f"FIFO Hit rate: {fifo_hit_rate:.4f}")

# Test Random Strategy
random_hit_rate = test_cache_strategy(RandomCache, 100, requests)
print(f"Random Hit rate: {random_hit_rate:.4f}")

# Test LFU Strategy
lfu_hit_rate = test_cache_strategy(LFUCache, 100, requests)
print(f"LFU Hit rate: {lfu_hit_rate:.4f}")

# Test LRU Strategy
lru_hit_rate = test_cache_strategy(LRUCache, 100, requests)
print(f"LRU Hit rate: {lru_hit_rate:.4f}")

# Test MRU Strategy
mru_hit_rate = test_cache_strategy(MRUCache, 100, requests)
print(f"MRU Hit rate: {mru_hit_rate:.4f}")

# Test SHiP Strategy
ship_hit_rate = test_cache_strategy(SHiPCache, 100, requests)
print(f"SHiP Hit rate: {ship_hit_rate:.4f}")
