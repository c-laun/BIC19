import numpy as np


class Eratosthenes(object):

    def __init__(self, limit):
        self.limit = limit
        self.sieve = np.empty(limit + 1, dtype=bool)
        self.sieve[:] = True
        self.sum = None

    def compute_sieve(self):

        self.sieve[0, 1] = False

        for i in range(2, self.limit+1):
            if self.sieve[i]:
                j = 2
                while i*j <= self.limit:
                    self.sieve[i*j] = False
                    j += 1

    def add_sieve(self):
        values = np.arange(self.limit + 1, dtype=int)
        self.sum = np.sum(values[self.sieve])

    def print_sum(self):
        print(f"The sum of primes up to {self.limit} is {self.sum}")

    def run(self):
        self.compute_sieve()
        self.add_sieve()
        self.print_sum()


Eratosthenes(1000).run()
