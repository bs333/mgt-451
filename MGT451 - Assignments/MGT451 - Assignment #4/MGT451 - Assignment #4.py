"""
MGT451 - Assignment #4 ~ Multi-Armed Bandit Problem

Author: Sid Bhatia

Date: October 19th, 2023

Pledge: I pledge my honor that I have abided by the Stevens Honor System.

Professor: Dr. Jordan Suchow

In this assignment, you will implement the following three strategies for the multi-armed bandit problem, using the attached code as a starting point:

- Epsilon-greedy
- UCB
- Thompson sampling
"""

import random
import math


def gamma(shape):
    return -math.log(math.prod(random.random() for _ in range(shape)))

def beta(alpha, beta):
    """
    Generate random number from beta distribution using the ratio of two gamma random variables.
    """
    x = gamma(alpha)
    y = gamma(beta)
    return x / (x + y)

# Example usage:
# sample = beta(1, 1)
# print(sample)

def f(history):
    # print(history)
    return random.random() > 0.5

numTrials = 10
p1 = 0.25
p2 = 0.75

totalReward = 0
history = []
for i in range(numTrials):
    choice = f(history)
    if choice:
        reward = random.random() < p1
    else:
        reward = random.random() < p2

    print("Trial {}: {}, {}".format(i, choice, reward))

    totalReward += reward
    history.append((choice, reward))



