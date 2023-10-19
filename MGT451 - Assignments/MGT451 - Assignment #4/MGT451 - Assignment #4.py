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

epsilon = 0.1

def epsilon_greedy(history):
    if random.random() < epsilon:
        # Exploration: Choose a random action
        return random.random() > 0.5
    else:
        # Exploitation: Choose the action with the highest observed reward.
        if history:
            return max(history, key=lambda x: x[1])[0]
        else:
            return random.random() > 0.5

def ucb(history, c):
    t = len(history) + 1

    # Initialize counts and rewards for each action
    action_counts = [0, 0]
    action_rewards = [0, 0]

    for i in range(len(history)):
        choice, reward = history[i]
        action_counts[choice] += 1
        action_rewards[choice] += reward

    ucb_values = []

    for action in range(2):
        if action_counts[action] == 0:
            # If an action hasn't been chosen yet, choose it
            ucb_values.append(float('inf'))
        else:
            # Calculate UCB value for the action
            mean_reward = action_rewards[action] / action_counts[action]
            ucb_value = mean_reward + c * math.sqrt(2 * math.log(t) / action_counts[action])
            ucb_values.append(ucb_value)

    # Choose the action with the highest UCB value
    return ucb_values.index(max(ucb_values))


def thompson_sampling(history):
    alpha1 = sum(1 for choice, reward in history if choice and reward) + 1
    beta1 = sum(1 for choice, reward in history if choice and not reward) + 1
    alpha2 = sum(1 for choice, reward in history if not choice and reward) + 1
    beta2 = sum(1 for choice, reward in history if not choice and not reward) + 1

    # Sample from beta distributions and choose the action with the highest sample
    action1 = beta(alpha1, beta1)
    action2 = beta(alpha2, beta2)
    
    if action1 > action2:
        return True
    else:
        return False

# For epsilon-greedy strategy
choice = epsilon_greedy(history)

# For UCB strategy
c = 1.0
choice = ucb(history, c)

# For Thompson Sampling strategy
choice = thompson_sampling(history)

# Test cases for Epsilon-Greedy Strategy
print("Epsilon-Greedy Strategy Test Cases:")
epsilon = 0.1  # Set epsilon for epsilon-greedy strategy
history = []  # Initialize history

# Test case 1: High epsilon for exploration
epsilon = 0.9
history = []
for i in range(10):
    choice = epsilon_greedy(history)
    print("Trial {}: Choice: {}, Reward: {}".format(i, choice, random.random() > 0.5))
    history.append((choice, random.random() > 0.5))

# Test case 2: Low epsilon for exploitation
epsilon = 0.1
history = []
for i in range(10):
    choice = epsilon_greedy(history)
    print("Trial {}: Choice: {}, Reward: {}".format(i, choice, random.random() > 0.5))
    history.append((choice, random.random() > 0.5))

# Test cases for UCB Strategy
print("\nUCB Strategy Test Cases:")
c = 1.0  # Set UCB exploration parameter
history = []  # Initialize history

# Test case 1: High exploration parameter
c = 2.0
history = []
for i in range(10):
    choice = ucb(history)
    print("Trial {}: Choice: {}, Reward: {}".format(i, choice, random.random() > 0.5))
    history.append((choice, random.random() > 0.5))

# Test case 2: Low exploration parameter
c = 0.5
history = []
for i in range(10):
    choice = ucb(history)
    print("Trial {}: Choice: {}, Reward: {}".format(i, choice, random.random() > 0.5))
    history.append((choice, random.random() > 0.5))

# Test cases for Thompson Sampling Strategy
print("\nThompson Sampling Strategy Test Cases:")
history = []  # Initialize history

# Test case 1: Initial values for beta distributions
alpha1, beta1 = 1, 1
alpha2, beta2 = 1, 1
history = []
for i in range(10):
    choice = thompson_sampling(history)
    reward = random.random() > 0.5
    print("Trial {}: Choice: {}, Reward: {}".format(i, choice, reward))
    history.append((choice, reward))

# Test case 2: Different initial values for beta distributions
alpha1, beta1 = 2, 3
alpha2, beta2 = 3, 2
history = []
for i in range(10):
    choice = thompson_sampling(history)
    reward = random.random() > 0.5
    print("Trial {}: Choice: {}, Reward: {}".format(i, choice, reward))
    history.append((choice, reward))
