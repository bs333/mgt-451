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
    """
    Generate a random sample from the gamma distribution using the provided shape.
    """
    return -math.log(math.prod(random.random() for _ in range(shape)))

def beta(alpha, beta):
    """
    Generate a random sample from the beta distribution using the given alpha and beta values.
    
    Args:
    - alpha (int): Number of successes + 1.
    - beta (int): Number of failures + 1.
    
    Returns:
    - float: A sample from the Beta(alpha, beta) distribution.
    """

    x = gamma(alpha)
    y = gamma(beta)
    return x / (x + y)


def epsilon_greedy(numTrials, p1, p2, epsilon):
    """
    Epsilon-Greedy strategy for multi-armed bandit problem.
    
    With probability epsilon, a random action is taken.
    With probability (1-epsilon), the action with the current highest estimated value is taken.
    
    Args:
    - numTrials (int): Number of times the bandit is played.
    - p1 (float): Probability of reward for action 1.
    - p2 (float): Probability of reward for action 2.
    - epsilon (float): Probability to explore (i.e., take random action).
    
    Returns:
    - int: Total reward obtained over numTrials.
    """
    estimated_rewards = [0, 0]  # Track estimated rewards for each action
    counts = [0, 0]  # Track count of each action being chosen
    totalReward = 0  # Total rewards accumulated

    for _ in range(numTrials):
        if random.random() < epsilon:  # Exploration
            choice = random.choice([0, 1])
        else:  # Exploitation
            choice = estimated_rewards.index(max(estimated_rewards))
        
        reward = random.random() < (p1 if choice else p2)  # Simulate getting reward
        counts[choice] += 1
        # Update estimated reward for chosen action
        estimated_rewards[choice] = ((estimated_rewards[choice] * (counts[choice]-1)) + reward) / counts[choice]
        totalReward += reward
    
    return totalReward

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


def epsilon_greedy(numTrials, p1, p2, epsilon):
    """
    Function that implements the epsilon
    """
    estimated_rewards = [0, 0]
    counts = [0, 0]
    totalReward = 0

    for i in range(numTrials):
        if random.random() < epsilon:
            choice = random.choice([0, 1])
        else:
            choice = estimated_rewards.index(max(estimated_rewards))
        
        reward = random.random() < (p1 if choice else p2)
        counts[choice] += 1
        estimated_rewards[choice] = ((estimated_rewards[choice] * (counts[choice]-1)) + reward) / counts[choice]
        totalReward += reward
    
    return totalReward
