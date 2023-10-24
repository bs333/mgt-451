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
    """Generate a random sample from the gamma distribution for integer shape parameters."""
    if shape < 1:
        raise ValueError("Shape parameter must be an integer greater than or equal to 1.")
    
    # For shape = 1, the gamma distribution simplifies to the exponential distribution
    if shape == 1:
        return -math.log(1.0 - random.random())

    # Acceptance-rejection method
    while True:
        x = 1.0
        for _ in range(shape):
            x *= random.random()
        y = (1.0 - random.random()) ** (shape - 1)
        if x > y:
            return -math.log(x)


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

def ucb(numTrials, p1, p2):
    """
    Upper Confidence Bound (UCB) strategy for multi-armed bandit problem.
    
    Balances between exploration and exploitation based on uncertainty in reward estimate of an action.
    
    Args:
    - numTrials (int): Number of times the bandit is played.
    - p1 (float): Probability of reward for action 1.
    - p2 (float): Probability of reward for action 2.
    
    Returns:
    - int: Total reward obtained over numTrials.
    """
    estimated_rewards = [0.5, 0.5]  # Track estimated rewards for each action
    counts = [1, 1]  # Initialize with 1 to avoid division by zero later
    totalReward = 0  # Total rewards accumulated

    for i in range(numTrials):
        # Calculate UCB value for each action
        ucb_values = [
            estimated_rewards[j] + math.sqrt(2 * math.log(i+2) / counts[j])
            for j in range(2)
        ]
        
        choice = ucb_values.index(max(ucb_values))  # Choose action with max UCB value
        reward = random.random() < (p1 if choice else p2)  # Simulate getting reward
        counts[choice] += 1
        # Update estimated reward for chosen action
        estimated_rewards[choice] = ((estimated_rewards[choice] * (counts[choice]-1)) + reward) / counts[choice]
        totalReward += reward

    return totalReward

def thompson_sampling(numTrials, p1, p2):
    """
    Thompson Sampling strategy for multi-armed bandit problem.
    
    Uses Bayesian posterior distributions of rewards to probabilistically choose an action.
    
    Args:
    - numTrials (int): Number of times the bandit is played.
    - p1 (float): Probability of reward for action 1.
    - p2 (float): Probability of reward for action 2.
    
    Returns:
    - int: Total reward obtained over numTrials.
    """
    alphas = [1, 1]  # Number of successes + 1 for each action
    betas = [1, 1]  # Number of failures + 1 for each action
    totalReward = 0  # Total rewards accumulated

    for _ in range(numTrials):
        sampled_rewards = [beta(alphas[j], betas[j]) for j in range(2)]  # Sample reward for each action
        choice = sampled_rewards.index(max(sampled_rewards))  # Choose action with max sampled reward
        reward = random.random() < (p1 if choice else p2)  # Simulate getting reward
        
        # Update successes and failures for chosen action
        if reward:
            alphas[choice] += 1
        else:
            betas[choice] += 1
        totalReward += reward

    return totalReward

# Example usage
numTrials = 1000
p1 = 0.25
p2 = 0.75
epsilon = 0.1

print("Epsilon Greedy Reward:", epsilon_greedy(numTrials, p1, p2, epsilon))
print("UCB Reward:", ucb(numTrials, p1, p2))
print("Thompson Sampling Reward:", thompson_sampling(numTrials, p1, p2))

