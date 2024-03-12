# POLICY ITERATION ALGORITHM

## AIM
To implement a policy iteration algorithm for the given MDP.

## State Space:

The states include two terminal states: 0-Hole[H] and 6-Goal[G]. It has five non terminal states includin starting state.
Action Space:

  Left:0
  Right:1

## Transition Probability:

The transition probabilities for the problem statement is:

  50% - The agent moves in intended direction.
  33.33% - The agent stays in the same state.
  16.66% - The agent moves in orthogonal direction.

## Reward:
To reach state 7 (Goal) : +1 otherwise : 0

## Graphical Representation:

![Screenshot 2024-03-12 084736](https://github.com/Anas536/policy-iteration-algorithm/assets/139841834/aa68b254-7ce6-4046-9d25-6392f0132492)


## PROBLEM STATEMENT


## POLICY ITERATION ALGORITHM
The algorithm implemented in the policy_iteration is a method used to find the optimal policy in a Markov decision process (MDP). Here's a step-by-step explanation of the algorithm:

  1. Initialize the policy pi. In this implementation, a random action is chosen for each state s in the MDP P. The initial policy is represented by the lambda function pi=lambda s:{s:a for s,a in enumerate(random_actions)}[s], where random_actions is a list of randomly chosen actions for each state.

  2. Enter a loop that continues until the policy pi is no longer changing. This is determined by comparing the previous policy (old_pi) with the current policy computed in the loop.

  3. Store the previous policy as old_pi for comparison later.

  4. Perform policy evaluation using the function policy_evaluation. This step calculates the state-values (V) for each state s given the current policy pi. The state-values represent the expected cumulative rewards starting from state s following policy pi and discounting future rewards by a factor of gamma. The function policy_evaluation is called with the arguments pi, P, gamma, and theta.

  5. Perform policy improvement using the function policy_improvement. This step updates the policy pi based on the current state-values V. The function policy_improvement is called with the arguments V, P, and gamma.

  6. Check if the policy has converged by comparing the previous policy old_pi with the current policy {s:pi(s) for s in range(len(P))}. If they are the same for all states s, the loop is exited.

  7. Return the final state-values V and the optimal policy pi.

To summarize, policy iteration iteratively improves the policy by alternating between policy evaluation and policy improvement steps until convergence is reached. The algorithm guarantees to find the optimal policy for the given MDP P with a discount factor gamma.
## POLICY IMPROVEMENT FUNCTION
```
# Name: Mohamed Anas 0.I
# Reg No: 212223110028
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    while True:
        V = np.zeros(len(P), dtype=np.float64)
        for s in range(len(P)):
            for prob, next_state, reward, done in P[s][pi(s)]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
        if np.max(np.abs(prev_V - V)) < theta:
            break
        prev_V = V.copy()
    return V
```

## POLICY ITERATION FUNCTION
Include the policy iteration function

## OUTPUT:
Mention the optimal policy, optimal value function , success rate for the optimal policy.

## RESULT:

Write your result here
