# Reinforcement Learning (David Silver主讲）

## Lecture 1: introduction to reinforcement learning

### outline:

1. About reinforcement learning
2. The reinforcement learning problem
3. Inside an RL agent
4. Problems within reinforcement learning

### textbooks

1. An introduction to reinforcement learning, Sutton and Barto, 1998
   MIT Press, 1998
   http://webdocs.cs.ualberta.ca/~sutton/book/the-book.html
2. Algorithms for reinforcement learning, Szepesvari
   Morgan and Claypool, 2010
   http://www.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf

### characteristics of reinforcement learning

What makes reinforcement learning different from other machine learning paradigms?

1. There is no supervisor, only a reward signal
2. Feedback is delayed, not instantaneous
3. Time really matters (sequential, non i.i.d data)
4. Agent’s actions affect the subsequent data it receives

### Rewards

1. A reward $\mathcal{R_t}$ is a scalar feedback signal
2. Indicates how well agent is doing at step $t$
3. The agent’s job is to maximize cumulative reward
   Reinforcement learning is based on the reward hypothesis

#### Definition (Reward Hypothesis)

All goals can be described by the maximization of expected cumulative reward

### Agent and Environment

1. At each step $t$ the agent :
   1. Executes action $A_t$
   2. Receives observation $O_t$
   3. Receives scalar reward $\mathcal{R_t}$
2. The environment :
   1. Receives action $A_t$
   2. Emits observation $O_{t+1}$
   3. Emits scalar reward $\mathcal{R_{t+1}}$
3. $t$ increments at environment step

### History and State

1. The history is the sequence of observations, actions, rewards	$\mathcal{H_t} = {O_1}, {\mathcal{R_1}, {A_1}, \ldots, {A_{t-1}}, {O_t}, {\mathcal{R_t}}}$
2. i.e. all observable variables up to time $t$
3. i.e. the sensorimotor stream of a robot or embodied agent
4. What happens next depends on the history :
   1. The agent selects actions
   2. The environment selects observations/rewards
5. State is the information used to determine what happens next
6. Formally, state is a function of the history : $\mathcal{S_t} = f(\mathcal{H_t})$

### Environment State

1. The environment state $\mathcal{S_{t}^{e}}$ is the environment’s private representation
2. i.e. whatever data the environment uses to pick the next observation/rewards
3. The environment state is not usually visible to the agent
4. Even if $\mathcal{S_{t}^{e}}$ is visible, it may contain irrelevant information

### Agent State

1. The agent state $\mathcal{S_t^a}$ is the agent’s internal representation
2. i.e. whatever information the agent uses to pick the next action
3. i.e. it is the information used by reinforcement learning algorithms
4. It can be any function of history : $\mathcal{S_t^a} = f(\mathcal{H_t})$

### Information State

An information state(a.k.a Markov State) contains all useful information from the history

#### Definition

A state $\mathcal{S_t}$ is Markov if and only if $\mathbb{P}[\mathcal{S_{t+1}}|\mathcal{S_t}] = \mathbb{P}[\mathcal{S_{t+1}}|\mathcal{S_1}, \mathcal{S_2}, \ldots, \mathcal{S_t}]$

1. “The future is independent of the past given the present”	$\mathcal{H_{1:t} \to \mathcal{S_t} \to \mathcal{H_{t+1:\infty}}}$
2. Once the state is known, the history may be thrown away
3. i.e. the state is a sufficient statistic of the future
4. The environment state $\mathcal{S_t^e}$ is Markov
5. The history $\mathcal{H_t}$ is Markov

### Full Observation Environments

Fully observability : agent directly observes environment state $O_t = \mathcal{S_t^a} = \mathcal{S_t^e}$

1. Agent state = environment state = information state
2. Formally, this is a Markov Decision Process(MDP)

### Partially Observable Environments

1. Partial observability : agent indirectly observes environment
2. Now agent state $\neq$ environment state
3. Formally this is a partially observable Markov Decision Process(POMDP)
4. Agent must construct its own state representation $\mathcal{S_t^a}$, e.g.
   1. Complete history : $\mathcal{S_t} = \mathcal{H_t}$
   2. Beliefs of environment state : $\mathcal{S_t^a} = (\mathbb{P}[\mathcal{S_t^e} = s^1], \ldots, \mathbb{P}[\mathcal{S_t^e} = s^n])$
   3. Recurrent neural network : $\mathcal{S_t^a} = \sigma(\mathcal{S_{t-1}^a}W_s + O_tW_o)$

### Major Components of an RL Agent

An RL agent may include one or more of these components:

1. Policy : agent’s behavior function
2. Value function : how good is each state and/or action
3. Model : agent’s representation of the environment

#### Policy

1. A policy is the agent’s behavior
2. It is a map from state to action, e.g.
3. Deterministic policy: $a = \pi(s)$
4. Stochastic policy: $\pi(a|s) = \mathbb{P}[A_t = a|\mathcal{S_t} = s]$

#### Value Function

1. Value function is a prediction of future reward
2. Used to evaluate the goodness /badness of states
3. And therefore to select between actions, e.g.
   $V_\pi(s) = \mathbb{E}[\mathcal{R_t + \gamma\mathcal{R_{t+1}} + \gamma^2\mathcal{R_{t+3}} + \ldots|\mathcal{S_t = s}}]$

#### Model

1. A model predicts what the environment will do next
2. $\mathcal{P}$ predicts the next state
3. $\mathcal{R}$ predicts the next (immediate) reward, e.g.
   $\mathcal{P_{ss'}^a} = \mathbb{P}[\mathcal{S_{t+1} = s'|\mathcal{S_t} = s}, A_t = a]$
   $\mathcal{R_s^a} = \mathbb{E}[\mathcal{R_{t+1}}|\mathcal{S_t = s}, A_t = a]$

### categorizing RL agents (1)

1. Value Based : no policy(implicit) + value function
2. Policy Based : policy + no value function
3. Actor Critic : policy + value function

### categorizing RL agents (2)

1. Model Free : policy and/or value function + no model
2. Model Based : policy and/or value function + model

### learning and planning

Two fundamental problems in sequential decision making

1. Reinforcement learning:
   1. The environment is initially unknown
   2. The agent interacts with the environment
   3. The agent improves its policy
2. Planning:
   1. A model of the environment is known
   2. The agent performs computations with its model (without any external interaction)
   3. The agent improves its policy
   4. A.K.A deliberation, reasoning, introspection, pondering, thought, search

### exploration and exploitation (1)

1. Reinforcement learning is like trial-and-error learning
2. The agent should discover a good policy
3. From its experiences of the environment
4. Without losing too much reward along the way

### exploration and exploitation (2)

1. Exploration finds more information about the environment
2. Exploitation exploits known information to maximize reward
3. It is usually important to explore as well as exploit

### prediction and control

1. Prediction : evaluate the future + given a policy
2. Control : optimize the future + find the best policy

## course outline:

### part 1 : elementary reinforcement learning

1. Introduction to RL
2. Markov decision process
3. Planning by dynamic programming
4. Model-free prediction
5. Model-free control

### part 2 : reinforcement learning in practice

1. Value function approximation
2. Policy gradient methods
3. Integrating learning and planning
4. Exploration and exploitation
5. Case study - RL in games

## Lecture 2: Markov Decision Process

### outline:

1. Markov processes
2. Markov reward processes
3. Markov decision processes
4. Extensions to MDPs

### introduction to MDPs

1. Markov decision processes formally describe an environment for reinforcement learning
2. Where the environment is fully observable
3. i.e. the current state completely characterizes the process
4. Almost all RL problems can be formalized as MDPs, e.g.
   1. Optimal control primarily deals with continuous MDPs
   2. Partially observable problems can be converted into MDPs
   3. Bandits are MDPs with one state

### Markov property

“The future is independent of the past given the present”

#### Definition

A state $\mathcal{S_t}$ is Markov if and only if $\mathbb{P}[\mathcal{S_{t+1}|S_t}] = \mathbb{P}[\mathcal{S_{t+1}|S_1, S_2, \ldots, S_t}]$

1. The state captures all relevant information from the history
2. Once the state is known, the history may be thrown away
3. i.e. the state is a sufficient statistic of the future

### state transition matrix

For a Markov state $s$ and successor state $s'$, the state transition probability is defined by $\mathcal{P_{ss'}} = \mathbb{P}[\mathcal{S_{t+1} = s'|S_t = s}]$
State transition matrix $\mathcal{P}$ defines transition probabilities from all states $s$ to all successor states $s'$,

$$
\mathcal{P} = \begin{pmatrix}
\mathcal{P}_{11} & \mathcal{P}_{12} & \cdots & \mathcal{P}_{1n} \\
\mathcal{P}_{21} & \mathcal{P}_{22} & \cdots & \mathcal{P}_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
\mathcal{P}_{n1} & \mathcal{P}_{n2} & \cdots & \mathcal{P}_{nn}
\end{pmatrix}
$$

Where each row of the matrix sums to 1

### Markov process

A Markov process is a memoryless random process, i.e. a sequence of random states $\mathcal{S_1}, \mathcal{S_2}, \ldots$ with the Markov property

#### Definition

A Markov process(or Markov chain) is a tuple $<\mathcal{S}, \mathcal{P}>$

1. $\mathcal{S}$ is a (finite) set of states
2. $\mathcal{P}$ is a state transition probability matrix, $\mathcal{P_{ss'}} = \mathbb{P}[\mathcal{S_{t+1} = s'|S_t = s}]$

### Markov reward process

A Markov reward process is a Markov chain with values

#### Definition

A Markov reward process is a tuple $<\mathcal{S, P, R, \gamma}>$

1. $\mathcal{S}$ is a finite set of states
2. $\mathcal{P}$ is a state transition probability matrix, $\mathcal{P_{ss'}} = \mathbb{P}[\mathcal{S_{t+1} = s'|S_t = s}]$
3. $\mathcal{R}$ is a reward function, $\mathcal{R_{s}} = \mathbb{E}[\mathcal{R_{t+1}|S_t = s}]$
4. $\gamma$ is a discount factor, $\gamma \in [0,1]$

### Return

#### Definition

The return $G_t$ is the total discounted reward from time-step $t$
$G_t = \mathcal{R_{t+1} + \gamma R_{t+2}} + \ldots = \sum_{k = 0}^{\infty}{\gamma \mathcal{R_{t+k+1}}}$

1. The discount $\gamma \in [0,1]$ is the present value of future rewards
2. The value of receiving reward $\mathcal{R}$ after $k + 1$ time-steps is $\gamma^k \mathcal{R}$
3. This values immediate reward above delayed reward
   1. $\gamma$ close to $0$ leads to “myopic” evaluation
   2. $\gamma$ close to $1$ leads to “far-sighted” evaluation

### why discount?

Most Markov reward and decision processes are discounted. Why?

1. Mathematically convenient to discount rewards
2. Avoids infinite returns in cyclic Markov processes
3. Uncertainty about the future may not be fully represented
4. If the reward is financial, immediate rewards may earn more interest than delayed rewards
5. Animal/human behavior shows preference for immediate rewards
6. It is sometimes possible to use undiscounted  Markov reward processes (i.e. $\gamma = 1$), e.g. if all sequences terminate

### Value Function

The value function $V(s)$ gives the long-term value of state $s$

#### Definition

The state value function $V(s)$ of an MRP is the expected return starting from state $s$
$V(s) = \mathbb{E}[G_t|\mathcal{S_t} = s]$

### Bellman Equation for MRPs (1)

The value function can be decomposed into two parts:

1. Immediate reward $\mathcal{R_t}$
2. Discounted value of successor state $\gamma V(\mathcal{S_{t+1}})$

$$
\begin{aligned}
V(s) &= \mathbb{E}[G_t | \mathcal{S}_t = s] \\
&= \mathbb{E}[\mathcal{R}_{t+1} + \gamma \mathcal{R}_{t+2} + \gamma^2 \mathcal{R}_{t+3} + \dots | \mathcal{S}_t = s] \\
&= \mathbb{E}[\mathcal{R}_{t+1} + \gamma (\mathcal{R}_{t+2} + \gamma \mathcal{R}_{t+3} + \dots) | \mathcal{S}_t = s] \\
&= \mathbb{E}[\mathcal{R}_{t+1} + \gamma G_{t+1} | \mathcal{S}_t = s] \\
&= \mathbb{E}[\mathcal{R}_{t+1} + \gamma V(\mathcal{S}_{t+1}) | \mathcal{S}_t = s]
\end{aligned}
$$

### Bellman Equation for MRPs (2)

$V(s) = \mathbb{E}[\mathcal{R}_{t+1} + \gamma V(\mathcal{S}_{t+1}) | \mathcal{S}_t = s]$
$V(s) = \mathcal{R}_{s} + \gamma \sum_{s' \in \mathcal{S}}{\mathcal{P_{ss'}}V(s')}$

### Bellman Equation in Matrix Form

The Bellman equation can be expressed concisely using matrices $V = \mathcal{R} + \gamma \mathcal{P}V$
Where $V$ is a column vector with one entry per state

$$
\begin{bmatrix}
V(1) \\
\vdots \\
V(n)
\end{bmatrix}
=
\begin{bmatrix}
\mathcal{R}_1 \\
\vdots \\
\mathcal{R}_n
\end{bmatrix}
+ \gamma
\begin{bmatrix}
\mathcal{P}_{11} & \cdots & \mathcal{P}_{1n} \\
\vdots & \ddots & \vdots \\
\mathcal{P}_{n1} & \cdots & \mathcal{P}_{nn}
\end{bmatrix}
\begin{bmatrix}
V(1) \\
\vdots \\
V(n)
\end{bmatrix}
$$

### Solving the Bellman Equation

1. The Bellman equation is a linear equation
2. It can be solved directly:

$$
\begin{align}
&V = \mathcal{R} + \gamma \mathcal{P}V \\
&(I - \gamma \mathcal{P})V = \mathcal{R} \\
&V = (I - \gamma \mathcal{P})^{-1}\mathcal{R} \\
\end{align}
$$

3. Computational complexity is $\mathcal{O(n^3)}$ for $n$ states
4. Direct solution only possible for small MRPs
5. There are many iterative methods for large MRPs, e.g.
   1. Dynamic programming
   2. Monte-Carlo evaluation
   3. Temporal-Difference learning

### Markov decision process

A Markov decision process (MDPs) is a Markov reward process with decisions. It is an environment in which all states are Markov

#### Definition

A Markov decision process is a tuple $<\mathcal{S, A, P, R, \gamma}>$

1. $\mathcal{S}$ is a finite set of states
2. $\mathcal{A}$ is a finite set of actions
3. $\mathcal{P}$ is a state transition probability matrix, $\mathcal{P_{ss'}^a} = \mathbb{P}[\mathcal{S_{t+1} = s'|S_t = s}, A_t =a]$
4. $\mathcal{R}$ is a reward function, $\mathcal{R_s^a} = \mathbb{E}[\mathcal{R_{t+1}|S_t = s}, A_t =a]$
5. $\gamma$ is a discount factor $\gamma \in [0,1]$

### policies (1)

#### Definition

A policy $\pi$ is a distribution over actions given states, $\pi(a|s) = \mathbb{P}[A_t = a|S_t = s]$

1. A policy fully defines the behavior of an agent
2. MDP policies depend on the current state (not the history)
3. i.e. policies are stationary (time-independent), $A_t \sim \pi(\cdot|\mathcal{S_t}), \forall t > 0$

### policies (2)

1. Given an MDP $\mathcal{M} = <\mathcal{S, A, P, R, \gamma}>$ and a policy $\pi$
2. The state sequence $\mathcal{S_1, S_2, \ldots}$ is a Markov process $<\mathcal{S, P^\pi}>$
3. The state and reward sequence $\mathcal{S_1, R_2, S_2, \ldots}$ is a Markov reward process $<\mathcal{S, P^\pi, R^\pi, \gamma}>$
4. Where $\mathcal{P_{ss’}^{\pi}} = \sum_{a \in \mathcal{A}}{\pi(a|s)\mathcal{P_{ss’}^a}}$,  $\mathcal{R_{s}^{\pi}} = \sum_{a \in \mathcal{A}}{\pi(a|s)\mathcal{R_{s}^a}}$

### value function

#### Definition

The state-value function $V_\pi(s)$ of an MDP is the expected return starting from state $s$, and then following policy $\pi$
$V_\pi(s) = \mathbb{E}[G_t|\mathcal{S_t = s}]$

#### Definition

The action-value function $Q_\pi(s,a)$ is the expected return starting from state $s$,  taking action $a$, and then following policy $\pi$
$Q_\pi(s,a) = \mathbb{E}_\pi[G_t|\mathcal{S_t = s,} A_t = a]$

### Bellman Expectation Equation

The state-value function can again be decomposed into immediate reward plus discounted value of successor state, $V_\pi(s) = \mathbb{E}_\pi[\mathcal{R_{t+1}} + \gamma V_\pi(\mathcal{S_{t+1}})|\mathcal{S_t = s}]$
The action-value function can similarly be decomposed,
$Q_\pi(s,a) = \mathbb{E}_\pi[\mathcal{R_{t+1}} + \gamma Q_\pi(\mathcal{S_{t+1}}, A_{t+1})|\mathcal{S_t = s}, A_t = a]$

### Bellman Expectation Equation for $V_\pi$

$V_\pi(s) = \sum_{a \in \mathcal{A}}{\pi(a|s)Q_\pi(s, a)}$

### Bellman Expectation Equation for $Q_\pi$

$Q_\pi(s,a) = \mathcal{R}_{s}^a + \gamma \sum_{s' \in \mathcal{S}}\mathcal{P_{ss'}^a}V_\pi(s')$

### Bellman Expectation Equation for $V_\pi$ (2)

$V_\pi(s) = \sum_{a \in \mathcal{A}}{\pi(a|s) [\mathcal{R_s^a} + \gamma \sum_{s' \in \mathcal{S}}{\mathcal{P_{ss'}^a}}V_\pi(s')}]$

### Bellman Expectation Equation for $Q_\pi$ (2)

$Q_\pi(s, a) = \mathcal{R_s^a} + \gamma \sum_{s' \in \mathcal{S}}{\mathcal{P_{ss'}^a}}\sum_{a' \in \mathcal{A}}\pi(a'|s')Q_\pi(s', a')$

### Bellman Expectation Equation(Matrix Form)

The Bellman expectation equation can be expressed concisely using the induced MRP, $V_\pi = \mathcal{R_\pi} + \gamma \mathcal{P_\pi}V_\pi$ with direct solution $V_\pi = (I - \gamma \mathcal{P_\pi})^{-1}\mathcal{R}_\pi$

### Optimal Value Function

#### Definition

The optimal state-value function $V_*(s)$ is the maximum value function over all policies $V_*(s) = \underset{\pi}{max} \, V_{\pi}(s)$
The optimal action-value function $Q_*(s, a)$ is the maximum action-value function over all policies $Q_*(s, a) = \underset{\pi}{max} \, Q_{\pi}(s, a)$

1. The optimal value function specifies the best possible performance in the MDP
2. An MDP is “solved” when we know the optimal value function

### Optimal Policy

Define a partial ordering over policies $\pi \geq \pi'$ if $V_\pi(s) \geq V_{\pi'}(s), \forall s$

#### theorem

For any Markov Decision Process

1. There exists an optimal policy $\pi_*$ that is better than or equal to all other policies, $\pi_* \geq \pi, \forall \pi$
2. All optimal policies achieve the optimal value function, $V_{\pi_{*}}(s) = V_*(s)$
3. All optimal policies achieve the optimal action-value function, $Q_{\pi_{*}}(s, a) = Q_*(s, a)$

### Finding an Optimal Policy

An optimal policy can be found by maximising over $Q_{\pi_{*}}(s, a)$,

$$
\pi_*(a | s) =
\begin{cases} 
1, & \text{if } a = \operatorname*{arg\,max}_{a \in \mathcal{A}} q_*(s, a), \\[4pt]
0, & \text{otherwise}.
\end{cases}
$$

1. There is always a deterministic optimal policy for any MDP
2. If we know $Q_{\pi_{*}}(s, a)$, we immediately have the optimal policy

### Bellman Optimality Equation for $V_*$

The optimal value functions are recursively related by the Bellman optimality equations: $V_*(s) = \underset{a}{max} \, Q_{\pi}(s, a)$

### Bellman Optimality Equation for $Q_*$

$Q_{*}(s, a) = \mathcal{R_s^a} + \gamma \sum_{s' \in \mathcal{S}}{\mathcal{P_{ss'}^a}V_*(s')}$

### Bellman Optimality Equation for $V_*$ (2)

$V_*(s) = \underset{a}{max} \mathcal{R_s^a} + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P_{ss'}^a}V_*(s')$

### Bellman Optimality Equation for $Q_*$ (2)

$Q_*(s, a) = \mathcal{R_s^a} + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P_{ss'}^a}\underset{a'}{max}\, Q_*(s', a')$

### Solving the Bellman Optimality Equation

1. Bellman Optimality Equation is non-linear
2. No closed form solution (in general)
3. Many iterative solution methods
   1. Value Iteration
   2. Policy Iteration
   3. Q-learning
   4. Sarsa

### Extensions to MDPs

1. Infinite and continuous MDPs
2. Partially observable MDPs
3. Undiscounted, average reward MDPs

### Infinite MDPs

The following extensions are all possible:

1. Countably infinite state and/or action spaces: Straightforward
2. Continuous state and/or action spaces: Closed form for linear quadratic model (LQR)
3. Continuous time:
   1. Requires partial differential equations
   2. Hamilton-Jacobi-Bellman (HJB) equation
   3. Limiting case of Bellman equation as time-step $t \to 0$

### POMDPs

A Partially Observable Markov Decision Process is an MDP with hidden states. It is a hidden Markov model with actions.

#### Definition

A POMDP is a tuple $<\mathcal{S, A, O, P, R, Z, \gamma}>$

1. $\mathcal{S}$ is a finite set of states
2. $\mathcal{A}$ is a finite set of actions
3. $\mathcal{O}$ is a finite set of observations
4. $\mathcal{P}$ is a state transition probability matrix, $\mathcal{P_{ss'}^a} = \mathbb{P}[\mathcal{S_{t+1} = s'|S_t = s}, A_t =a]$
5. $\mathcal{R}$ is a reward function, $\mathcal{R_{s}^a} = \mathbb{E}[\mathcal{R_{t+1}|S_t = s}, A_t =a]$
6. $\mathcal{Z}$ is an observation function, $\mathcal{Z_{s'o}^a} = \mathbb{P}[\mathcal{O_{t+1} = o|S_{t+1} = s'}, A_t =a]$
7. $\gamma$ is a discount factor, $\gamma \in [0,1]$

### Belief States

#### Definition

A history $\mathcal{H_t}$ is a sequence of actions, observations and rewards,
$\mathcal{H_t} = A_0, O_1, \mathcal{R_1}, \ldots, A_{t-1}, O_t, \mathcal{R_t}$

#### Definition

A belief state $b(h)$ is a probability distribution over states, conditioned on the history $\mathcal{H}$
$b(h) = (\mathbb{P}[\mathcal{S_t} = s^1|\mathcal{H_t = h}], \ldots, \mathbb{P}[\mathcal{S_t} = s^n|\mathcal{H_t = h}])$

### Reductions of POMDPs

1. The history $\mathcal{H_t}$ satisfies the Markov property
2. The belief state $b(\mathcal{H_t})$ satisfies the Markov property
3. A POMDP can be reduced to an (infinite) history tree
4. A POMDP can be reduced to an (infinite) belief state tree

### Ergodic Markov Process

An ergodic Markov process is:

1. Recurrent: each state is visited an infinite number of times
2. Aperiodic: each state is visited without any systematic period

#### theorem

An ergodic Markov process has a limiting stationary distribution $d^\pi(s)$ with the property $d^\pi(s) = \sum_{s' \in \mathcal{S}}d^\pi(s')\mathcal{P_{s's}}$

### Ergodic MDP

#### Definition

An MDP is ergodic if the Markov chain induced by any policy is ergodic.
For any policy $\pi$, an ergodic MDP has an average reward per time-step $\rho^\pi$ that is independent of start state.	$\rho^\pi = \underset{T \to \infty}{lim} {\frac{1}{T}}\mathbb{E}[\sum_{t = 1}^{T}\mathcal{R_t}]$

### Average Reward Value Function

1. The value function of an undiscounted, ergodic MDP can be expressed in terms of average reward.
2. $\widetilde{V}_\pi(s)$ is the extra reward due to starting from state s, $\widetilde{V}_\pi(s) = \mathbb{E_\pi}[\sum_{k=1}^{\infty}(\mathcal{R_{t+k} - \rho^\pi})|\mathcal{S_t = s}]$
3. There is a corresponding average reward Bellman equation,

$$
\begin{align*}
\widetilde{V}_\pi(s) 
&= \mathbb{E}_\pi \left[ (\mathcal{R}_{t+1} - \rho^\pi) + \sum_{k=1}^{\infty} (\mathcal{R}_{t+k+1} - \rho^\pi) | \mathcal{S}_t = s \right] \\
&= \mathbb{E}_\pi \left[ (\mathcal{R}_{t+1} - \rho^\pi) + \widetilde{V}_\pi(\mathcal{S}_{t+1}) | \mathcal{S}_t = s \right]
\end{align*}
$$

## Lecture 3: Planning by Dynamic Programming

### outline:

1. Introduction
2. Policy evaluation
3. Policy iteration
4. Value iteration
5. Extensions to dynamic programming
6. Contraction mapping

### What is Dynamic Programming?

Dynamic sequential or temporal component to the problem
Programming optimising a “program”, i.e. a policy c.f. linear programming

1. A method for solving complex problems
2. By breaking them down into subproblems
   1. Solve the subproblems
   2. Combine solutions to subproblems

### Requirements for Dynamic Programming

Dynamic Programming is a very general solution method for problems which have two properties:

1. Optimal substructure
   1. Principle of optimality applies
   2. Optimal solution can be decomposed into subproblems
2. Overlapping subproblems
   1. Subproblems recur many times
   2. Solutions can be cached and reused
3. Markov decision processes satisfy both properties
   1. Bellman equation gives recursive decomposition
   2. Value function stores and reuses solutions

### Planning by Dynamic Programming

1. Dynamic programming assumes full knowledge of the MDP
2. It is used for planning in an MDP
3. For prediction:
   1. Input: MDP $<\mathcal{S, A, P, R, \gamma}>$ and policy $\pi$
   2. or: MRP $<\mathcal{S, A, P^\pi, R^\pi, \gamma}>$
   3. Output: value function $V_\pi$
4. Or for control:
   1. Input: MDP $<\mathcal{S, A, P, R, \gamma}>$
   2. Output: optimal value function $V_*$
   3. and: optimal policy $\pi_*$

### Other Applications of Dynamic Programming

Dynamic programming is used to solve many other problems, e.g.

1. Scheduling algorithms
2. String algorithms (e.g. sequence alignment)
3. Graph algorithms (e.g. shortest path algorithms)
4. Graphical models (e.g. Viterbi algorithm)
5. Bioinformatics (e.g. lattice models)

### Iterative Policy Evaluation (1)

1. Problem: evaluate a given policy $\pi$
2. Solution: iterative application of Bellman expectation backup
3. $V_1 \to V_2 \to \ldots \to V_\pi$
4. Using synchronous backups,
   1. At each iteration $k + 1$
   2. For all states $s \in \mathcal{S}$
   3. Update $V_{k+1}(s)$ from $V_k(s')$
   4. where $s_0$ is a successor state of $s$
5. We will discuss asynchronous backups later
6. Convergence to $V_\pi$ will be proven at the end of the lecture

### Iterative Policy Evaluation (2)

$$
\begin{align}
V_{k+1}(s) &= \sum_{a \in \mathcal{A}}\pi(a|s)(\mathcal{R_s^a} + \gamma \sum_{s' \in \mathcal{S} }\mathcal{P}_{ss'}^aV_k(s')) \\
V_{k+1} &= \mathcal{R_\pi} + \gamma \mathcal{P_\pi}V_k \\
\end{align}
$$

### How to Improve a Policy

1. Given a policy $\pi$
   1. Evaluate the policy $\pi$
      $V_\pi(s) = \mathbb{E}_\pi[\mathcal{R_{t+1}} + \gamma \mathcal{R_{t+2}} + \ldots|\mathcal{S_t = s}]$
   2. Improve the policy by acting greedily with respect to $V_\pi$
      $\pi' = greedy(V_\pi)$
2. In Small Gridworld improved policy was optimal, $\pi' = \pi_*$
3. In general, need more iterations of improvement/evaluation
4. But this process of policy iteration always converges to $\pi_*$

### Policy Iteration

1. Policy evaluation Estimate $V_\pi$
   Iterative policy evaluation
2. Policy improvement Generate $\pi' \geq \pi$
   Greedy policy improvement

### Policy Improvement

1. Consider a deterministic policy, $a = \pi(s)$
2. We can improve the policy by acting greedily $\pi'(s) = arg \, \underset{a \in \mathcal{A}}{max}\, Q_\pi(s, a)$
3. This improves the value from any state $s$ over one step,
   $Q_\pi(s, \pi'(s)) = \underset{a \in \mathcal{A}}{max} \, Q_\pi(s, a) \geq Q_\pi(s, \pi(s)) = V_\pi(s)$
4. It therefore improves the value function, $V_{\pi'}(s) \geq V_\pi(s)$

$$
\begin{aligned}
V_{\pi}(s) &\leq Q_{\pi}(s, \pi'(s)) = \mathbb{E}_{\pi'} [ \mathcal{R}_{t+1} + \gamma \mathrm{V}_{\pi}(\mathcal{S}_{t+1}) | \mathcal{S}_t = s] \\
&\leq \mathbb{E}_{\pi'} [ \mathcal{R}_{t+1} + \gamma Q_{\pi}(\mathcal{S}_{t+1}, \pi'(\mathcal{S}_{t+1})) | \mathcal{S}_t = s] \\
&\leq \mathbb{E}_{\pi'} [ \mathcal{R}_{t+1} + \gamma \mathcal{R}_{t+2} + \gamma^2 Q_{\pi}(\mathcal{S}_{t+2}, \pi'(\mathcal{S}_{t+2})) | \mathcal{S}_t = s] \\
&\leq \mathbb{E}_{\pi'} [ \mathcal{R}_{t+1} + \gamma \mathcal{R}_{t+2} + \cdots | \mathcal{S}_t = s] = V_{\pi'}(s).
\end{aligned}
$$

### Policy Improvement (2)

1. If improvements stop, $Q_\pi(s, \pi'(s)) = \underset{a \in \mathcal{A}}{max} \, Q_\pi(s, a) = Q_\pi(s, \pi(s)) = V_\pi(s)$
2. Then the Bellman optimality equation has been satisfied $V_\pi(s) = \underset{a \in \mathcal{A}}{max}\, Q_\pi(s, a)$
3. Therefore $V_\pi(s) = V_*(s)$ for all $s \in \mathcal{S}$
4. so $\pi$ is an optimal policy

### Modified Policy Iteration

1. Does policy evaluation need to converge to $V_\pi$?
2. Or should we introduce a stopping condition
   e.g. $\epsilon$-convergence of value function
3. Or simply stop after $k$ iterations of iterative policy evaluation?
4. For example, in the small gridworld $k=3$ was sufficient to achieve optimal policy
5. Why not update policy every iteration?  i.e. stop after $k=1$
   This is equivalent to value iteration (next section)

### Generalised Policy Iteration

1. Policy evaluation Estimate $V_\pi$
   Any policy evaluation algorithm
2. Policy improvement Generate $\pi' \geq \pi$
   Any policy improvement algorithm

### Principle of Optimality

Any optimal policy can be subdivided into two components:

1. An optimal first action $A_*$
2. Followed by an optimal policy from successor state $S'$

#### theorem (Principle of Optimality)

A policy $\pi(a|s)$ achieves the optimal value from state $s$, $V_\pi(s) = V_*(s)$, if and only if :

1. For any state $s’$ reachable from $s$
2. $\pi$ achieves the optimal value from state $s'$, $V_\pi(s') = V_*(s')$

### Deterministic Value Iteration

1. If we know the solution to subproblems $V_*(s')$
2. Then solution $V_*(s)$ can be found by one-step lookahead $V_*(s) \gets \underset{a \in \mathcal{A}}{max}\, \mathcal{R_s^a} + \gamma \sum_{s' \in \mathcal{S}}\mathcal{P_{ss'}^a}V_*(s')$
3. The idea of value iteration is to apply these updates iteratively
4. Intuition: start with final rewards and work backwards
5. Still works with loopy, stochastic MDPs

### Value Iteration

1. Problem: find optimal policy $\pi$
2. Solution: iterative application of Bellman optimality backup
3. $V_1 \to V_2 \to \ldots \to V_*$
4. Using synchronous backups
   1. At each iteration $k+1$
   2. For all states $s \in \mathcal{S}$
   3. Update $V_{k+1}(s)$ from $V_k(s')$
5. Convergence to $V_*$ will be proven later
6. Unlike policy iteration, there is no explicit policy
7. Intermediate value functions may not correspond to any policy

### Value Iteration (2)

$$
\begin{align}
V_{k+1}(s) &= \underset{a \in \mathcal{A}}{max}\,(\mathcal{R_s^a} + \gamma \sum_{s' \in \mathcal{S} }\mathcal{P}_{ss'}^aV_k(s')) \\
V_{k+1} &= \underset{a \in \mathcal{A}}{max}\, \mathcal{R^a} + \gamma \mathcal{P^a}V_k \\
\end{align}
$$

### Example of Value Iteration in Practice

Http://www.cs.ubc.ca/~poole/demos/mdp/vi.html

### Synchronous Dynamic Programming Algorithms

|  problem  | Bellman Equation                                         |          Algorithm          |
| :--------: | -------------------------------------------------------- | :-------------------------: |
| Prediction | Bellman Expectation Equation                             | Iterative policy evaluation |
|  Control  | Bellman Expectation Equation + Greedy Policy Improvement |      Policy iteration      |
|  Control  | Bellman Optimality Equation                              |       Value iteration       |

1. Algorithms are based on state-value function $V_\pi(s)$ or $V_*(s)$
2. Complexity $\mathcal{O(mn^2)}$ per iteration, for $m$ actions and $n$ states
3. Could also apply to action-value function $Q_\pi(s, a)$ or $Q_*(s, a)$
4. Complexity $\mathcal{O(m^2n^2)}$ per iteration

### Asynchronous Dynamic Programming

1. DP methods described so far used synchronous backups
2. i.e. all states are backed up in parallel
3. Asynchronous DP backs up states individually, in any order
4. For each selected state, apply the appropriate backup
5. Can significantly reduce computation
6. Guaranteed to converge if all states continue to be selected
   Three simple ideas for asynchronous dynamic programming:
7. In-place dynamic programming
8. Prioritized sweeping
9. Real-time dynamic programming

### In-Place Dynamic Programming

1. Synchronous value iteration stores two copies of value function for all $s$ in $\mathcal{S}$

$$
\begin{align}
  V_{\text{new}}(s) \leftarrow \max_{a \in \mathcal{A}}& \left( \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^{a}  V_{\text{old}}(s') \right) \\
 &V_{\text{old}} \leftarrow V_{\text{new}} \\
\end{align}
$$

2. In-place value iteration only stores one copy of value function for all $s$ in $\mathcal{S}$

$$
V(s) \leftarrow \max_{a \in \mathcal{A}} \left( \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^{a} V(s') \right)
$$

### Prioritised Sweeping

1. Use magnitude of Bellman error to guide state selection, e.g.

$$
|\max_{a \in \mathcal{A}} \left( \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a V(s') \right) - V(s)|
$$

2. Backup the state with the largest remaining Bellman error
3. Update Bellman error of a↵ected states after each backup
4. Requires knowledge of reverse dynamics (predecessor states)
5. Can be implemented efficiently by maintaining a priority queue

### Real-Time Dynamic Programming

1. Idea: only states that are relevant to agent
2. Use agent’s experience to guide the selection of states
3. After each time-step $\mathcal{S_t, A_t, \mathcal{R_{t+1}}}$
4. Backup the state $\mathcal{S_t}$

$$
V(\mathcal{S_t}) \gets \max_{a \in \mathcal{A}} \left( \mathcal{R}_\mathcal{S_t}^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{\mathcal{S_t}s'}^a V(s') \right)
$$

### Full-Width Backups

1. DP uses full-width backups
2. For each backup (sync or async)
   1. Every successor state and action is considered
   2. Using knowledge of the MDP transitions and reward function
3. DP is effective for medium-sized problems (millions of states)
4. For large problems DP suffers Bellman's curse of dimensionality
   Number of states $n = |{\mathcal{S}}|$ grows exponentially with number of state variables
5. Even one backup can be too expensive

### Sample Backups

1. In subsequent lectures we will consider sample backups
2. Using sample rewards and sample transitions $<\mathcal{S, A, R, S'}>$
3. Instead of reward function $\mathcal{R}$ and transition dynamics $\mathcal{P}$
4. Advantages:
   1. Model-free: no advance knowledge of MDP required
   2. Breaks the curse of dimensionality through sampling
   3. Cost of backup is constant, independent of $n = |{\mathcal{S}}|$

### Approximate Dynamic Programming

1. Approximate the value function
2. Using a function approximator $\hat{V}(s, w)$
3. Apply dynamic programming to $\hat{V}(\cdot|w)$
4. e.g. Fitted Value Iteration repeats at each iteration $k$,
   1. Sample states $\tilde{\mathcal{S}} \subset \mathcal{S}$
   2. For each state $s \in \tilde{\mathcal{S}}$, estimate target value using Bellman optimality equation, $\widetilde{V}_k(s) = \max_{a \in \mathcal{A}} \left( \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a \widehat{V}(s', w_k) \right)$
   3. Train next value function $\hat{V}(\cdot|w_{k+1})$ using targets $\left\{ \left\langle s, \widetilde{\mathrm{V}}_k(s) \right\rangle \right\}$

### Some Technical Questions

1. How do we know that value iteration converges to $V_*$?
2. Or that iterative policy evaluation converges to $V_\pi$?
3. And therefore that policy iteration converges to $V_*$?
4. Is the solution unique?
5. How fast do these algorithms converge?
6. These questions are resolved by contraction mapping theorem

### Value Function Space

1. Consider the vector space $\mathcal{V}$ over value functions
2. There are $|{\mathcal{S}}|$ dimensions
3. Each point in this space fully specifies a value function $V(s)$
4. What does a Bellman backup do to points in this space?
5. We will show that it brings value functions closer
6. And therefore the backups must converge on a unique solution

### Value Function $\infty - Norm$

1. We will measure distance between state-value functions $u$ and $v$ by the $\infty - Norm$
2. i.e. the largest difference between state values, $|| \mathrm{u} - \mathrm{v} ||_{\infty} = \underset{{s \in \mathcal{S}}}{max}\, || \mathrm{u}(s) - \mathrm{v}(s) ||$

### Bellman Expectation Backup is a Contraction

1. Define the Bellman expectation backup operator $T^\pi$

$$
T^\pi(V) = \mathcal{R^\pi} + \gamma \mathcal{P^\pi}V
$$

2. This operator is a $\gamma$-contraction, i.e. it makes value functions closer by at least $\gamma$,

$$
\begin{align*}
\| T^\pi(\mathrm{u}) - T^\pi(\mathrm{v}) \|_\infty
&= \left\| \bigl( \mathcal{R}^\pi + \gamma \mathcal{P}^\pi \mathrm{u} \bigr) - \bigl( \mathcal{R}^\pi + \gamma \mathcal{P}^\pi \mathrm{v} \bigr) \right\|_\infty \\
&= \left\| \gamma \mathcal{P}^\pi (\mathrm{u} - \mathrm{v}) \right\|_\infty \\
&\leq \| \gamma \mathcal{P}^\pi \|_\infty \; \| \mathrm{u} - \mathrm{v} \|_\infty \\
&\leq \gamma \, \| \mathrm{u} - \mathrm{v} \|_\infty
\end{align*}
$$

### Contraction Mapping Theorem

#### theorem (Contraction Mapping Theorem)

For any metric space $\mathcal{V}$ that is complete (i.e. closed) under an operator $T(V)$, where $T$ is a -contraction,

1. $T$ converges to a unique fixed point
2. At a linear convergence rate of $\gamma$

### Convergence of Iter. Policy Evaluation and Policy Iteration

1. The Bellman expectation operator $T^\pi$ has a unique fixed point
2. $V_\pi$ is a fixed point of $T^\pi$ (by Bellman expectation equation)
3. By contraction mapping theorem
4. Iterative policy evaluation converges on $V_\pi$
5. Policy iteration converges on $V_*$

### Bellman Optimality Backup is a Contraction

Define the Bellman optimality backup operator $T_*$,

$$
T^*(\mathrm{V}) = \max_{a \in \mathcal{A}} \mathcal{R}^a + \gamma \mathcal{P}^a \mathrm{V}
$$

This operator is a $\gamma$-contraction, i.e. it makes value functions closer by at least $\gamma$ (similar to previous proof)

$$
\| T^*(\mathrm{u}) - T^*(\mathrm{v}) \|_\infty \leq \gamma \| \mathrm{u} - \mathrm{v} \|_\infty
$$

### Convergence of Value Iteration

1. The Bellman optimality operator $T_*$ has a unique fixed point
2. $V_*$ is a fixed point of $T_*$ (by Bellman optimality equation)
3. By contraction mapping theorem
4. Value iteration converges on $V_*$

## Lecture 4: Model-Free Prediction

### outline:

1. Introduction
2. Monte-Carlo learning
3. Temporal-Difference learning
4. TD $(\lambda)$

### Model-Free Reinforcement Learning

1. Last lecture:
   1. Planning by dynamic programming
   2. Solve a known MDP
2. This lecture:
   1. Model-free prediction
   2. Estimate the value function of an unknown MDP
3. Next lecture:
   1. Model-free control
   2. Optimize the value function of an unknown MDP

### Monte-Carlo Reinforcement Learning

1. MC methods learn directly from episodes of experience
2. MC is model-free: no knowledge of MDP transitions/rewards
3. MC learns from complete episodes: no bootstrapping
4. MC uses the simplest possible idea: value = mean return
5. Caveat: can only apply MC to episodic MDPs
   All episodes must terminate

### Monte-Carlo Policy Evaluation

1. Goal: learn $V_\pi$ from episodes of experience under policy $\pi$

$$
\mathcal{S_1}, A_1, \mathcal{R_2}, \ldots, \mathcal{S_k} \sim \pi
$$

2. Recall that the return is the total discounted reward:

$$
G_t = \mathcal{R_{t+1}} + \gamma\mathcal{R_{t+2}} + \ldots + \gamma^{T-1} \mathcal{R}_T
$$

3. Recall that the value function is the expected return:

$$
V_\pi(s) = \mathbb{E}[G_t|\mathcal{S_t} = s]
$$

4. Monte-Carlo policy evaluation uses empirical mean return instead of expected return

### First-Visit Monte-Carlo Policy Evaluation

1. To evaluate state $s$
2. The first time-step $t$ that state $s$ is visited in an episode
3. Increment counter $N(s) \gets N(s) + 1$
4. Increment total return $\mathcal{S}(s) \gets \mathcal{S}(s) + G_t$
5. Value is estimated by mean return $V(s) = \mathcal{S}(s) / N(s)$
6. By law of large numbers, $V(s) \to V_\pi(a)$ as $N(s) \to \infty$

### Every-Visit Monte-Carlo Policy Evaluation

1. To evaluate state $s$
2. Every time-step $t$ that state $s$ is visited in an episode
3. Increment counter $N(s) \gets N(s) + 1$
4. Increment total return $\mathcal{S}(s) \gets \mathcal{S}(s) + G_t$
5. Value is estimated by mean return $V(s) = \mathcal{S}(s) / N(s)$
6. Again, $V(s) \to V_\pi(s)$ as $N(s) \to \infty$

### Incremental Mean

The mean $\mu_1, \mu_2, \ldots$ of a sequence $x_1, x_2, \ldots$ can be computed incrementally

$$
\begin{align*}
\mu_k &= \frac{1}{k} \sum_{j=1}^k x_j \\
&= \frac{1}{k} \left( x_k + \sum_{j=1}^{k-1} x_j \right) \\
&= \frac{1}{k} \bigl( x_k + (k-1)\mu_{k-1} \bigr) \\
&= \mu_{k-1} + \frac{1}{k} (x_k - \mu_{k-1})
\end{align*}
$$

### Incremental Monte-Carlo Updates

1. Update $V(s)$ incrementally after episode $\mathcal{S_1}, A_1, \mathcal{R_2}, \ldots, \mathcal{S}_T$
2. For each state $\mathcal{S}_t$ with return $G_t$

$$
\begin{align*}
N(\mathcal{S_t}) &\gets N(\mathcal{S_t}) + 1 \\
V(\mathcal{S_t}) &\gets + {\frac{1}{N(\mathcal{S_t})}}(G_t - V(\mathcal{S_t})) \\
\end{align*}
$$

3. In non-stationary problems, it can be useful to track a running mean, i.e. forget old episodes

$$
V(\mathcal{S_t}) \gets V(\mathcal{S_t}) + \alpha (G_t - V(\mathcal{S_t}))
$$

### Temporal-Difference Learning

1. TD methods learn directly from episodes of experience
2. TD is model-free: no knowledge of MDP transitions/rewards
3. TD learns from incomplete episodes, by bootstrapping
4. TD updates a guess towards a guess

### MC and TD

1. Goal: learn $V_\pi$ online from experience under policy $\pi$
2. Incremental every-visit Monte-Carlo
   Update value $V(\mathcal{S_t})$ toward actual return $G_t$

   $$
   V(\mathcal{S_t}) \gets V(\mathcal{S_t}) + \alpha (G_t - V(\mathcal{S_t}))
   $$
3. Simplest temporal-difference learning algorithm: $TD(0)$

   1. Update value $V(\mathcal{S_t})$ toward estimated return $\mathcal{R_{t+1}} + \gamma V (\mathcal{S_{t+1}})$

   $$
   V(\mathcal{S_t}) \gets V(\mathcal{S_t}) + \alpha (\mathcal{R_{t+1}} + \gamma V(\mathcal{S_{t+1}}) - V(\mathcal{S_t}))
   $$

   2. $\mathcal{R}_{t+1} + V (\mathcal{S}_{t+1})$ is called the TD target
   3. $\delta_t = \mathcal{R}_{t+1} + \gamma V (\mathcal{S}_{t+1}) - V (\mathcal{S_t} )$ is called the TD error

### Advantages and Disadvantages of MC vs TD (1)

1. TD can learn before knowing the final outcome
   1. TD can learn online after every step
   2. MC must wait until end of episode before return is known
2. TD can learn without the final outcome
   1. TD can learn from incomplete sequences
   2. MC can only learn from complete sequences
   3. TD works in continuing (non-terminating) environments
   4. MC only works for episodic (terminating) environments

### Bias/Variance Trade-Off

1. Return $G_t = \mathcal{R_{t+1}} + \gamma \mathcal{R_{t+2}} + \ldots + \gamma^{T-1}\mathcal{R}_{T}$ is unbiased estimate of $V_\pi(\mathcal{S_t})$
2. True TD target $\mathcal{R_{t+1}} + \gamma V_\pi(\mathcal{S_t})$ is unbiased estimate of $V_\pi(\mathcal{S_t})$
3. TD target $\mathcal{R_{t+1}} + \gamma V(\mathcal{S_t})$ is biased estimate of $V_\pi(\mathcal{S_t})$
4. TD target is much lower variance than the return:
   1. Return depends on many random actions, transitions, rewards
   2. TD target depends on one random action, transition, reward

### Advantages and Disadvantages of MC vs TD (2)

1. MC has high variance, zero bias
   1. Good convergence properties
   2. (even with function approximation)
   3. Not very sensitive to initial value
   4. Very simple to understand and use
2. TD has low variance, some bias
   1. Usually more efficient than MC
   2. $TD(0)$ converges to $V_\pi(s)$
   3. (but not always with function approximation)
   4. More sensitive to initial value

### Batch MC and TD

1. MC and TD converge: $V(s) \to V_\pi(s)$ as experience $\to \infty$
2. But what about batch solution for finite experience?

$$
\begin{align*}
&s_1^1, a_1^1, r_2^1, \dots, s_{T_1}^1 \\
&\vdots \\
&s_1^K, a_1^K, r_2^K, \dots, s_{T_K}^K
\end{align*}
$$

    1. e.g. Repeatedly sample episode$k \in [1,K]$
		2. Apply MC or $TD(0)$ to episode $k$

### Certainty Equivalence

1. MC converges to solution with minimum mean-squared error
   Best fit to the observed returns $\sum_{k=1}^K \sum_{t=1}^{T_k}(G_t^k - V(s_t^k))^2$
2. $TD(0)$ converges to solution of max likelihood Markov model
   Solution to the MDP $<\mathcal{S, A, \widehat{P}, \widehat{R}, \gamma}>$ that best fits the data
   $$
   \begin{align*}
   \widehat{\mathcal{P}}_{s,s'}^a = \frac{1}{N(s,a)} \sum_{k=1}^{K} \sum_{t=1}^{T_k} \mathbf{1}(s_t^k = s,\ a_t^k = a,\ s_{t+1}^k = s')]\\
   \widehat{\mathcal{R}}_s^a = \frac{1}{N(s,a)} \sum_{k=1}^{K} \sum_{t=1}^{T_k} \mathbf{1}(s_t^k = s,\ a_t^k = a) \, r_t^k\\
   \end{align*}
   $$

### Advantages and Disadvantages of MC vs TD (3)

1. TD exploits Markov property
   Usually more efficient in Markov environments
2. MC does not exploit Markov property
   Usually more effective in non-Markov environments

### Monte-Carlo Backup

$$
V(\mathcal{S_t}) \gets V(\mathcal{S_t}) + \alpha (G_t - V(\mathcal{S_t}))
$$

### Temporal-Difference Backup

$$
V(\mathcal{S_t}) \gets V(\mathcal{S_t}) + \alpha (\mathcal{R_{t+1}} +\gamma V(\mathcal{S_{t+1}}) - V(\mathcal{S_t}))
$$

### Dynamic Programming Backup

$$
V(\mathcal{S_t}) \gets \mathbb{E}[\mathcal{R_{t+1}} +\gamma V(\mathcal{S_{t+1}})]
$$

### Bootstrapping and Sampling

1. Bootstrapping: update involves an estimate
   1. MC does not bootstrap
   2. DP bootstraps
   3. TD bootstraps
2. Sampling: update samples an expectation
   1. MC samples
   2. DP does not sample
   3. TD samples

### N-step TD

#### N-step prediction

   Let TD target look $n$ steps into the future

#### N-step return

1. Consider the following n-step returns for $n = 1, 2, \ldots, \infty$:

$$
\begin{array}{ccl}
n=1 & \text{(TD)} & G_t^{(1)} = \mathcal{R}_{t+1} + \gamma \mathrm{V}(\mathcal{S}_{t+1}) \\[6pt]
n=2 & \text{(TD)}& G_t^{(2)} = \mathcal{R}_{t+1} + \gamma \mathcal{R}_{t+2} + \gamma^2 \mathrm{V}(\mathcal{S}_{t+2}) \\[6pt]
\vdots & & \vdots \\[6pt]
n=\infty & \text{(MC)} & G_t^{(\infty)} = \mathcal{R}_{t+1} + \gamma \mathcal{R}_{t+2} + \ldots + \gamma^{T-1} \mathcal{R}_T
\end{array}
$$

2. Define the n-step return

$$
G_t^{(n)} = \mathcal{R}_{t+1} + \gamma \mathcal{R}_{t+2} + \ldots + \gamma^{n-1} \mathcal{R}_{t+n} + \gamma^n \mathrm{V}(\mathcal{S}_{t+n})
$$

3. n-step temporal-difference learning

$$
\mathrm{V}(\mathcal{S}_t) \gets \mathrm{V}(\mathcal{S}_t) + \alpha \left( G_t^{(n)} - \mathrm{V}(\mathcal{S}_t) \right)
$$

#### Averaging n-Step Returns

1. We can average n-step returns over different $n$
2. e.g. average the 2-step and 4-step returns $\frac{1}{2}G^(2) + \frac{1}{2}G^(4)$
3. Combines information from two different time-steps
4. Can we efficiently combine information from all time-steps?

### forward view of $TD(\lambda)$

#### $\lambda$ - return

1. The -return $G_t^\lambda$ combines all n-step returns $G_t^{(n)}$
2. Using weight $(1-\lambda)\lambda^{n-1}$, $G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}$
3. Forward-view $TD(\lambda)$, $\mathrm{V}(\mathcal{S}_t) \gets \mathrm{V}(\mathcal{S}_t) + \alpha \left( G_t^{\lambda} - \mathrm{V}(\mathcal{S}_t) \right)$

#### $TD(\lambda)$ weighting function

$$
G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}
$$

#### Forward-view $TD(\lambda)$

1. Update value function towards the $\lambda$-return
2. Forward-view looks into the future to compute $G_t^\lambda$
3. Like MC, can only be computed from complete episodes

### backward view of $TD(\lambda)$

#### Backward view $TD(\lambda)$

1. Forward view provides theory
2. Backward view provides mechanism
3. Update online, every step, from incomplete sequences

#### Eligibility Traces

1. Credit assignment problem: did bell or light cause shock?
2. Frequency heuristic: assign credit to most frequent states
3. Recency heuristic: assign credit to most recent states
4. Eligibility traces combine both heuristics

$$
E_0(s) = 0,
E_t(s) = \gamma \lambda E_{t-1}(s) + \mathbf{1}(\mathcal{S_t} = s)
$$

#### Backward View $TD(\lambda)$

1. Keep an eligibility trace for every state $s$
2. Update value $V(s)$ for every state $s$
3. In proportion to TD-error $\delta_t$ and eligibility trace $E_t(s)$

$$
\begin{align*}
\delta_t = \mathcal{R}_{t+1} + \gamma \mathrm{V}(\mathcal{S}_{t+1}) - \mathrm{V}(\mathcal{S}_t)\\
\mathrm{V}(s) \leftarrow \mathrm{V}(s) + \alpha \, \delta_t \, E_t(s)
\end{align*}
$$

### Relationship Between Forward and Backward TD

#### $TD(\lambda)$ and $TD(0)$

1. When $\lambda = 0$, only current state is updated

$$
\begin{align*}
E_t(s) &= \mathbf{1}(\mathcal{S_t} = s)\\
\mathrm{V}(s) &\leftarrow \mathrm{V}(s) + \alpha \, \delta_t \, E_t(s)
\end{align*}
$$

2. This is exactly equivalent to $TD(0)$ update

$$
\mathrm{V}(\mathcal{S_t}) \leftarrow \mathrm{V}(\mathcal{S_t}) + \alpha \, \delta_t
$$

#### $TD(\lambda)$ and MC

1. When $\lambda =1$, credit is deferred until end of episode
2. Consider episodic environments with offline updates
3. Over the course of an episode, total update for $TD(1)$ is the same as total update for MC

##### theorem

The sum of offline updates is identical for forward-view and backward-view $TD(\lambda)$

$$
\sum_{t=1}^T \alpha \, \delta_t \, E_t(s) = \sum_{t=1}^T \alpha \left( G_t^\lambda - \mathrm{V}(\mathcal{S}_t) \right) \mathbf{1}(\mathcal{S}_t = s)
$$

### Forward and Backward Equivalence

#### MC and $TD(1)$

1. Consider an episode where $s$ is visited once at time-step $k$
2. $TD(1)$ eligibility trace discounts time since visit
   $$
   \begin{align*}
   E_t(s) = \gamma E_{t-1}(s) + \mathbf{1}(\mathcal{S}_t = s) = \begin{cases} 0, & \text{if } t < k, \\\gamma^{t-k}, & \text{if } t \geq k.\end{cases}
   \end{align*}
   $$
3. $TD(1)$ updates accumulate error online
   $$
   \sum_{t=1}^{T-1} \alpha \, \delta_t \, E_t(s) = \alpha \sum_{t=k}^{T-1} \gamma^{t-k} \delta_t = \alpha \bigl( G_k - \mathrm{V}(\mathcal{S}_k) \bigr)
   $$
4. By end of episode it accumulates total error
   $$
   \delta_k + \gamma \delta_{k+1} + \gamma^2 \delta_{k+2} + \dots + \gamma^{T-1-k} \delta_{T-1}
   $$

#### Telescoping in $TD(1)$

When  $\lambda = 1$, sum of  TD errors telescopes into MC error,

$$
\begin{aligned}
&\delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+2} + \dots + \gamma^{T-1-t} \delta_{T-1} \\
&= \mathcal{R}_{t+1} + \gamma \mathrm{V}(\mathcal{S}_{t+1}) - \mathrm{V}(\mathcal{S}_t) \\
&\quad + \gamma \mathcal{R}_{t+2} + \gamma^2 \mathrm{V}(\mathcal{S}_{t+2}) - \gamma \mathrm{V}(\mathcal{S}_{t+1}) \\
&\quad + \gamma^2 \mathcal{R}_{t+3} + \gamma^3 \mathrm{V}(\mathcal{S}_{t+3}) - \gamma^2 \mathrm{V}(\mathcal{S}_{t+2}) \\
&\quad \vdots \\
&\quad + \gamma^{T-1-t} \mathcal{R}_T + \gamma^{T-t} \mathrm{V}(\mathcal{S}_T) - \gamma^{T-1-t} \mathrm{V}(\mathcal{S}_{T-1}) \\
&= \mathcal{R}_{t+1} + \gamma \mathcal{R}_{t+2} + \gamma^2 \mathcal{R}_{t+3} + \dots + \gamma^{T-1-t} \mathcal{R}_T - \mathrm{V}(\mathcal{S}_t) \\
&= G_t - \mathrm{V}(\mathcal{S}_t)
\end{aligned}
$$

#### $TD(\lambda)$ and $TD(1)$

1. $TD(1)$ is roughly equivalent to every-visit Monte-Carlo
2. Error is accumulated online, step-by-step
3. If value function is only updated offline at end of episode
4. Then total update is exactly the same as MC

#### Telescoping in $TD(\lambda)$

For general $\lambda$, TD errors also telescope to $\lambda$-error, $G_t^\lambda - V(\mathcal{S_t})$

$$
\begin{align*}
G_t^\lambda - V(\mathcal{S}_t) &= -V(\mathcal{S}_t) + (1-\lambda)\lambda^0 \bigl(\mathcal{R}_{t+1} + \gamma V(\mathcal{S}_{t+1})\bigr) \\
&\quad\qquad\qquad + (1-\lambda)\lambda^1 \bigl(\mathcal{R}_{t+1} + \gamma \mathcal{R}_{t+2} + \gamma^2 V(\mathcal{S}_{t+2})\bigr) \\
&\quad\qquad\qquad + (1-\lambda)\lambda^2 \bigl(\mathcal{R}_{t+1} + \gamma \mathcal{R}_{t+2} + \gamma^2 \mathcal{R}_{t+3} + \gamma^3 V(\mathcal{S}_{t+3})\bigr) \\
&\quad\qquad\qquad + \dotsb \\
&= -V(\mathcal{S}_t) + (\gamma\lambda)^0 \bigl(\mathcal{R}_{t+1} + \gamma V(\mathcal{S}_{t+1}) - \gamma\lambda V(\mathcal{S}_{t+1})\bigr) \\
&\quad\qquad\qquad + (\gamma\lambda)^1 \bigl(\mathcal{R}_{t+2} + \gamma V(\mathcal{S}_{t+2}) - \gamma\lambda V(\mathcal{S}_{t+2})\bigr) \\
&\quad\qquad\qquad + (\gamma\lambda)^2 \bigl(\mathcal{R}_{t+3} + \gamma V(\mathcal{S}_{t+3}) - \gamma\lambda V(\mathcal{S}_{t+3})\bigr) \\
&\quad\qquad\qquad + \dotsb \\
&= (\gamma\lambda)^0 \bigl(\mathcal{R}_{t+1} + \gamma V(\mathcal{S}_{t+1}) - V(\mathcal{S}_t)\bigr) \\
&\quad + (\gamma\lambda)^1 \bigl(\mathcal{R}_{t+2} + \gamma V(\mathcal{S}_{t+2}) - V(\mathcal{S}_{t+1})\bigr) \\
&\quad + (\gamma\lambda)^2 \bigl(\mathcal{R}_{t+3} + \gamma V(\mathcal{S}_{t+3}) - V(\mathcal{S}_{t+2})\bigr) \\
&\quad + \dotsb \\
&= \delta_t + \gamma\lambda \delta_{t+1} + (\gamma\lambda)^2 \delta_{t+2} + \dotsb
\end{align*}
$$

#### Forwards and Backwards $TD(\lambda)$

1. Consider an episode where $s$ is visited once at time-step $k$
2. $TD(\lambda)$ eligibility trace discounts time since visit
   $$
   \begin{align*}
   E_t(s) = \gamma \lambda E_{t-1}(s) + \mathbf{1}(\mathcal{S}_t = s) = \begin{cases} 0, & \text{if } t < k, \\(\gamma\lambda)^{t-k}, & \text{if } t \geq k.\end{cases}
   \end{align*}
   $$
3. Backward $TD(\lambda)$ updates accumulate error online
   $$
   \sum_{t=1}^{T} \alpha \, \delta_t \, E_t(s) = \alpha \sum_{t=k}^{T} (\gamma\lambda)^{t-k} \delta_t = \alpha \bigl( G_k^\lambda - \mathrm{V}(\mathcal{S}_k) \bigr)
   $$
4. By end of episode it accumulates total error for $\lambda$-return
5. For multiple visits to $s$, $E_t(s)$ accumulates many errors

#### Offline Equivalence of Forward and Backward TD

Offline updates

1. Updates are accumulated within episode
2. but applied in batch at the end of episode

#### Onine Equivalence of Forward and Backward TD

Online updates

1. $TD(\lambda)$ updates are applied online at each step within episode
2. Forward and backward-view $TD(\lambda)$ are slightly different
3. NEW: Exact online $TD(\lambda)$ achieves perfect equivalence
4. By using a slightly different form of eligibility trace
5. Sutton and von Seijen, ICML 2014

#### Summary of Forward and Backward $TD(\lambda)$

$$
\begin{array}{|c|c|c|c|}
\hline
\text{Offline updates} & \lambda = 0 & \lambda \in (0,1) & \lambda = 1 \\
\hline
\text{Backward view} & \text{TD}(0) & \text{TD}(\lambda) & \text{TD}(1) \\
& \| & \| & \| \\
\text{Forward view} & \text{TD}(0) & \text{Forward TD}(\lambda) & \text{MC} \\
\hline
\text{Online updates} & \lambda = 0 & \lambda \in (0,1) & \lambda = 1 \\
\hline
\text{Backward view} & \text{TD}(0) & \text{TD}(\lambda) & \text{TD}(1) \\
& \| & \nmid & \nmid \\
\text{Forward view} & \text{TD}(0) & \text{Forward TD}(\lambda) & \text{MC} \\
& \| & \| & \| \\
\text{Exact Online} & \text{TD}(0) & \text{Exact Online TD}(\lambda) & \text{Exact Online TD}(1) \\
\hline
\end{array}
$$

$=$ here indicates equivalence in total update at end of episode

## Lecture 5: Model-Free Control

### outline:

1. Introduction
2. On-Policy Monte-Carlo Control
3. On-Policy Temporal-Difference Learning
4. Off-Policy Learning
5. Summary

### Model-Free Reinforcement Learning

1. Last lecture:
   1. Model-free prediction
   2. Estimate the value function of an unknown MDP
2. This lecture:
   1. Model-free control
   2. Optimize the value function of an unknown MDP

#### Uses of Model-Free Control

Some example problems that can be modelled as MDPs
For most of these problems, either:

1. MDP model is unknown, but experience can be sampled
2. MDP model is known, but is too big to use, except by samples
   Model-free control can solve these problem

#### On and Off-Policy Learning

1. On-policy learning:
   “Learn on the job”
   Learn about policy $\pi$ from experience sampled from $\pi$
2. Off-policy learning
   “Look over someone’s shoulder”
   Learn about policy $\pi$ from experience sampled from $\mu$

### Generalized Policy Iteration

#### Generalized Policy Iteration (Refresher)

Policy evaluation Estimate $V_\pi$, e.g. Iterative policy evaluation
Policy improvement Generate $\pi' \geq \pi$, e.g. Greedy policy improvement

#### Generalized Policy Iteration With Monte-Carlo Evaluation

Policy evaluation Monte-Carlo policy evaluation, $V = V_\pi$?
Policy improvement Greedy policy improvement?

#### Model-Free Policy Iteration Using Action-Value Function

1. Greedy policy improvement over $V(s)$ requires model of MDP
   $\pi'(s) = \arg\max_{a \in \mathcal{A}} \mathcal{R}_s^a + \mathcal{P}_{ss'}^a V(s')$
2. Greedy policy improvement over $Q(s,a)$ is model-free
   $\pi'(s) = \arg\max_{a \in \mathcal{A}} Q(s, a)$

#### Generalised Policy Iteration with Action-Value Function

1. Policy evaluation Monte-Carlo policy evaluation, $Q =Q_\pi$
2. Policy improvement Greedy policy improvement?

### Exploration

#### $\epsilon$-Greedy exploration

1. Simplest idea for ensuring continual exploration
2. All m actions are tried with non-zero probability
3. With probability $1-\epsilon$ choose the greedy action
4. With probability $\epsilon$ choose an action at random
   $$
   \pi(a|s) = 
   \begin{cases} \frac{\epsilon}{m} + 1 - \epsilon & \text{if } a^* = \arg\max_{a \in \mathcal{A}} Q(s, a) \\
   \frac{\epsilon}{m} & \text{otherwise}\end{cases}
   $$

#### $\epsilon$-Greedy policy improvement

##### theorem

For any $\epsilon$-greedy policy $\pi$, the $\epsilon$-greedy policy $\pi’$ with respect to $Q_\pi$ is an improvement, $V_\pi'(s) \geq V_\pi(s)$

$$
\begin{align*}
Q_{\pi}(s, \pi'(s)) &= \sum_{a \in \mathcal{A}} \pi'(a|s) Q_{\pi}(s, a) \\
&= \frac{\epsilon}{m} \sum_{a \in \mathcal{A}} Q_{\pi}(s, a) + (1 - \epsilon) \max_{a \in \mathcal{A}} Q_{\pi}(s, a) \\
&\geq \frac{\epsilon}{m} \sum_{a \in \mathcal{A}} Q_{\pi}(s, a) + (1 - \epsilon) \sum_{a \in \mathcal{A}} \frac{\pi(a|s) - \epsilon/m}{1 - \epsilon} Q_{\pi}(s, a) \\
&= \sum_{a \in \mathcal{A}} \pi(a|s) Q_{\pi}(s, a) = V_{\pi}(s)
\end{align*}
$$

Therefore from policy improvement theorem, $V_{\pi'}(s) \geq V_{\pi}(s)$

#### Monte-Carlo Policy Iteration

Policy evaluation Monte-Carlo policy evaluation, $Q=Q_\pi$
Policy improvement $\epsilon$-greedy policy improvement

#### Monte-Carlo Control

Every episode:
Policy evaluation Monte-Carlo policy evaluation, $Q \approx Q_\pi$
Policy improvement $\epsilon$-greedy policy improvement

### GILE

#### Definition

Greedy in the Limit with Infinite Exploration (GLIE)

1. All state-action pairs are explored infinitely many times, $\underset{k\to\infty}{lim}\,N_k(s,a)=\infty$
2. The policy converges on a greedy policy,
   $\underset{k\to\infty}{lim}\pi_k(a|s) = \mathbf{1}(a=arg\,\underset{a' \in \mathcal{A}}{max}\,Q_k(s,a'))$
   For example, $\epsilon$-greedy is GLIE if $\epsilon$ reduces to zero at $\epsilon_k=1$

#### GLIE Monte-Carlo Control

1. Sample kth episode using $\pi : \{\mathcal{S_1}, A_1, \mathcal{R_2}, \ldots, \mathcal{S}_T\}$
2. For each state $\mathcal{S_t}$ and action $A_t$ in the episode,

$$
\begin{align*}
N(\mathcal{S_t} , A_t)&\gets N(\mathcal{S_t} , A_t ) + 1\\
Q(\mathcal{S_t},A_t) &\gets Q(\mathcal{S_t} , A_t ) + \frac{1}{N(\mathcal{S_t} , A_t )} (G_t - Q(\mathcal{S_t} , A_t ))
\end{align*}
$$

3. Improve policy based on new action-value function

$$
\begin{align*}
\epsilon &\gets \frac{1}{k}\\
\pi &\gets \epsilon\text{-}greedy(Q) 
\end{align*}
$$

##### theorem

GLIE Monte-Carlo control converges to the optimal action-value function, $Q(s,a) \to Q_*(s,a)$

### On-Policy Temporal-Difference Learning

#### MC vs TD Control

1. Temporal-difference(TD) learning has several advantages over Monte-Carlo(MC)
   1. Lower variance
   2. Online
   3. Incomplete sequences
2. Natural idea: use TD instead of MC in our control loop
   1. Apply TD to $Q(\mathcal{S},A)$
   2. Use $\epsilon$-greedy policy improvement
   3. Update every time-step

### $Sarsa(\lambda)$

#### Updating Action-Value Functions with Sarsa

$$
Q(\mathcal{S},A) \gets Q(\mathcal{S},A) + \alpha (\mathcal{R} +\gamma Q(\mathcal{S'},A') -Q(\mathcal{S},A))
$$

#### On-Policy Control With Sarsa

Every time-step:
Policy evaluation Sarsa, $Q \approx Q_\pi$
Policy improvement $\epsilon$-greedy policy improvement

#### Sarsa Algorithm for On-Policy Control

Initialize $Q(s, a), \forall s \in \mathcal{S}, a \in \mathcal{A(s)}$, arbitrarily, and $Q(\text{terminal-state}, \cdot) = 0$
Repeat(for each episode):
	Initialize $\mathcal{S}$
	Choose $A$ from $\mathcal{S}$ using policy derived from $Q$ (e.g., $\epsilon$-greedy)
	Repeat(for each episode):
         	Take action $A$, observe $\mathcal{R, S'}$
        		Choose ${A'}$ from $\mathcal{S'}$ using policy derived from $Q$ (e.g., $\epsilon$-greedy)
        		$Q(\mathcal{S}, A) \gets Q(\mathcal{S}, A) + \alpha [\mathcal{R} + \gamma Q(\mathcal{S'}, A') - Q(\mathcal{S}, A)]$
        		$\mathcal{S} \leftarrow \mathcal{S'}; A \leftarrow A'$
	Until $\mathcal{S}$ is terminal

#### Convergence of Sarsa

##### theorem

Sarsa converges to the optimal action-value function, $Q(s,a) \to Q_*(s,a)$, under the following conditions:

1. GLIE sequence of policies $\pi_t (a|s)$
2. Robbins-Monro sequence of step-sizes $\alpha_t$

$$
\begin{align*}
\sum_{t=1}^\infty \alpha_t = \infty \\
\sum_{t=1}^\infty \alpha_t^2 < \infty \\
\end{align*}
$$

#### n-Step Sarsa

1. Consider the following n-step returns for $n = 1, 2, \ldots, \infty$:

$$
\begin{align*}
&n = 1 \quad (\text{Sarsa}) \quad Q_t^{(1)} = \mathcal{R}_{t+1} + \gamma \mathrm{Q}(\mathcal{S}_{t+1}, {A}_{t+1})\\
&n = 2 \quad (\text{Srasa}) \quad Q_t^{(2)} = \mathcal{R}_{t+1} + \gamma \mathcal{R}_{t+2} + \gamma^2 \mathrm{Q}(\mathcal{S}_{t+2}, {A}_{t+2})\\
&\qquad\vdots\qquad\qquad \quad\qquad \qquad \vdots\\
&n = \infty \quad (\text{MC}) \quad Q_t^{(\infty)} = \mathcal{R}_{t+1} + \gamma \mathcal{R}_{t+2} + \dots + \gamma^{T-1} \mathcal{R}_T\\
\end{align*}
$$

2. Define the n-step Q-return

$$
Q_t^{(n)} = \mathcal{R}_{t+1} + \gamma \mathcal{R}_{t+2} + \dots + \gamma^{n-1} \mathcal{R}_{t+n} + \gamma^n \mathrm{Q}(\mathcal{S}_{t+n}, {A}_{t+n})
$$

3. n-step Sarsa updates $Q(s, a)$ towards the n-step Q-return

$$
\mathrm{Q}(\mathcal{S}_t, {A}_t) \leftarrow \mathrm{Q}(\mathcal{S}_t, {A}_t) + \alpha \left( Q_t^{(n)} - \mathrm{Q}(\mathcal{S}_t, {A}_t) \right)
$$

#### Forward view $Sarsa(\lambda)$

1. The $Q^\lambda$ return combines all n-step Q-returns $Q_t^{(n)}$
2. Using weight $(1-\lambda)\lambda^{n-1}$, $Q_t^\lambda = (1-\lambda)\sum_{n-1}^\infty \lambda^{n-1}Q_t^{(n)}$
3. Forward-view $Sarsa(\lambda)$,

$$
Q(\mathcal{S_t}, A_t) \gets Q(\mathcal{S_t},A_t) + \alpha (Q_t^\lambda -Q(\mathcal{S_t},A_t))
$$

#### Backward View $Sarsa(\lambda)$

1. Just like $TD(\lambda)$, we use eligibility traces in an online algorithm
2. But $Sarsa(\lambda)$ has one eligibility trace for each state-action pair

$$
\mathbb{E}_0(s,a) = 0,\, \mathbb{E}_t(s,a) = \gamma \lambda \mathbb{E}_{t-1}(s,a) + \mathbf{1}(\mathcal{S_t} =s, A_t =a)
$$

3. $Q(s,a)$ is updated for every state $s$ and action $a$
4. In proportion to TD-error $\delta_t$ and eligibility trace $\mathbb{E}_t (s, a)$

$$
\begin{align*}
\delta_t  = \mathcal{R_{t+1}} &+ \gamma Q(\mathcal{S_{t+1}}, A_{t+1}) - Q(\mathcal{S_t}, A_t)\\
Q(s,a) &\gets Q(s,a) + \alpha \delta_t \mathbb{E}_t(s,a)\\
\end{align*}
$$

#### $Sarsa(\lambda)$ Algorithm

Initialize $Q(s, a)$, arbitrarily, for all $s \in \mathcal{S}, a \in \mathcal{A}(s)$
Repeat(for each episode):
	$\mathbb{E}(s,a) = 0$, for all  $s \in \mathcal{S}, a \in \mathcal{A}(s)$
	Initialize $\mathcal{S}, A$
	Repeat(for each episode):
         	Take action $A$, observe $\mathcal{R, S'}$
        		Choose ${A'}$ from $\mathcal{S'}$ using policy derived from $Q$ (e.g., $\epsilon$-greedy)
		$\delta \gets \mathcal{R} + \gamma Q(\mathcal{S'},A') - Q(\mathcal{S},A)$
        		$\mathbb{E}(\mathcal{S}, A) \gets \mathbb{E}(\mathcal{S}, A) + 1$
		For all $s \in \mathcal{S}, a \in \mathcal{A}(s)$:
			$Q(s,a) \gets Q(s,a) + \alpha \delta \mathbb{E}(s,a)$
			$\mathbb{E}(s,a) \gets \gamma \lambda \mathbb{E}(s,a)$
        		$\mathcal{S} \leftarrow \mathcal{S'}; A \leftarrow A'$
	Until $\mathcal{S}$ is terminal

### Off-Policy Learning

1. Evaluate target policy $\pi(a|s)$ to compute $V_\pi(s)$ or $Q_\pi(s, a)$
2. While following behaviour policy $\mu(a|s)$

$$
\{\mathcal{S_1}, A_1, \mathcal{R}_2, \ldots, \mathcal{S}_T\} \sim \mu
$$

3. Why is this important?
   1. Learn from observing humans or other agents
   2. Re-use experience generated from old policies $\pi_1, \pi_2, \ldots, \pi_{t-1}$
   3. Learn about optimal policy while following exploratory policy
   4. Learn about multiple policies while following one policy

### Important Sampling

Estimate the expectation of a different distribution

$$
\begin{align*}
\mathbb{E}_{X \sim P}[f(X)] &= \sum P(X)f(X)\\
&= \sum Q(X)\frac{P(X)}{Q(X)}f(X)\\
&= \mathbb{E}_{X \sim Q}[\frac{P(X)}{Q(X)}f(X)]\\
\end{align*}
$$

#### Importance Sampling for Off-Policy Monte-Carlo

1. Use returns generated from $\mu$ to evaluate $\pi$
2. Weight return $G_t$ according to similarity between policies
3. Multiply importance sampling corrections along whole episode

$$
G_t^{\pi/\mu} = \frac{\pi(A_t | \mathcal{S}_t) \pi(A_{t+1} | \mathcal{S}_{t+1})}{\mu(A_t | \mathcal{S}_t) \mu(A_{t+1} | \mathcal{S}_{t+1})} \cdots \frac{\pi(A_T | \mathcal{S}_T)}{\mu(A_T | \mathcal{S}_T)} G_t
$$

4. Update value towards corrected return

$$
V(\mathcal{S_t}) \gets V(\mathcal{S_t}) + \alpha (G_t^{\pi / \mu} - V(\mathcal{S_t}))
$$

6. Cannot use if $\mu$ is zero when $\pi$ is non-zero
7. Importance sampling can dramatically increase variance

#### Importance Sampling for Off-Policy TD

1. Use TD targets generated from $\mu$ to evaluate $\pi$
2. Weight TD target $\mathcal{R} + \gamma V (\mathcal{S'})$ by importance sampling
3. Only need a single importance sampling correction

$$
\mathrm{V}(\mathcal{S}_t) \leftarrow \mathrm{V}(\mathcal{S}_t) + \alpha \left( \frac{\pi(A_t | \mathcal{S}_t)}{\mu(A_t | \mathcal{S}_t)} \bigl( \mathcal{R}_{t+1} + \gamma \mathrm{V}(\mathcal{S}_{t+1}) \bigr) - \mathrm{V}(\mathcal{S}_t) \right)
$$

4. Much lower variance than Monte-Carlo importance sampling
5. Policies only need to be similar over a single step

### Q-Learning

1. We now consider off-policy learning of action-values $Q(s, a)$
2. No importance sampling is required
3. Next action is chosen using behaviour policy $A_{t+1} \gets \mu (\cdot | \mathcal{S}_t )$
4. But we consider alternative successor action $A’ \sim  \pi (\cdot |\mathcal{S}_t )$
5. And update $Q(\mathcal{S}_t , A_t )$ towards value of alternative action

$$
Q(\mathcal{S_t}, A_t) \gets Q(\mathcal{S_t}, A_t) + \alpha (\mathcal{R_{t+1}} + \gamma Q(\mathcal{S_{t+1}}, A') -Q(\mathcal{S_t}, A_t))
$$

#### Off-Policy Control with Q-Learning

1. We now allow both behaviour and target policies to improve
2. The target policy $\pi$ is greedy w.r.t. $Q(s, a)$

$$
\pi(\mathcal{S}_{t+1}) = arg \, \underset{a'}{max}\,Q(\mathcal{S}_{t+1}, a')
$$

3. The behaviour policy $\mu$ is e.g. $\epsilon$-greedy w.r.t. $Q(s, a)$
4. The Q-learning target then simplifies:

$$
\begin{align*}
&\quad\mathcal{R}_{t+1} + \gamma \mathrm{Q}(\mathcal{S}_{t+1}, {A}') \\
&= \mathcal{R}_{t+1} + \gamma \mathrm{Q}\bigl(\mathcal{S}_{t+1}, \arg\max_{a'} \mathrm{Q}(\mathcal{S}_{t+1}, a')\bigr) \\
&= \mathcal{R_{t+1}} + \underset{a'}{max}\, \gamma Q(\mathcal{S_{t+1}}, a')\\
\end{align*}
$$

#### Q-Learning Control Algorithm

$$
\mathrm{Q}(\mathcal{S}, {A}) \leftarrow \mathrm{Q}(\mathcal{S}, {A}) + \alpha \left( \mathcal{R} + \gamma \max_{a'} \mathrm{Q}(\mathcal{S}', a') - \mathrm{Q}(\mathcal{S}, {A}) \right)
$$

##### theorem

Q-learning control converges to the optimal action-value function,  $\mathrm{Q}(s, a) \to Q_*(s, a)$

#### Q-Learning Algorithm for Off-Policy Control

Initialize $Q(s, a)$, $\forall s \in \mathcal{S}, a \in \mathcal{A}(s)$ arbitrarily, and $Q(terminal-state, \cdot) = 0$
Repeat(for each episode):
	Initialize $\mathcal{S}$
	Repeat(for each episode):
        		Choose ${A}$ from $\mathcal{S}$ using policy derived from $Q$ (e.g., $\epsilon$-greedy)
		Take action $A$, observe $\mathcal{R}, \mathcal{S'}$
		$Q(\mathcal{S},A) \gets Q(\mathcal{S},A) + \alpha [\mathcal{R} + \gamma \,\underset{a}{max}\,Q(\mathcal{S'},a) - Q(\mathcal{S},A)]$
        		$\mathcal{S} \leftarrow \mathcal{S'}$
	Until $\mathcal{S}$ is terminal

### Summary

#### Relationship Between DP and TD (1)

$$
\begin{array}{|p{4cm}|p{3.5cm}|p{3.5cm}|}
\hline
& \text{Full Backup (DP)} & \text{Sample Backup (TD)} \\
\hline
\text{Bellman Expectation} \newline \text{ Equation for } V_\pi(s) & \text{Iterative Policy Evaluation} & \text{TD Learning} \\
\hline
\text{Bellman Expectation}\newline \text{ Equation for } Q_\pi(s, a) & \text{Q-Policy Iteration} & \text{Sarsa} \\
\hline
\text{Bellman Optimality}\newline \text{ Equation for } Q_*(s, a) & \text{Q-Value Iteration} & \text{Q-Learning} \\
\hline
\end{array}
$$

#### Relationship Between DP and TD (2)

$$
\begin{array}{|c|c}
\hline
\textbf{Full Backup (DP)} & \textbf{Sample Backup (TD)} \\
\hline
{\text{Iterative Policy Evaluation}} & {\text{TD Learning}}\\
\hline
\mathrm{V}(s) \leftarrow \mathbb{E}\bigl[\mathcal{R} + \gamma \mathrm{V}(\mathcal{S}') | s\bigr] & \mathrm{V}(\mathcal{S}) \xleftarrow{\alpha} \mathcal{R} + \gamma \mathrm{V}(\mathcal{S}') \\
\hline
{\text{Q-Policy Iteration}} & {\text{Sarsa}}\\
\hline
\mathrm{Q}(s, a) \leftarrow \mathbb{E}\bigl[\mathcal{R} + \gamma \mathrm{Q}(\mathcal{S}', {A}') | s, a\bigr] & \mathrm{Q}(\mathcal{S}, {A}) \xleftarrow{\alpha} \mathcal{R} + \gamma \mathrm{Q}(\mathcal{S}', {A}') \\
\hline
{\text{Q-Value Iteration}} & {\text{TD Learning}} \\
\hline
\mathrm{Q}(s, a) \leftarrow \mathbb{E}\bigl[\mathcal{R} + \gamma \max_{a' \in \mathcal{A}} \mathrm{Q}(\mathcal{S}', a') | s, a\bigr] & \mathrm{Q}(\mathcal{S}, {A}) \xleftarrow{\alpha} \mathcal{R} + \gamma \max_{a' \in \mathcal{A}} \mathrm{Q}(\mathcal{S}', a') \\
\hline
\end{array}
$$

where $x \xleftarrow{\alpha} y \equiv x \leftarrow x + \alpha (y - x)$

## Lecture 6: Value Function Approximation

### outline:

1. Introduction
2. Incremental methods
3. Batch methods

### Large-Scale Reinforcement Learning

Reinforcement learning can be used to solve large problems
How can we scale up the model-free methods for prediction and control from the last two lectures?

### Value Function Approximation

1. So far we have represented value function by a lookup table
   1. Every state $s$ has an entry $V(s)$
   2. Or every state-action pair $s,a$ has an entry $Q(s, a)$
2. Problem with large MDPs:
   1. There are too many states and/or actions to store in memory
   2. It is too slow to learn the value of each state individually
3. Solution for large MDPs:
   1. Estimate value function with function approximation
      $$
      \begin{align*}
      \widehat{V}(s,w) &\approx V_\pi(s)\\
      \text{or } \widehat{Q}(s,a,w) &\approx Q_\pi(s,a)\\
      \end{align*}
      $$
   2. Generalise from seen states to unseen states
   3. Update parameter $w$ using MC or TD learning

### Which Function Approximator?

We consider differentiable function approximators, e.g.

1. Linear combinations of features
2. Neural network
3. Decision tree
4. Nearest neighbour
5. Fourier / wavelet bases
6. $\dots$
   Furthermore, we require a training method that is suitable for non-stationary, non-iid data

### Gradient Descent

1. Let $J(w)$ be a differentiable function of parameter vector $w$
2. Define the gradient of $J(w)$ to be $\nabla_wJ(w) = (\frac{\partial J(w)}{\partial w_1}, \frac{\partial J(w)}{\partial w_2}, \ldots, \frac{\partial J(w)}{\partial w_n})^T$
3. To find a local minimum of $J(w)$
4. Adjust $w$ in direction of gradient $\Delta w = -\frac{1}{2}\alpha \nabla_wJ(w)$ where $\alpha$ is a step-size parameter

#### Value Function Approximation By Stochastic Gradient Descent

1. Goal: find parameter vector $w$ minimising mean-squared error between approximate value $\widehat{V}(s, w)$ and true value $V_\pi(s)$

$$
J(w) = \mathbb{E}_\pi[(V_\pi(\mathcal{S})-\widehat{V}(\mathcal{S},w))^2]
$$

2. Gradient descent finds a local minimum

$$
\begin{align*}
\Delta w &=-\frac{1}{2} \alpha \nabla_wJ(w)\\
&= \alpha \mathbb{E}_\pi[(V_\pi(\mathcal{S})-\widehat{V}(\mathcal{S},w))\nabla_w\widehat{V}(\mathcal{S},w)]\\
\end{align*}
$$

3. Stochastic gradient descent samples the gradient

$$
\Delta w = \alpha (V_\pi(\mathcal{S}) - \widehat{V}(\mathcal{S},w))\nabla_w \widehat{V}(\mathcal{S},w)
$$

4. Expected update is equal to full gradient update

### Linear Function Approximation

#### Linear Function Approximation

1. Represent value function by a linear combination of features

$$
\widehat{V}(\mathcal{S},w) = x(\mathcal{S}^T)w = \sum_{j=1}^n x_j(\mathcal{S}^T)w_j
$$

2. Objective function is quadratic in parameters $w$

$$
J(w) = \mathbb{E}_\pi [(V_\pi (\mathcal{S}) - x(\mathcal{S})^Tw)^2]
$$

3. Stochastic gradient descent converges on global optimum
4. Update rule is particularly simple

$$
\begin{align*}
\nabla_w \widehat{V}(\mathcal{S},w) &= x(\mathcal{S})\\
\Delta w &= \alpha (V_\pi(\mathcal{S}) - \widehat{V}(\mathcal{S},w))x(\mathcal{S})\\
\end{align*}
$$

    Update$=$ step-size $\times$ prediction error $\times$ feature value

#### Table Lookup Features

1. Table lookup is a special case of linear value function approximation
2. Using table lookup features

$$
x^\text{table}(\mathcal{S}) = (\mathbf{1}(\mathcal{S} = s_1), \mathbf{1}(\mathcal{S} = s_2), \ldots, \mathbf{1}(\mathcal{S} = s_n))^T
$$

3. Parameter vector $w$ gives value of each individual state

$$
\hat{\mathrm{V}}(\mathcal{S}, {w}) =
\begin{pmatrix}
\mathbf{1}(\mathcal{S} = s_1) \\
\vdots \\
\mathbf{1}(\mathcal{S} = s_n)
\end{pmatrix}
\cdot
\begin{pmatrix}
{w}_1 \\
\vdots \\
{w}_n
\end{pmatrix}
$$

### Incremental Prediction Algorithms

#### Incremental Prediction Algorithms

1. Have assumed true value function $V_\pi(s)$ given by supervisor
2. But in RL there is no supervisor, only rewards
3. In practice, we substitute a target for $V_\pi(s)$
   1. For MC, the target is the return $G_t$
      $$
      \Delta w = \alpha \bigl( G_t - \widehat{\mathrm{V}}(\mathcal{S}_t, w) \bigr) \nabla_w \widehat{\mathrm{V}}(\mathcal{S}_t, w)
      $$
   2. For $TD(0)$, the target is the TD target $\mathcal{R_{t+1}} + \gamma \widehat{V}(\mathcal{S}_{t+1},w)$
      $$
      \Delta {w} = \alpha \bigl( \mathcal{R}_{t+1} + \gamma \widehat{\mathrm{V}}(\mathcal{S}_{t+1}, w) - \widehat{\mathrm{V}}(\mathcal{S}_t, w) \bigr) \nabla_w \widehat{\mathrm{V}}(\mathcal{S}_t, w)
      $$
   3. For $TD(\lambda)$, the target is the $\lambda$-return $G_t^\lambda$
      $$
      \Delta {w} = \alpha \bigl( G_t^\lambda - \widehat{\mathrm{V}}(\mathcal{S}_t, w) \bigr) \nabla_w \widehat{\mathrm{V}}(\mathcal{S}_t, w)
      $$

#### Monte-Carlo with Value Function Approximation

1. Return $G_t$ is an unbiased, noisy sample of true value $V_\pi(\mathcal{S_t })$
2. Can therefore apply supervised learning to “training data”:

$$
<\mathcal{S_1}, G_1>, <\mathcal{S_2}, G_2>, \ldots, <\mathcal{S}_T, G_T>
$$

3. For example, using linear Monte-Carlo policy evaluation

$$
\begin{align*}
\Delta w &= \alpha (G_t - \widehat{V}(\mathcal{S_t},w))\nabla \widehat{V}(\mathcal{S_t},w)\\
&= \alpha (G_t - \widehat{V}(\mathcal{S_t},w))x(\mathcal{S_t})\\
\end{align*}
$$

4. Monte-Carlo evaluation converges to a local optimum
5. Even when using non-linear value function approximation

#### TD Learning with Value Function Approximation

1. The TD-target $\mathcal{R}_{t+1} + V(\mathcal{S}_{t+1}, w)$ is a biased sample of true value $V_\pi(\mathcal{S_t})$
2. Can still apply supervised learning to “training data”:

$$
<\mathcal{S}_1, \mathcal{R}_2 + V(\mathcal{S}_2, w)>, <\mathcal{S}_2, \mathcal{R}_3 + V(\mathcal{S}_3, w)>, \ldots, <\mathcal{S}_{T-1}, \mathcal{R}_T>
$$

3. For example, using linear $TD(0)$

$$
\Delta w = \alpha(\mathcal{R} + V(\mathcal{S'}, w) - \widehat{V}(\mathcal{S},w)\nabla_w \widehat{V}(\mathcal{S},w) = \alpha \delta x(\mathcal{S})
$$

4. Linear $TD(0)$ converges (close) to global optimum

#### $TD(\lambda)$ with Value Function Approximation

1. The $\lambda$-return $G_t^\lambda$ is also a biased sample of true value $V_\pi(s)$
2. Can again apply supervised learning to “training data”:

$$
<\mathcal{S_1}, G_1^\lambda>, <\mathcal{S_2}, G_2^\lambda>, \ldots, <\mathcal{S}_{T-1}, G_{T-1}^\lambda>
$$

3. Forward view linear $TD(\lambda)$

$$
\Delta w = \alpha \bigl( G_t^\lambda - \widehat{\mathrm{V}}(\mathcal{S}_t, {w}) \bigr) \nabla_{w} \widehat{\mathrm{V}}(\mathcal{S}_t, {w})\\
= \alpha \bigl( G_t^\lambda - \hat{\mathrm{V}}(\mathcal{S}_t, {w}) \bigr) {x}(\mathcal{S}_t)
$$

4. Backward view linear $TD(\lambda)$

$$
\begin{align*}
  \delta_t = \mathcal{R}_{t+1} + &\gamma \widehat{\mathrm{V}}(\mathcal{S}_{t+1}, {w}) - \widehat{\mathrm{V}}(\mathcal{S}_t, {w})
  \\
  E_t = \gamma &\lambda E_{t-1} + {x}(\mathcal{S}_t)
  \\
  \Delta &{w} = \alpha \delta_t E_t
  \\
\end{align*}
$$

    Forward view and backward view linear$TD(\lambda)$ are equivalent

#### Control with Value Function Approximation

Policy evaluation: Approximate policy evaluation, $\widehat{Q}(\cdot, \cdot, w) \approx Q_\pi$
Policy improvement: $\epsilon$-greedy policy improvement

#### Action-Value Function Approximation

1. Approximate the action-value function $Q(\mathcal{S}, A, w) \approx Q_\pi(\mathcal{S}, A)$
2. Minimize mean-squared error between approximate action-value $Q(\mathcal{S}, A, w)$ and true action-value $Q_\pi(\mathcal{S}, A)$
   $$
   J(w) = \mathbb{E}_{\pi} \left[ \bigl( \mathrm{Q}_\pi(\mathcal{S}, {A}) - \widehat{\mathrm{Q}}(\mathcal{S}, {A}, {w}) \bigr)^2 \right]
   $$
3. Use stochastic gradient descent to find a local minimum

$$
\begin{align*}
-\frac{1}{2} \nabla_{{w}} J({w}) = \bigl( \mathrm{Q}_\pi(\mathcal{S}, {A}) - \widehat{\mathrm{Q}}(\mathcal{S}, {A}, {w}) \bigr) \nabla_{{w}} \widehat{\mathrm{Q}}(\mathcal{S}, {A}, {w})\\
\Delta {w} = \alpha \bigl( \mathrm{Q}_\pi(\mathcal{S}, {A}) - \widehat{\mathrm{Q}}(\mathcal{S}, {A}, {w}) \bigr) \nabla_{{w}} \widehat{\mathrm{Q}}(\mathcal{S}, {A}, {w})\\
\end{align*}
$$

#### Linear Action-Value Function Approximation

1. Represent state and action by a feature vector
   $$
   x(\mathcal{S}, {A}) = \begin{pmatrix}x_1(\mathcal{S}, {A}) \\\vdots \\x_n(\mathcal{S}, {A})\end{pmatrix}
   $$
2. Represent action-value by linear combination of features
   $$
   \widehat{\mathrm{Q}}(\mathcal{S}, {A}, {w}) = {x}(\mathcal{S}, {A})^\top {w} = \sum_{j=1}^n x_j(\mathcal{S}, {A}) w_j
   $$
3. Stochastic gradient descent update

$$
\begin{align*}
&\nabla_{{w}} \widehat{\mathrm{Q}}(\mathcal{S}, {A}, {w}) = {x}(\mathcal{S}, {A})\\
\Delta w &= \alpha (Q_\pi(\mathcal{S},A) - \widehat{Q}(\mathcal{S}, A, w))x(\mathcal{S},A)\\
\end{align*}
$$

#### Incremental Control Algorithms

Like prediction, we must substitute a target for $Q_\pi(\mathcal{S,} A)$

1. For MC, the target is the return $G_t$
   $$
   \Delta {w} = \alpha \bigl( G_t - \widehat{\mathrm{Q}}(\mathcal{S}_t, {A}_t, {w}) \bigr) \nabla_{{w}} \widehat{\mathrm{Q}}(\mathcal{S}_t, {A}_t, {w})
   $$
2. For $TD(0)$, the target is the TD target $\mathcal{R}_{t+1} + \gamma Q(\mathcal{S}_{t+1}, A_{t+1})$

$$
\Delta {w} = \alpha \bigl( \mathcal{R}_{t+1} + \gamma \widehat{\mathrm{Q}}(\mathcal{S}_{t+1}, {A}_{t+1}, {w}) - \widehat{\mathrm{Q}}(\mathcal{S}_t, {A}_t, {w}) \bigr) \nabla_{{w}} \widehat{\mathrm{Q}}(\mathcal{S}_t, {A}_t, {w})
$$

3. For forward-view $TD(\lambda)$, target is the action-value $\lambda$-return

$$
\Delta {w} = \alpha \bigl( Q_t^\lambda - \widehat{\mathrm{Q}}(\mathcal{S}_t, {A}_t, {w}) \bigr) \nabla_{{w}} \widehat{\mathrm{Q}}(\mathcal{S}_t, {A}_t, {w})
$$

4. For backward-view $TD(\lambda)$, equivalent update is

$$
\begin{align*}
\delta_t = \mathcal{R}_{t+1} + \gamma& \widehat{\mathrm{Q}}(\mathcal{S}_{t+1}, {A}_{t+1}, {w}) - \widehat{\mathrm{Q}}(\mathcal{S}_t, {A}_t, {w})
\\
\mathbb{E}_t = \gamma& \lambda \mathbb{E}_{t-1} + \nabla_{{w}} \widehat{\mathrm{Q}}(\mathcal{S}_t, {A}_t, {w})
\\
&\Delta {w} = \alpha \delta_t \mathbb{E}_t\\\
\end{align*}
$$

#### Convergence of Prediction Algorithms

| On/Off-Policy        | Algorithm | Table Lookup | Linear | Non-Linear |
| -------------------- | --------- | ------------ | ------ | ---------- |
| **On-Policy**  | MC        | ✓           | ✓     | ✓         |
|                      | TD(0)     | ✓           | ✓     | ✗         |
|                      | TD(λ)    | ✓           | ✓     | ✗         |
| **Off-Policy** | MC        | ✓           | ✓     | ✓         |
|                      | TD(0)     | ✓           | ✗     | ✗         |
|                      | TD(λ)    | ✓           | ✗     | ✗         |

#### Gradient Temporal-Difference Learning

1. TD does not follow the gradient of any objective function
2. This is why TD can diverge when off-policy or using
3. non-linear function approximation
4. Gradient TD follows true gradient of projected Bellman error| Algorithm            | Table Lookup | Linear | Non-Linear |
   | -------------------- | ------------ | ------ | ---------- |
   | **On-Policy**  |              |        |            |
   | MC                   | ✓           | ✓     | ✓         |
   | TD                   | ✓           | ✓     | ✗         |
   | Gradient TD          | ✓           | ✓     | ✓         |
   | **Off-Policy** |              |        |            |
   | MC                   | ✓           | ✓     | ✓         |
   | TD                   | ✓           | ✗     | ✗         |
   | Gradient TD          | ✓           | ✓     | ✓         |

#### Convergence of Control Algorithms

| Algorithm                                          | Table Lookup | Linear | Non-Linear |
| -------------------------------------------------- | ------------ | ------ | ---------- |
| Monte-Carlo Control                                | ✓           | (✓)   | ✗         |
| Sarsa                                              | ✓           | (✓)   | ✗         |
| Q-learning                                         | ✓           | ✗     | ✗         |
| Gradient Q-learning                                | ✓           | ✓     | ✗         |
| (✓) = chatters around near-optimal value function |              |        |            |

### Batch Methods

#### Batch Reinforcement Learning

1. Gradient descent is simple and appealing
2. But it is not sample efficient
3. Batch methods seek to find the best fitting value function
4. Given the agent’s experience (“training data”)

#### Least Squares Prediction

1. Given value function approximation $V(s, w)\approx V_\pi(s)$
2. And experience $\mathcal{D}$ consisting of $<state, value>$ pairs

$$
D= \{<s_1,  V_1^\pi>, <s_2, V_\pi^2>, \ldots, <s_T, V_\pi^T>\}
$$

3. Which parameters $w$ give the best fitting value $\widehat{V}(s, w)$?
4. Least squares algorithms find parameter vector $w$ minimizing sum-squared error between $\widehat{V}(s_t , w)$ and target values $V_t^\pi$

$$
\begin{align*}
LS(\mathbf{w}) &= \sum_{t=1}^T \bigl( V_t^\pi - \widehat{\mathrm{V}}({s}_t, {w}) \bigr)^2
\\
&= \mathbb{E}_\mathcal{D} \left[ \bigl( V^\pi - \widehat{\mathrm{V}}({s}, {w}) \bigr)^2 \right]
\\
\end{align*}
$$

#### Stochastic Gradient Descent with Experience Replay

Given experience consisting of $<state, value>$ pairs

$$
\mathcal{D} = \{<s_1,V_1^\pi>, <s_2,V_2^\pi>, \ldots, <s_T,V_T^\pi>\}
$$

Repeat:

1. Sample state, value from experience $<s,V^\pi> \sim \mathcal{D}$
2. Apply stochastic gradient descent update $\Delta w = \alpha (V^\pi - \widehat{V}(s,w))\nabla_w\widehat{V}(s,w)$
   Converges to least squares solution $w^\pi = arg\, \underset{w}{min} \, LS(w)$

#### Experience Replay in Deep Q-Networks (DQN)

DQN uses experience replay and fixed Q-targets

1. Take action at according to $\epsilon$-greedy policy
2. Store transition $(s_t , a_t , r_{t+1}, s_{t+1})$ in replay memory $\mathcal{D}$
3. Sample random mini-batch of transitions $(s, a, r , s’)$ from $\mathcal{D}$
4. Compute Q-learning targets w.r.t. old, fixed parameters $w$
5. Optimize MSE between Q-network and Q-learning targets

$$
\mathcal{L}_i (w_i ) = \mathbb{E}_{s, a, r ,s'\sim \mathcal{D}}[(r + \gamma \underset{a'}{max}\, Q(s', a'; w_i^-) - Q(s, a; w_i))^2]
$$

6. Using variant of stochastic gradient descent

#### Linear Least Squares Prediction

1. Experience replay finds least squares solution
2. But it may take many iterations
3. Using linear value function approximation $\widehat{V}(s,w) = x(s)^Tw$
4. We can solve the least squares solution directly
5. At minimum of $LS(w)$, the expected update must be zero

$$
\begin{align*}
&\mathbb{E}_\mathcal{D}[\Delta w] = 0
\\
\alpha \sum_{t=1}^T x(s_t)&(V_t^\pi - x(s_t)^\top w) = 0
\\
\sum_{t=1}^T x(s_t) V_t^\pi &= \sum_{t=1}^T x(s_t) x(s_t)^\top w
\\
\end{align*}
$$

$$
w = \left( \sum_{t=1}^T x(s_t)x(s_t)^\top \right)^{-1} \sum_{t=1}^T x(s_t) V_t^\pi
$$

6. For $N$ features, direct solution time is $\mathcal{O}(N3)$
7. Incremental solution time is $\mathcal{O}(N2)$ using Shermann-Morrison

#### Linear Least Squares Prediction Algorithms

1. We do not know true values $V_t^\pi$
2. In practice, our “training data” must use noisy or biased samples of $V_t^\pi$
   1. LSMC:
      Least Squares Monte-Carlo uses return $V_t^\pi \approx G_t$
   2. LSTD:
      Least Squares Temporal-Difference uses TD target $V_t^\pi \approx \mathcal{R}_{t+1} + \gamma \widehat{V}(\mathcal{S}_{t+1}, w)$
   3. $\text{LSTD}(\lambda)$:
      Least Squares $TD( \lambda)$ uses $\lambda \text{-return}$ $V_t^\pi \approx G_t^\lambda$
3. In each case solve directly for fixed point of MC/TD/$TD(\lambda)$

#### Linear Least Squares Prediction Algorithms (2)

1. LSMC:

$$
\begin{align*}
&0 = \sum_{t=1}^T \alpha(G_t - \widehat{V}(\mathcal{S}_t, w)) x(\mathcal{S}_t)
\\
&w = \left( \sum_{t=1}^T x(\mathcal{S}_t) x(\mathcal{S}_t)^\top \right)^{-1} \sum_{t=1}^T x(\mathcal{S}_t) G_t
\\
\end{align*}
$$

2. LSTD:

$$
\begin{align*}
&0 = \sum_{t=1}^T \alpha(\mathcal{R}_{t+1} + \gamma \widehat{V}(\mathcal{S}_{t+1}, w) - \widehat{V}(\mathcal{S}_t, w)) x(\mathcal{S}_t)
\\
&w = \left( \sum_{t=1}^T x(\mathcal{S}_t) (x(\mathcal{S}_t) - \gamma x(\mathcal{S}_{t+1}))^\top \right)^{-1} \sum_{t=1}^T x(\mathcal{S}_t) \mathcal{R}_{t+1}
\\
\end{align*}
$$

3. $\text{LSTD}(\lambda)$

$$
\begin{align*}
&0 = \sum_{t=1}^T \alpha \delta_t {E}_t\\
&w = \left( \sum_{t=1}^T {E}_t (x(\mathcal{S}_t) - \gamma x(\mathcal{S}_{t+1}))^\top \right)^{-1} \sum_{t=1}^T {E}_t \mathcal{R}_{t+1}\\
\end{align*}
$$

#### Convergence of Linear Least Squares Prediction Algorithms

| On/Off-Policy        | Algorithm | Table Lookup | Linear | Non-Linear |
| -------------------- | --------- | ------------ | ------ | ---------- |
| **On-Policy**  | MC        | ✓           | ✓     | ✓         |
|                      | LSMC      | ✓           | ✓     | -          |
|                      | TD        | ✓           | ✓     | ✗         |
|                      | LSTD      | ✓           | ✓     | -          |
| **Off-Policy** | MC        | ✓           | ✓     | ✓         |
|                      | LSMC      | ✓           | ✓     | -          |
|                      | TD        | ✓           | ✗     | ✗         |
|                      | LSTD      | ✓           | ✓     | -          |

#### Least Squares Policy Iteration

Policy evaluation: Policy evaluation by least squares Q-learning
Policy improvement: Greedy policy improvement

#### Least Squares Action-Value Function Approximation

1. Approximate action-value function $Q_\pi(s, a)$
2. using linear combination of features $x(s, a)$

$$
\widehat{Q}(s, a, w) = x(s, a)^\top w \approx Q_\pi(s, a)
$$

3. Minimise least squares error between $\widehat{Q}(s, a, w)$ and $Q_\pi(s, a)$
4. from experience generated using policy $\pi$
5. consisting of $<(state, action), value>$ pairs

$$
\mathcal{D} = \{<(s_1, a_1), V_1^\pi>, <(s_2, a_2), V_2^\pi>, \ldots, <(s_T, a_T), V_T^\pi>\}
$$

#### Least Squares Control

1. For policy evaluation, we want to efficiently use all experience
2. For control, we also want to improve the policy
3. This experience is generated from many policies
4. So to evaluate $Q_\pi(\mathcal{S}, A)$ we must learn off-policy
5. We use the same idea as Q-learning:
   1. Use experience generated by old policy $\mathcal{S}_t, A_t, \mathcal{R}_{t+1}, \mathcal{S}_{t+1} \sim \pi_{\text{old}}$
   2. Consider alternative successor action $A’ = \pi_{\text{new}}(\mathcal{S}_{t+1})$
   3. Update $Q(\mathcal{S}_t, A_t, w)$ towards value of alternative action $\mathcal{R}_{t+1} + \gamma \widehat{Q}(\mathcal{S}_{t+1}, A', w))$

#### Least Squares Q-Learning

1. Consider the following linear Q-learning update

$$
\begin{align*}
\delta = \mathcal{R}_{t+1} &+ \gamma \widehat{Q}(\mathcal{S}_{t+1}, \pi(\mathcal{S}_{t+1}), w) - \widehat{Q}(\mathcal{S}_t, A_t, w)\\
&\Delta w = \alpha \delta x(\mathcal{S}_t, A_t)\\
\end{align*}
$$

2. LSTDQ algorithm: solve for total update = zero

$$
\begin{align*}
&0 = \sum_{t=1}^T \alpha (\mathcal{R}_{t+1} + \gamma \widehat{Q}(\mathcal{S}_{t+1}, \pi(\mathcal{S}_{t+1}), w) - \widehat{Q}(\mathcal{S}_t, A_t, w)) x(\mathcal{S}_t, A_t)\\
&w = \left( \sum_{t=1}^T x(\mathcal{S}_t, A_t) (x(\mathcal{S}_t, A_t) - \gamma x(\mathcal{S}_{t+1}, \pi(\mathcal{S}_{t+1})))^\top \right)^{-1} \sum_{t=1}^T x(\mathcal{S}_t, A_t) \mathcal{R}_{t+1}\\
\end{align*}
$$

#### Least Squares Policy Iteration Algorithm

1. The following pseudocode uses LSTDQ for policy evaluation
2. It repeatedly re-evaluates experience $\mathcal{D}$ with different policies

```latex
function LSPI-TD(D, π₀)
    π' ← π₀
    repeat
        π ← π'
        Q ← LSTDQ(π, D)
        for all s ∈ S do
            π'(s) ← argmax_a Q(s, a)
        end for
    until (π ≈ π')
    return π
end function
```

#### Convergence of Control Algorithms

| Algorithm                                          | Table Lookup | Linear | Non-Linear |
| -------------------------------------------------- | ------------ | ------ | ---------- |
| Monte-Carlo Control                                | ✓           | (✓)   | ✗         |
| Sarsa                                              | ✓           | (✓)   | ✗         |
| Q-learning                                         | ✓           | ✗     | ✗         |
| LSPI                                               | ✓           | (✓)   | -          |
| (✓) = chatters around near-optimal value function |              |        |            |

## Lecture 7: Policy Gradient

### outline:

1. Introduction
2. Finite difference policy gradient
3. Monte-Carlo policy gradient
4. Actor-Critic policy gradient

### Policy-Based Reinforcement Learning

1. In the last lecture we approximated the value or action-value function using parameters $\theta$  $V_\theta(s)\approx V_\pi(s), Q_\theta(s, a)\approx Q_\pi(s, a)$
2. A policy was generated directly from the value function, e.g. using $\epsilon$-greedy
3. In this lecture we will directly parametrise the policy $\pi_\theta(s, a) = \mathbb{P}[a|s, \theta]$
4. We will focus again on model-free reinforcement learning

### Value-Based and Policy-Based RL

1. Value Based: Learnt Value Function + Implicit policy(e.g. $\epsilon$-greedy)
2. Policy Based: No Value Function + Learnt Policy
3. Actor-Critic: Learnt Value Function + Learnt Policy

### Advantages of Policy-Based RL

1. Advantages:
   1. Better convergence properties
   2. Effective in high-dimensional or continuous action spaces
   3. Can learn stochastic policies
2. Disadvantages:
   1. Typically converge to a local rather than global optimum
   2. Evaluating a policy is typically inefficient and high variance

#### Policy Objective Functions

1. Goal: given policy $\pi_\theta(s, a)$ with parameters $\theta$, find best $\theta$
2. But how do we measure the quality of a policy $\pi_\theta$?
3. In episodic environments we can use the start value

$$
J_1(\theta) = V^{\pi_\theta}(s_1) = \mathbb{E}_{\pi_\theta}[V_1]
$$

4. In continuing environments we can use the average value

$$
J_{avV} (\theta) = \sum_{s} d^{\pi_\theta}(s)V^{\pi_\theta}(s)
$$

5. Or the average reward per time-step

$$
J_{av\mathcal{R}} (\theta) = \sum_{s} d^{\pi_\theta}(s)\sum_{a}{\pi_\theta}(s,a)\mathcal{R}_s^a
$$

6. where $d^{\pi_\theta}(s)$ is stationary distribution of Markov chain for $\pi_\theta$

#### Policy Optimization

1. Policy based reinforcement learning is an optimisation problem
2. Find $\theta$ that maximises $J(\theta)$
3. Some approaches do not use gradient
   1. Hill climbing
   2. Simplex/amoeba/Nelder Mead
   3. Genetic algorithms
4. Greater eciency often possible using gradient
   1. Gradient descent
   2. Conjugate gradient
   3. Quasi-newton
5. We focus on gradient descent, many extensions possible
6. And on methods that exploit sequential structure

### Finite Difference Policy Gradient

#### Policy Gradient

1. Let $J(\theta)$ be any policy objective function
2. Policy gradient algorithms search for a local maximum in $J(\theta)$ by ascending the gradient of the policy, w.r.t. parameters $\theta$, $\Delta \theta = \alpha \nabla_\theta J(\theta)$
3. Where $\nabla_\theta J(\theta)$ is the policy gradient $\nabla_\theta J(\theta) = (\frac{\partial J(\theta)}{\partial \theta_1}, \frac{\partial J(\theta)}{\partial \theta_2}, \ldots,\frac{\partial J(\theta)}{\partial \theta_n})^\top$
4. and $\alpha$ is a step-size parameter

#### Computing Gradients By Finite Differences

1. To evaluate policy gradient of $\pi_\theta(s,a)$
2. For each dimension $k \in [1, n]$

   1. Estimate $k\text{-th}$ partial derivative of objective function w.r.t. $\theta$
   2. By perturbing $\theta$ by small amount $\epsilon$ in $k\text{-th}$ dimension

   $$
   \frac{\partial J(\theta)}{\partial \theta_k} \approx \frac{J(\theta + \epsilon u_k) - J(\theta)}{\epsilon}
   $$

   where $u_k$ is unit vector with 1 in $k\text{-th}$ component, 0 elsewhere
   3. Uses $n$ evaluations to compute policy gradient in $n$ dimensions
   4. Simple, noisy, inefficient,  but sometimes effective
   5. Works for arbitrary policies, even if policy is not differentiable

### Monte-Carlo Policy Gradient

#### Score Function

1. We now compute the policy gradient analytically
2. Assume policy $\pi_\theta$ is differentiable whenever it is non-zero
3. and we know the gradient $\nabla_\theta \pi_\theta(s,a)$
4. Likelihood ratios exploit the following identity

$$
\begin{align*}
\nabla_\theta \pi_\theta(s, a) &= \pi_\theta(s, a) \frac{\nabla_\theta \pi_\theta(s, a)}{\pi_\theta(s, a)}
\\
&= \pi_\theta(s, a) \nabla_\theta \log \pi_\theta(s, a)
\end{align*}
$$

5. The score function is $\nabla_\theta \log\pi_\theta(s,a)$

#### Softmax Policy

1. We will use a softmax policy as a running example
2. Weight actions using linear combination of features $\phi(s, a)^\top\theta$
3. Probability of action is proportional to exponentiated weight $\pi_\theta(s, a) \propto e^{\phi(s, a)^\top \theta}$
4. The score function is $\nabla_\theta \log \pi_\theta(s, a) = \phi(s, a) - \mathbb{E}_{\pi_\theta}[\phi(s, \cdot)]$

#### Gaussian Policy

1. In continuous action spaces, a Gaussian policy is natural
2. Mean is a linear combination of state features $\mu(s) = \phi(s)^\top\theta$
3. Variance may be fixed $\sigma^2$, or can also parametrised
4. Policy is Gaussian, $a\sim \mathcal{N}(\mu(s), \sigma^2)$
5. The score function is $\nabla_\theta \log \pi_\theta(s, a) = \frac{(a - \mu(s))\phi(s)}{\sigma^2}$

#### One-Step MDPs

1. Consider a simple class of one-step MDPs
   1. Starting in state $s\sim d(s)$
   2. Terminating after one time-step with reward $r = \mathcal{R}_{s,a}$
2. Use likelihood ratios to compute the policy gradient

$$
\begin{align*}
J(\theta) &= \mathbb{E}_{\pi_\theta}[r]
\\
&= \sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi_\theta(s, a) \mathcal{R}_{s,a}
\\
\nabla_\theta J(\theta) &= \sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi_\theta(s, a) \nabla_\theta \log \pi_\theta(s, a) \mathcal{R}_{s,a}
\\
&= \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(s, a) r]
\end{align*}
$$

#### Policy Gradient Theorem

1. The policy gradient theorem generalises the likelihood ratio approach to multi-step MDPs
2. Replaces instantaneous reward $r$ with long-term value $Q_\pi(s, a)$
3. Policy gradient theorem applies to start state objective, average reward and average value objective

##### theorem

For any di↵erentiable policy $\pi_\theta(s, a)$, for any of the policy objective functions $J= J_1, J_{av\mathcal{R}}$, or $\frac{1}{1-\gamma}J_{avV}$, the policy gradient is

$$
\nabla_\theta J(\theta)= \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(s, a) Q^{\pi_\theta}(s,a)]
$$

#### Monte-Carlo Policy Gradient (REINFORCE)

1. Update parameters by stochastic gradient ascent
2. Using policy gradient theorem
3. Using return vt as an unbiased sample of $Q^{\pi_\theta}(s_t, a_t)$

$$
\Delta \theta_t = \alpha \nabla_\theta \log \pi_\theta(s, a)V_t
$$

##### function REINFORCE

    Initialize$\theta$ arbitrarily
    for each episode$\{s_1, a_1, r_2, \dots, s_{T-1}, a_{T-1}, r_T\} \sim \pi_\theta$ do
    for$t = 1$ to $T - 1$ do
  			$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(s_t, a_t) V_t$
		end for
	end for
    return$\theta$
end function

### Actor-Critic Policy Gradient

#### Reducing Variance Using a Critic

1. Monte-Carlo policy gradient still has high variance
2. We use a critic to estimate the action-value function, $Q_w(s, a)\approx Q^{\pi_\theta}(s, a)$
3. Actor-critic algorithms maintain two sets of parameters
   Critic: Updates action-value function parameters $w$
   Actor: Updates policy parameters $\theta$, in direction suggested by critic
4. Actor-critic algorithms follow an approximate policy gradient

$$
\begin{align*}
\nabla_\theta J(\theta) \approx \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s, a) \, Q_w(s, a) \right]
\\
\Delta \theta = \alpha \nabla_\theta \log \pi_\theta(s, a) \, Q_w(s, a)
\end{align*}
$$

#### Estimating the Action-Value Function

1. The critic is solving a familiar problem: policy evaluation
2. How good is policy $\pi_\theta$ for current parameters $\theta$?
3. This problem was explored in previous two lectures, e.g.
   1. Monte-Carlo policy evaluation
   2. Temporal-Difference learning
   3. TD($\lambda$)
4. Could also use e.g. least-squares policy evaluation

#### Action-Value Actor-Critic

1. Simple actor-critic algorithm based on action-value critic
2. Using linear value approx $Q_w (s, a) = \phi(s, a)^\top w$
   Critic: Updates $w$ by linear TD(0)
   Actor: Updates $\theta$ by policy gradient

##### function QAC

    Initialize$s, \theta$
    	Sample $a \sim \pi_\theta$
    	for each step do
		Sample reward $r = \mathcal{R}_s^a$; sample transition $s' \sim \mathcal{P}_s^a$
        		Sample action $a' \sim \pi_\theta(s', a')$
       		$\delta = r + \gamma Q_w(s', a') - Q_w(s, a)$
        		$\theta = \theta + \alpha \nabla_\theta \log \pi_\theta(s, a) Q_w(s, a)$
        		$w \leftarrow w + \beta \delta \phi(s, a)$
        		$a \leftarrow a', s \leftarrow s'$
    	end for
end function

#### Bias in Actor-Critic Algorithms

1. Approximating the policy gradient introduces bias
2. A biased policy gradient may not find the right solution
   e.g. if $Q_w (s, a)$ uses aliased features, can we solve gridworld example?
3. Luckily, if we choose value function approximation carefully
4. Then we can avoid introducing any bias
5. i.e. We can still follow the exact policy gradient

#### Compatible Function Approximation

##### theorem(Compatible Function Approximation Theorem)

If the following two conditions are satisfied:

1. Value function approximator is compatible to the policy

$$
\nabla_w Q_w (s, a) = \nabla_\theta \log \pi_\theta(s, a)
$$

2. Value function parameters $w$ minimise the mean-squared error

$$
\varepsilon = \mathbb{E}_{\pi_\theta}[(Q^{\pi_\theta}(s, a) - Q_w (s, a))^2]
$$

Then the policy gradient is exact,

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s, a) \, Q_w(s, a) \right]
$$

#### Proof of Compatible Function Approximation Theorem

If $w$ is chosen to minimise mean-squared error, gradient of $\varepsilon$ w.r.t. $w$ must be zero,

$$
\begin{align*}
\nabla_w \varepsilon &= 0
\\
\mathbb{E}_{\pi_\theta} \left[ (Q^\theta(s, a) - Q_w(s, a)) \nabla_w Q_w(s, a) \right] &= 0
\\
\mathbb{E}_{\pi_\theta} \left[ (Q^\theta(s, a) - Q_w(s, a)) \nabla_\theta \log \pi_\theta(s, a) \right] &= 0
\\
\mathbb{E}_{\pi_\theta} \left[ Q^\theta(s, a) \nabla_\theta \log \pi_\theta(s, a) \right] &= \mathbb{E}_{\pi_\theta} \left[ Q_w(s, a) \nabla_\theta \log \pi_\theta(s, a) \right]
\end{align*}
$$

So $Q_w(s, a)$ can be substituted directly into the policy gradient,

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s, a) Q_w(s, a) \right]
$$

#### Reducing Variance Using a Baseline

1. We subtract a baseline function $B(s)$ from the policy gradient
2. This can reduce variance, without changing expectation

$$
\begin{align*}
\mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s, a) B(s) \right] &= \sum_{s \in \mathcal{S}} d^{\pi_\theta}(s) \sum_a \nabla_\theta \pi_\theta(s, a) B(s)\\
&= \sum_{s \in \mathcal{S}} d^{\pi_\theta} B(s) \nabla_\theta \sum_{a \in \mathcal{A}} \pi_\theta(s, a)\\
&= 0\\
\end{align*}
$$

3. A good baseline is the state value function $B(s) = V^{\pi_\theta}(s)$
4. So we can rewrite the policy gradient using the advantage function $A^{\pi_\theta}(s,a)$

$$
\begin{align*}
A^{\pi_\theta}(s, a) &= Q^{\pi_\theta}(s, a) - V^{\pi_\theta}(s)
\\
\nabla_\theta J(\theta) &= \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s, a) A^{\pi_\theta}(s, a) \right]
\end{align*}
$$

#### Estimating the Advantage Function (1)

1. The advantage function can significantly reduce variance of policy gradient
2. So the critic should really estimate the advantage function
3. For example, by estimating both $V^{\pi_\theta}(s)$ and $Q^{\pi_\theta}(s, a)$
4. Using two function approximators and two parameter vectors

$$
\begin{align*}
V_v(s) &\approx V^{\pi_0}(s)
\\
Q_w(s, a) &\approx Q^{\pi_0}(s, a)
\\
A(s, a) &= Q_w(s, a) - V_v(s)
\end{align*}
$$

5. And updating both value functions by e.g. TD learning

#### Estimating the Advantage Function (2)

1. For the true value function $V^{\pi_\theta}(s)$, the TD error $\delta^{\pi_\theta}$

$$
\delta^{\pi_\theta} = r + \gamma V^{\pi_\theta}(s') - V^{\pi_\theta}(s)
$$

2. is an unbiased estimate of the advantage function

$$
\begin{align*}
\mathbb{E}_{\pi_\theta}[\delta^{\pi_0}|s, a] &= \mathbb{E}_{\pi_\theta}[r + \gamma V^{\pi_\theta}(s')|s, a] - V^{\pi_\theta}(s)
\\
&= Q^{\pi_\theta}(s, a) - V^{\pi_\theta}(s)
\\
&= A^{\pi_\theta}(s, a)
\end{align*}
$$

3. So we can use the TD error to compute the policy gradient

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(s, a) \delta^{\pi_\theta}]
$$

4. In practice we can use an approximate TD error

$$
\delta_v = r + \gamma V_v(s') - V_v(s)
$$

5. This approach only requires one set of critic parameters \( v \)

#### Critics at Different Time-Scales

Critic can estimate value function $V_\theta(s)$ from many targets at different time-scales From last lecture...

1. For MC, the target is the return $V_t$, $\Delta \theta = \alpha (V_t - V_\theta(s)) \phi(s)$
2. For TD(0), the target is the TD target $r + \gamma V(s')$, $\Delta \theta = \alpha (r + \gamma V(s') - V_\theta(s)) \phi(s)$
3. For forward-view TD($\lambda$), the target is the $\lambda$-return $V_t^\lambda$, $\Delta \theta = \alpha (V_t^\lambda - V_\theta(s)) \phi(s)$
4. For backward-view TD($\lambda$), we use eligibility traces

$$
\begin{align*}
\delta_t &= r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
\\
e_t &= \gamma \lambda e_{t-1} + \phi(s_t)
 \\
 \Delta \theta &= \alpha \delta_t e_t
 \end{align*}
$$

#### Actors at Different Time-Scales

1. The policy gradient can also be estimated at many time-scales

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} [\nabla_\theta \log \pi_\theta(s, a) A^{\pi_\theta}(s, a)]
$$

2. Monte-Carlo policy gradient uses error from complete return

$$
\Delta \theta = \alpha (V_t - V_v(s_t)) \nabla_\theta \log \pi_\theta(s_t, a_t)
$$

3. Actor-critic policy gradient uses the one-step TD error

$$
\Delta \theta = \alpha (r + \gamma V_v(s_{t+1}) - V_v(s_t)) \nabla_\theta \log \pi_\theta(s_t, a_t)
$$

#### Policy Gradient with Eligibility Traces

1. Just like forward-view TD($\lambda$), we can mix over time-scales

$$
\Delta \theta = \alpha (V_t^\lambda - V_v(s_t)) \nabla_\theta \log \pi_\theta(s_t, a_t)
$$

2. where $V_t^\lambda - V_v(s_t)$ is a biased estimate of advantage
3. Like backward-view TD($\lambda$), we can also use eligibility traces
   By equivalence with TD($\lambda$), substituting $\phi(s) = \nabla_\theta \log \pi_\theta(s, a)$

$$
\begin{align*}
\delta &= r_{t+1} + \gamma V_v(s_{t+1}) - V_v(s_t)
\\\
e_{t+1} &= \lambda e_t + \nabla_\theta \log \pi_\theta(s, a)
\\\
\Delta \theta &= \alpha \delta e_t
\end{align*}
$$

4. This update can be applied online, to incomplete sequences

#### Alternative Policy Gradient Directions

1. Gradient ascent algorithms can follow any ascent direction
2. A good ascent direction can significantly speed convergence
3. Also, a policy can often be reparametrised without changing action probabilities
4. For example, increasing score of all actions in a softmax policy
5. The vanilla gradient is sensitive to these reparametrisations

#### Natural Policy Gradient

1. The natural policy gradient is parametrisation independent
2. It finds ascent direction that is closest to vanilla gradient, when changing policy by a small, fixed amount

$$
\nabla_{\theta}^{nat} \pi_{\theta}(s, a) = G_{\theta}^{-1} \nabla_{\theta} \pi_{\theta}(s, a)
$$

3. where $G_\theta$ is the Fisher information matrix

$$
G_{\theta} = \mathbb{E}_{\pi_{\theta}} \left[ \nabla_{\theta} \log \pi_{\theta}(s, a) \nabla_{\theta} \log \pi_{\theta}(s, a)^T \right]
$$

#### Natural Actor-Critic

1. Using compatible function approximation, $\nabla_w A_w(s, a) = \nabla_{\theta} \log \pi_{\theta}(s, a)$
2. So the natural policy gradient simplifies,

$$
\begin{align*}
\nabla_{\theta} J(\theta) &= \mathbb{E}_{\pi_{\theta}} \left[ \nabla_{\theta} \log \pi_{\theta}(s, a) A^{\pi_{\theta}}(s, a) \right]
\\
&= \mathbb{E}_{\pi_{\theta}} \left[ \nabla_{\theta} \log \pi_{\theta}(s, a) \nabla_{\theta} \log \pi_{\theta}(s, a)^T w \right]
\\
&= G_{\theta} w
\\
\nabla_{\theta}^{nat} J(\theta) &= w
\end{align*}
$$

#### Summary of Policy Gradient Algorithms

1. The policy gradient has many equivalent forms

$$
\begin{align*}
\nabla_\theta J(\theta) &= \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s, a) \, V_t \right] \quad \quad\qquad\qquad \text{REINFORCE} \\
&= \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s, a) \, Q^w(s, a) \right] \quad  \qquad \text{Q Actor-Critic} \\
&= \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s, a) \, A^w(s, a) \right] \quad \text{Advantage Actor-Critic} \\
&= \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s, a) \, \delta \right] \quad  \quad \quad \quad \quad \quad\text{TD Actor-Critic} \\
&= \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s, a) \, \delta e \right] \quad  \quad \quad \quad \quad\text{TD($\lambda$) Actor-Critic} \\
G_\theta^{-1} \nabla_\theta J(\theta) &= w \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \text{Natural Actor-Critic}
\end{align*}
$$

2. Each leads a stochastic gradient ascent algorithm
3. Critic uses policy evaluation (e.g. MC or TD learning) to estimate $Q^\pi(s,a)$, $A^\pi(s,a)$ or $V^\pi(s)$

## Lecture 8: Integrating Learning and Planning

### outline:

1. Introduction
2. Model-Based reinforcement learning
3. Integrated architectures
4. Simulation-Based search

### Model-Based Reinforcement Learning

1. Last lecture: learn policy directly from experience
2. Previous lectures: learn value function directly from experience
3. This lecture: learn model directly from experience
4. and use planning to construct a value function or policy
5. Integrate learning and planning into a single architecture

#### Model-Based and Model-Free RL

1. Model-Free RL: No model + Learn value function(and/or policy) from experience
2. Model-Based RL: Learn a model from experience + Plan value function(and/or policy) from model

#### Advantages of Model-Based RL

Advantages:

1. Can efficiently learn model by supervised learning methods
2. Can reason about model uncertainty
   Disadvantages:
   First learn a model, then construct a value function
   $\implies$ two sources of approximation error

#### What is a Model?

1. A model $\mathcal{M}$ is a representation of an MDP $<\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}>$, parametrized by $\eta$
2. We will assume state space $\mathcal{S}$ and action space $\mathcal{A}$ are known
3. So a model $\mathcal{M} = <\mathcal{P}_{\eta}, \mathcal{R}_{\eta}>$ represents state transitions, $\mathcal{P}_{\eta} \approx \mathcal{P}$ and rewards $\mathcal{R}_{\eta} \approx \mathcal{R}$

$$
\begin{align*}
\mathcal{S}_{t+1} \sim \mathcal{P}_{\eta}(\mathcal{S}_{t+1} | \mathcal{S}_t, A_t)\\
\mathcal{R}_{t+1} = \mathcal{R}_{\eta}(\mathcal{R}_{t+1} | \mathcal{S}_t, A_t)\\
\end{align*}
$$

4. Typically assume conditional independence between state transitions and rewards

$$
\mathbb{P}[\mathcal{S}_{t+1}, \mathcal{R}_{t+1} | \mathcal{S}_t, A_t] = \mathbb{P}[\mathcal{S}_{t+1} | \mathcal{S}_t, A_t] \, \mathbb{P}[\mathcal{R}_{t+1} | \mathcal{S}_t, A_t]
$$

#### Model Learning

1. Goal: estimate model $\mathcal{M}_{\eta}$ from experience $\{\mathcal{S}_1, A_1, \mathcal{R}_2, \dots, \mathcal{S}_T\}$
2. This is a supervised learning problem

$$
\begin{align*}
\mathcal{S}_1, A_1 &\to \mathcal{R}_2, \mathcal{S}_2 \\
\mathcal{S}_2, A_2 &\to \mathcal{R}_3, \mathcal{S}_3 \\
&\vdots \\
\mathcal{S}_{T-1}, A_{T-1} &\to \mathcal{R}_T, \mathcal{S}_T
\end{align*}
$$

3. Learning $s, a \to r$ is a regression problem
4. Learning $s, a \to s'$ is a density estimation problem
5. Pick loss function, e.g. mean-squared error, KL divergence, ...
6. Find parameters $\eta$ that minimise empirical loss

#### Examples of Models

1. Table Lookup Model
2. Linear Expectation Model
3. Linear Gaussian Model
4. Gaussian Process Model
5. Deep Belief Network Model
6. $\ldots$

#### Table Lookup Model

1. Model is an explicit MDP, $\widehat{\mathcal{P}}$, $\widehat{\mathcal{R}}$
2. Count visits $N(s, a)$ to each state action pair

$$
\begin{align*}
\widehat{\mathcal{P}}_{s,s'}^a &= \frac{1}{N(s,a)} \sum_{t=1}^{T} \mathbf{1}(\mathcal{S}_t, A_t, \mathcal{S}_{t+1} = s, a, s')\\
\widehat{\mathcal{R}}_s^a &= \frac{1}{N(s,a)} \sum_{t=1}^{T} \mathbf{1}(\mathcal{S}_t, A_t = s, a) \mathcal{R}_t
\end{align*}
$$

3. Alternatively
   1. At each time-step $t$, record experience tuple $<\mathcal{S_t}, A_t, \mathcal{R_{t+1}}, \mathcal{S}_{t+1}>$
   2. To sample model, randomly pick tuple matching $<s, a, \cdot, \cdot>$

#### Planning with an Inaccurate Model

1. Given an imperfect model $<\mathcal{P}_\eta, \mathcal{R}_\eta> \neq <\mathcal{P}, \mathcal{R}>$
2. Performance of model-based RL is limited to optimal policy for approximate MDP $<\mathcal{S}, \mathcal{A}, \mathcal{P}_\eta, \mathcal{R}_\eta>$
3. i.e. Model-based RL is only as good as the estimated model
4. When the model is inaccurate, planning process will compute a suboptimal policy
5. Solution 1: when model is wrong, use model-free RL
6. Solution 2: reason explicitly about model uncertainty

### Integrated Architectures

#### Real and Simulated Experience

We consider two sources of experience
Real experience Sampled from environment (true MDP)

$$
\begin{align*}
\mathcal{S}' &\sim \mathcal{P}_{s,s'}^a\\
\mathcal{R} &= \mathcal{R}_s^a
\end{align*}
$$

Simulated experience Sampled from model (approximate MDP)

$$
\begin{align*}
\mathcal{S}' \sim \mathcal{P}_{\eta}(\mathcal{S}' | \mathcal{S}, A)\\
\mathcal{R} = \mathcal{R}_{\eta}(\mathcal{R} | \mathcal{S}, A)
\end{align*}
$$

#### Integrating Learning and Planning

1. Model-Free RL: No model + Learn value function(and/or policy) from real experience
2. Model-Based RL(using Sample-Based Planning): Learn a model from real experience + Plan value function(and/or policy) from simulated experience
3. Dyna: Learn a model from real experience + Learn and plan value function(and/or policy) from real and simulated experience

#### Dyna-Q Algorithm

Initialize $Q(s, a)$ and $Model(s, a)$ for all $s \in \mathcal{S}$ and $a \in \mathcal{A}(s)$
Do forever:
(a) $\mathcal{S} \leftarrow$ current (nonterminal) state
(b) $A \leftarrow \epsilon \text{-greedy}(S, Q)$
(c) Execute action $A$; observe resultant reward, $\mathcal{R}$, and state, $\mathcal{S}'$
(d) $Q(\mathcal{S}, A) \leftarrow Q(\mathcal{S}, A) + \alpha [\mathcal{R} + \gamma \max_a Q(\mathcal{S}', a) - Q(\mathcal{S}, A)]$
(e) $Model(\mathcal{S}, A) \leftarrow \mathcal{R}, \mathcal{S}'$ (assuming deterministic environment)
(f) Repeat $n$ times:
    $S \leftarrow$ random previously observed state
    $A \leftarrow$ random action previously taken in $\mathcal{S}$
    $\mathcal{R}, \mathcal{S}' \leftarrow Model(\mathcal{S}, A)$
    $Q(\mathcal{S}, A) \leftarrow Q(\mathcal{S}, A) + \alpha [\mathcal{R} + \gamma \max_a Q(\mathcal{S}', a) - Q(\mathcal{S}, A)]$

### Simulation-Based Search

#### Forward Search

1. Forward search algorithms select the best action by lookahead
2. They build a search tree with the current state $s_t$ at the root
3. Using a model of the MDP to look ahead
4. No need to solve whole MDP, just sub-MDP starting from now

#### Simulation-Based Search

1. Forward search paradigm using sample-based planning
2. Simulate episodes of experience from now with the model
3. Apply model-free RL to simulated episodes
4. Simulate episodes of experience from now with the model

$$
\{\mathcal{S}_t^k, A_t^k, \mathcal{R}_{t+1}^k, \ldots, \mathcal{S}_T^k\}_{k=1}^K \sim \mathcal{M}_v
$$

5. Apply model-free RL to simulated episodes
   1. Monte-Carlo control $\to$ Monte-Carlo search
   2. Sarsa $\to$ TD search

#### Simple Monte-Carlo Search

1. Given a model $\mathcal{M}_v$ and a simulation policy $\pi$
2. For each action $a \in \mathcal{A}$

   1. Simulate $K$ episodes from current (real) state $s_t$

   $$
   \{{s}_t, a, \mathcal{R}_{t+1}^k, \mathcal{A}_{t+1}^k, \ldots, \mathcal{S}_T^k\}_{k=1}^K \sim \mathcal{M}_v, \pi
   $$

   2. Evaluate actions by mean return (Monte-Carlo evaluation)

   $$
   Q(s_t,a) = \frac{1}{K}\sum_{k=1}^K G_t \xrightarrow{P} Q_\pi (s_t,a)
   $$

   3. Select current (real) action with maximum value

   $$
   a_t = arg \,\underset{a \in \mathcal{A}}{max}\, Q(s_t,a)
   $$

#### Monte-Carlo Tree Search (Evaluation)

1. Given a model $\mathcal{M}_v$
2. Simulate $K$ episodes from current state $s_t$ using current simulation policy $\pi$

$$
\{s_t, A_t^k, \mathcal{R}_{t+1}^k , \mathcal{S}_{t+1}^k, \ldots, \mathcal{S}_T^k\}_{k=1}^K \sim \mathcal{M}_v, \pi
$$

3. Build a search tree containing visited states and actions
4. Evaluate states $Q(s, a)$ by mean return of episodes from $s, a$

$$
Q(s_t,a) = \frac{1}{N(s,a)}\sum_{k=1}^K \sum_{u=t}^T \mathbf{1}(\mathcal{S_u}, A_u = s,a)G_u \xrightarrow{P} Q_\pi (s,a)
$$

5. After search is finished, select current (real) action with maximum value in search tree

$$
a_t = arg \, \underset{a \in \mathcal{A}}{max} \, Q(s_t,a)
$$

#### Monte-Carlo Tree Search (Simulation)

1. In MCTS, the simulation policy $\pi$ improves
2. Each simulation consists of two phases (in-tree, out-of-tree)
   1. Tree policy(improves): pick actions to maximise $Q(S, A)$
   2. Default policy(fixed): pick actions randomly
3. Repeat (each simulation)
   1. Evaluate states $Q(S, A)$ by Monte-Carlo evaluation
   2. Improve tree policy, e.g. by $\epsilon$-greedy($Q$)
4. Monte-Carlo control applied to simulated experience
5. Converges on the optimal search tree, $Q(\mathcal{S},A) \to Q_*(\mathcal{S},A)$

#### Advantages of MC Tree Search

1. Highly selective best-first search
2. Evaluates states dynamically (unlike e.g. DP)
3. Uses sampling to break curse of dimensionality
4. Works for “black-box” models (only requires samples)
5. Computationally ecient, anytime, parallelisable

#### Temporal-Difference Search

1. Simulation-based search
2. Using TD instead of MC (bootstrapping)
3. MC tree search applies MC control to sub-MDP from now
4. TD search applies Sarsa to sub-MDP from now

#### MC vs TD search

1. For model-free reinforcement learning, bootstrapping is helpful
   1. TD learning reduces variance but increases bias
   2. TD learning is usually more ecient than MC
   3. TD($\lambda$) can be much more ecient than MC
2. For simulation-based search, bootstrapping is also helpful
   1. TD search reduces variance but increases bias
   2. TD search is usually more ecient than MC search
   3. TD($\lambda$) search can be much more ecient than MC search

#### TD Search

1. Simulate episodes from the current (real) state $s_t$
2. Estimate action-value function $Q(s, a)$
3. For each step of simulation, update action-values by Sarsa

$$
\Delta Q(\mathcal{S}, A) = \alpha (\mathcal{R} + \gamma Q(\mathcal{S’}, A’) - Q(\mathcal{S}, A))
$$

4. Select actions based on action-values $Q(s, a)$, e.g. $\epsilon$-greedy
5. May also use function approximation for $Q$

#### Dyna-2

1. In Dyna-2, the agent stores two sets of feature weights
   1. Long-term memory
   2. Short-term (working) memory
2. Long-term memory is updated from real experience using TD learning
   General domain knowledge that applies to any episode
3. Short-term memory is updated from simulated experience using TD search
   Specific local knowledge about the current situation
4. Over value function is sum of long and short-term memories

## Lecture 9: Exploration and Exploitation

### outline:

1. Introduction
2. Multi-Armed Bandits
3. Contextual Bandits
4. MDPs

### Exploration vs Exploitation Dilemma

1. Online decision-making involves a fundamental choice:
   Exploitation Make the best decision given current information
   Exploration Gather more information
2. The best long-term strategy may involve short-term sacrifices
3. Gather enough information to make the best overall decisions

### Principles

1. Naive Exploration: Add noise to greedy policy (e.g. $\epsilon$-greedy)
2. Optimistic Initialization: Assume the best until proven otherwise
3. Optimism in the Face of Uncertainty: Prefer actions with uncertain values
4. Probability Matching: Select actions according to probability they are best
5. Information State Search: Lookahead search incorporating value of information

### The Multi-Armed Bandit

1. A multi-armed bandit is a tuple $<\mathcal{A, R}>$
2. $\mathcal{A}$ is a known set of m actions (or “arms”)
3. $\mathcal{R}^a(r) = \mathbb{P}[r|a]$ is an unknown probability distribution over rewards
4. At each step $t$ the agent selects an action $a_t \in \mathcal{A}$
5. The environment generates a reward $r_t \sim \mathcal{R}^{a_t}$
6. The goal is to maximise cumulative reward $\sum_{\tau =1}^t r_\tau$

#### Regret

1. The action-value is the mean reward for action $a$, $Q(a) = \mathbb{E}[r | a]$
2. The optimal value $V^*$ is $V^* = Q(a^*) = \max_{a \in A} Q(a)$
3. The regret is the opportunity loss for one step $l_t = \mathbb{E}[V^* - Q(a_t)]$
4. The total regret is the total opportunity loss

$$
L_t = \mathbb{E}\left[\sum_{\tau=1}^t V^* - Q(a_\tau)\right]
$$

5. Maximize cumulative reward ≡ minimise total regret

#### Counting Regret

1. The count $N_t(a)$ is expected number of selections for action $a$
2. The gap $\Delta_a$ is the difference in value between action $a$ and optimal action $a^*$, $\Delta_a = V^* - Q(a)$
3. Regret is a function of gaps and the counts

$$
\begin{align*}
L_t &= \mathbb{E}\left[\sum_{\tau=1}^t V^* - Q(a_\tau)\right] \\
&= \sum_{a \in \mathcal{A}} \mathbb{E}[N_t(a)] (V^* - Q(a)) \\
&= \sum_{a \in \mathcal{A}} \mathbb{E}[N_t(a)] \Delta_a
\end{align*}
$$

4. A good algorithm ensures small counts for large gaps
5. Problem: gaps are not known!

#### Linear or Sublinear Regret

1. If an algorithm forever explores it will have linear total regret
2. If an algorithm never explores it will have linear total regret
3. Is it possible to achieve sublinear total regret?

#### Greedy Algorithm

1. We consider algorithms that estimate $\widehat{Q}_t(a) \approx Q(a)$
2. Estimate the value of each action by Monte-Carlo evaluation

$$
\widehat{Q}_t (a) = \frac{1}{N_t (a)}\sum_{t=1}^T r_t \mathbf{1}(a_
t = a)
$$

3. The greedy algorithm selects action with highest value $a_t^* = arg\, \underset{a \in \mathcal{A}}{max}\widehat{Q}_t(a)$
4. Greedy can lock onto a suboptimal action forever
5. $\implies$ Greedy has linear total regret

#### $\epsilon$-Greedy Algorithm

1. The $\epsilon$-greedy algorithm continues to explore forever
   1. With probability $1- \epsilon$ select $a = arg\, \underset{a \in \mathcal{A}}{max}\widehat{Q}(a)$
   2. With probability $\epsilon$ select a random action
2. Constant $\epsilon$ ensures minimum regret

$$
l_t \geq \frac{\epsilon}{\mathcal{A}}\sum_{a \in \mathcal{A}}\Delta_a
$$

3. $\implies$ $\epsilon$-greedy has linear total regret

#### Optimistic Initialization

1. Simple and practical idea: initialise $Q(a)$ to high value
2. Update action value by incremental Monte-Carlo evaluation
3. Starting with $N(a) > 0$, $\widehat{Q}_t (at ) = \widehat{Q}_{t-1} + \frac{1}{N_t (a_t )} (r_t - \widehat{Q}_{t-1})$
4. Encourages systematic exploration early on
5. But can still lock onto suboptimal action
6. $\implies$ greedy + optimistic initialisation has linear total regret
7. $\implies$ $\epsilon$-greedy + optimistic initialisation has linear total regret

#### Decaying $\epsilon_t$-Greedy Algorithm

1. Pick a decay schedule for $\epsilon_1, \epsilon_2, \ldots$
2. Consider the following schedule

$$
\begin{align*}
&c > 0\\
&d = \underset{a|\Delta_a>0}{min}\Delta_i\\
&\epsilon_t = min \, \{1, \frac{c |{\mathcal{A}}|}{d^2t}\}\\
\end{align*}
$$

3. Decaying $\epsilon_t$-greedy has logarithmic asymptotic total regret!
4. Unfortunately, schedule requires advance knowledge of gaps
5. Goal: find an algorithm with sublinear regret for any multi-armed bandit (without knowledge of $\mathcal{R}$)

#### Lower Bound

1. The performance of any algorithm is determined by similarity between optimal arm and other arms
2. Hard problems have similar-looking arms with di↵erent means
3. This is described formally by the gap a and the similarity in distributions $KL(\mathcal{R^a}||\mathcal{R^{a^*}})$

##### theorem(Lai and Robbins)

Asymptotic total regret is at least logarithmic in number of steps

$$
\underset{t \to \infty}{lim} L_t \geq \log t \sum_{a|\Delta_a \geq 0} \frac{\Delta_a}{KL(\mathcal{R^a}||\mathcal{R^{a^*}})}
$$

#### Optimism in the Face of Uncertainty

1. Which action should we pick?
2. The more uncertain we are about an action-value
3. The more important it is to explore that action
4. It could turn out to be the best action

#### Upper Confidence Bounds

1. Estimate an upper confidence $U_t (a)$ for each action value
2. Such that $Q(a) \leq \widehat{Q}_t (a) + \widehat{U}_t (a)$ with high probability
3. This depends on the number of times $N(a)$ has been selected
   1. Small $N_t (a) \implies$ large $\widehat{U}_t (a)$ (estimated value is uncertain)
   2. Large $Nt (a) \implies$ small $\widehat{U}_t (a)$ (estimated value is accurate)
4. Select action maximising Upper Confidence Bound (UCB)

$$
a_t = arg \,\underset{a \in \mathcal{A}}{max}\, \widehat{Q}_t(a) + \widehat{U}_t(a)
$$

#### Hoeffding’s Inequality

##### theorem(Hoeffding’s Inequality)

Let $X_1, \ldots, X_t$ be i.i.d. random variables in $[0,1]$, and let $X_ t =\frac{1}{\tau}\sum_{\tau =1}^t X_\tau$ be the sample mean. Then

$$
\mathbb{P}[\mathbb{E}[X] > \bar{X_t} + u] \leq e^{-2tu^2}
$$

1. We will apply Hoeffding’s Inequality to rewards of the bandit
2. conditioned on selecting action $a$

$$
\mathbb{P} \left[ Q(a) > \widehat{Q}_t(a) + U_t(a) \right] \leq e^{-2N_t(a)U_t(a)^2}
$$

#### Calculating Upper Confidence Bounds

1. Pick a probability $p$ that true value exceeds UCB
2. Now solve for $U_t(a)$

$$
\begin{align*}
e^{-2N_t(a)U_t(a)^2} &= p\\
U_t(a) &= \sqrt{\frac{-\log p}{2N_t(a)}}\\
\end{align*}
$$

3. Reduce $p$ as we observe more rewards, e.g. $p = t^{-4}$
4. Ensures we select optimal action as $t \to \infty$

$$
U_t(a) = \sqrt{\frac{2\log t}{N_t(a)}}
$$

#### UCB1

1. This leads to the UCB1 algorithm

$$
a_t = \arg\max_{a \in A} Q(a) + \sqrt{\frac{2\log t}{N_t(a)}}
$$

##### theorem

The UCB algorithm achieves logarithmic asymptotic total regret

$$
\lim_{t \to \infty} L_t \leq 8 \log t \sum_{a | \Delta_a > 0} \Delta_a
$$

#### Bayesian Bandits

1. So far we have made no assumptions about the reward distribution $\mathcal{R}$
   Except bounds on rewards
2. Bayesian bandits exploit prior knowledge of rewards, $p[\mathcal{R}]$
3. They compute posterior distribution of rewards $p [\mathcal{R}| h_t ]$
   where $h_t = a_1, r_1, \ldots, a_{t-1}, r_{t-1}$ is the history
4. Use posterior to guide exploration
   1. Upper confidence bounds (Bayesian UCB)
   2. Probability matching (Thompson sampling)
5. Better performance if prior knowledge is accurate

#### Bayesian UCB Example: Independent Gaussians

1. Assume reward distribution is Gaussian, $\mathcal{R}_a(r) = \mathcal{N}(r; \mu_a, \sigma_a^2)$
2. Compute Gaussian posterior over $\mu_a$ and $\sigma_a^2$(by Bayes law)

$$
p[\mu_a, \sigma_a^2 | h_t] \propto p[\mu_a, \sigma_a^2] \prod_{t | a_t = a} \mathcal{N}(r_t; \mu_a, \sigma_a^2)
$$

3. Pick action that maximises standard deviation of $Q(a)$

$$
a_t = \arg\max \mu_a + c\sigma_a / \sqrt{N(a)}
$$

#### Probability Matching

1. Probability matching selects action $a$ according to probability that $a$ is the optimal action

$$
\pi(a | h_t) = \mathbb{P}[Q(a) > Q(a'), \forall a' \neq a | h_t]
$$

2. Probability matching is optimistic in the face of uncertainty
   Uncertain actions have higher probability of being max
3. Can be difficult to compute analytically from posterior

#### Thompson Sampling

1. Thompson sampling implements probability matching

$$
\begin{align*}
\pi(a | h_t) &= \mathbb{P}[Q(a) > Q(a'), \forall a' \neq a | h_t] \\
&= \mathbb{E}_{\mathcal{R} | h_t} \left[ \mathbf{1}\left(a = \arg\max_{a \in \mathcal{A}} Q(a)\right) \right]
\end{align*}
$$

2. Use Bayes law to compute posterior distribution $p[\mathcal{R} | h_t]$
3. Sample a reward distribution $\mathcal{R}$ from posterior
4. Compute action-value function $Q(a) = \mathbb{E}[R_a]$
5. Select action maximising value on sample, $a_t = \arg\max_{a \in \mathcal{A}} Q(a)$
6. Thompson sampling achieves Lai and Robbins lower bound!

#### Value of Information

1. Exploration is useful because it gains information
2. Can we quantify the value of information?
   1. How much reward a decision-maker would be prepared to pay in order to have that information, prior to making a decision
   2. Long-term reward after getting information - immediate reward
3. Information gain is higher in uncertain situations
4. Therefore it makes sense to explore uncertain situations more
5. If we know value of information, we can trade-off exploration and exploitation optimally

#### Information State Space

1. We have viewed bandits as one-step decision-making problems
2. Can also view as sequential decision-making problems
3. At each step there is an information state $\tilde{s}$
   1. $\tilde{s}$ is a statistic of the history, $\tilde{s}_t = f (h_t )$
   2. summarizing all information accumulated so far
4. Each action $a$ causes a transition to a new information state $\tilde{s}’$(by adding information), with probability $\widetilde{\mathcal{P}}_{\tilde{s},\tilde{s}’}^a$
5. This defines MDP $\mathcal{M}$ in augmented information state space $\widetilde{\mathcal{M}} =<\widetilde{\mathcal{S}}, \mathcal{A}, \widetilde{\mathcal{P}}, \mathcal{R}, \gamma>$

#### Solving Information State Space Bandits

1. We now have an infinite MDP over information states
2. This MDP can be solved by reinforcement learning
3. Model-free reinforcement learning, e.g. Q-learning (Duff, 1994)
4. Bayesian model-based reinforcement learning
   1. e.g. Gittins indices (Gittins, 1979)
   2. This approach is known as Bayes-adaptive RL
   3. Finds Bayes-optimal exploration/exploitation trade-off with respect to prior distribution

#### Bayes-Adaptive Bernoulli Bandits

1. Start with Beta($\alpha_a, \beta_a$) prior over reward function $\mathcal{R}^a$
2. Each time $a$ is selected, update posterior for $\mathcal{R}^a$
   1. Beta($\alpha_a + 1, \beta_a$) if $r = 0$
   2. Beta($\alpha_a, \beta_a + 1$) if $r = 1$
3. This defines transition function $\widetilde{\mathcal{P}}$ for the Bayes-adaptive MDP
4. Information state $<\alpha,\beta>$ corresponds to reward model Beta($\alpha,\beta$)
5. Each state transition corresponds to a Bayesian model update

#### Gittins Indices for Bernoulli Bandits

1. Bayes-adaptive MDP can be solved by dynamic programming
2. The solution is known as the Gittins index
3. Exact solution to Bayes-adaptive MDP is typically intractable
   Information state space is too large
4. Recent idea: apply simulation-based search (Guez et al. 2012)
   1. Forward search in information state space
   2. Using simulations from current information state

#### Contextual Bandits

1. A contextual bandit is a tuple $<\mathcal{A, S, R}>$
2. $\mathcal{A}$ is a known set of actions (or “arms”)
3. $\mathcal{S}= \mathbb{P} [s]$ is an unknown distribution over states (or “contexts”)
4. $\mathcal{R}_s^a (r ) = \mathbb{P} [r |s, a]$ is an unknown probability distribution over rewards
5. At each step $t$
   1. Environment generates state $s_t \sim \mathcal{S}$
   2. Agent selects action $a_t \in \mathcal{A}$
   3. Environment generates reward $r_t \sim \mathcal{R}_{s_t}^{a_t}$
6. Goal is to maximise cumulative reward $\sum_{\tau =1}^t r_\tau$

#### Linear Regression

1. Action-value function is expected reward for state $s$ and action $a$, $Q(s, a) = \mathbb{E} [r |s, a]$
2. Estimate value function with a linear function approximator $Q_\theta(s, a) = \phi(s, a)^\top \theta \approx Q(s, a)$
3. Estimate parameters by least squares regression

$$
\begin{align*}
A_t &= \sum_{\tau=1}^t \phi(s_\tau, a_\tau) \phi(s_\tau, a_\tau)^\top\\
b_t &= \sum_{\tau=1}^t \phi(s_\tau, a_\tau) r_\tau\\
\theta_t &= A_t^{-1} b_t\\
\end{align*}
$$

#### Linear Upper Confidence Bounds

1. Least squares regression estimates the mean action-value $Q_\theta(s, a)$
2. But it can also estimate the variance of the action-value $\sigma_\theta^2 (s,a)$
3. i.e. the uncertainty due to parameter estimation error
4. Add on a bonus for uncertainty, $U_\theta(s, a) = c\sigma$
5. i.e. define UCB to be $c$ standard deviations above the mean

#### Geometric Interpretation

1. Define confidence ellipsoid Et around parameters $\theta_t$
2. Such that $\varepsilon_t$ includes true parameters $\theta^*$ with high probability
3. Use this ellipsoid to estimate the uncertainty of action values
4. Pick parameters within ellipsoid that maximise action value $arg \, \underset{\theta \in \varepsilon}{max} \, Q_\theta(s,a)$

#### Calculating Linear Upper Confidence Bounds

1. For least squares regression, parameter covariance is $A^{-1}$
2. Action-value is linear in features, $Q_\theta(s, a) = \phi(s, a)^\top \theta$
3. So action-value variance is quadratic, $\sigma_\theta^2(s, a) = \phi(s, a)^\top A^{-1} \phi(s, a)$
4. Upper confidence bound is $Q_\theta(s, a) + c\sqrt{\phi(s, a)^\top A^{-1} \phi(s, a)}$
5. Select action maximising upper confidence bound $a_t = \arg\max_{a \in \mathcal{A}} Q_\theta(s_t, a) + c\sqrt{\phi(s_t, a)^\top A_t^{-1} \phi(s_t, a)}$

#### Exploration/Exploitation Principles to MDPs

The same principles for exploration/exploitation apply to MDPs

1. Naive Exploration
2. Optimistic Initialisation
3. Optimism in the Face of Uncertainty
4. Probability Matching
5. Information State Search

#### Optimistic Initialisation: Model-Free RL

1. Initialise action-value function $Q(s, a)$ to $\frac{r_{max}}{1-\gamma}$
2. Run favourite model-free RL algorithm
   1. Monte-Carlo control
   2. Sarsa
   3. Q-learning
   4. $\ldots$
3. Encourages systematic exploration of states and action

#### Optimistic Initialisation: Model-Based RL

1. Construct an optimistic model of the MDP
2. Initialize transitions to go to heaven
   (i.e. transition to terminal state with rmax reward)
3. Solve optimistic MDP by favourite planning algorithm
   1. policy iteration
   2. value iteration
   3. tree search
   4. $\ldots$
4. Encourages systematic exploration of states and actions
5. e.g. RMax algorithm (Brafman and Tennenholtz)

#### Upper Confidence Bounds: Model-Free RL

1. Maximise UCB on action-value function $Q^\pi(s, a)$, $a_t = arg\, \underset{a \in \mathcal{A}}{max}Q(s_t , a) + U(s_t , a)$
   1. Estimate uncertainty in policy evaluation (easy)
   2. Ignores uncertainty from policy improvement
2. Maximise UCB on optimal action-value function $Q^*(s, a)$

$$
a_t = arg\, \underset{a \in \mathcal{A}}{max}\, Q(s_t , a) + U_1(s_t , a) + U_2(s_t , a)
$$

    1. Estimate uncertainty in policy evaluation (easy)
	2. plus uncertainty from policy improvement (hard)

#### Bayesian Model-Based RL

1. Maintain posterior distribution over MDP models
2. Estimate both transitions and rewards, $p [\mathcal{P, R} | h_t ]$
   where $h_t = s_1, a_1, r_2, \ldots, s_t$ is the history
3. Use posterior to guide exploration
   1. Upper confidence bounds (Bayesian UCB)
   2. Probability matching (Thompson sampling)

#### Thompson Sampling: Model-Based RL

1. Thompson sampling implements probability matching

$$
\begin{align*}
\pi(s, a | h_t ) &= \mathbb{P} [Q^*(s, a) > Q^*(s, a’), \forall a’ \neq a | h_t ]\\
&= \mathbb{E}_{\mathcal{P,R}|h_t} [\mathbf{1}(a = arg\, \underset{a \in \mathcal{A}}{max} \,Q^*(s, a))]\\
\end{align*}
$$

2. Use Bayes law to compute posterior distribution $p [\mathcal{P, R} | h_t ]$
3. Sample an MDP $\mathcal{P, R}$ from posterior
4. Solve MDP using favourite planning algorithm to get $Q^*(s, a)$
5. Select optimal action for sample MDP, $a_t = arg\, \underset{a \in \mathcal{A}}{max}\,Q^*(s_t , a)$

#### Information State Search in MDPs

1. MDPs can be augmented to include information state
2. Now the augmented state is $<s,\tilde{s}>$
   1. where $s$ is original state within MDP
   2. and $\tilde{s}$ is a statistic of the history (accumulated information)
3. Each action $a$ causes a transition
   1. to a new state $s’$ with probability $\mathcal{P}_{s,s’}^a$
   2. to a new information state $\tilde{s}’$
4. Defines MDP $\mathcal{M}$ in augmented information state space $\mathcal{M} = <\widetilde{\mathcal{S}}, \mathcal{A}, \widetilde{\mathcal{P}}, \mathcal{R}, \gamma>$

#### Bayes Adaptive MDPs

1. Posterior distribution over MDP model is an information state $\tilde{s}_t = \mathbb{P} [\mathcal{P, R}|h_t ]$
2. Augmented MDP over $<s,\tilde{s}>$ is called Bayes-adaptive MDP
3. Solve this MDP to find optimal exploration/exploitation trade-off(with respect to prior)
4. However, Bayes-adaptive MDP is typically enormous
5. Simulation-based search has proven effective (Guez et al.)

#### Conclusion

1. Have covered several principles for exploration/exploitation
   1. Naive methods such as $\epsilon$-greedy
   2. Optimistic initialisation
   3. Upper confidence bounds
   4. Probability matching
   5. Information state search
2. Each principle was developed in bandit setting
3. But same principles also apply to MDP setting

## Lecture 10: Classic Games

### outline:

1. State of the Art
2. Game theory
3. Minimal search
4. Self-Play reinforcement learning
5. Combining reinforcement learning and minimax search
6. Reinforcement learning in Imperfect-Information Games
7. Conclusions

### State of the Art

#### AI in Games: State of the Art

|  Program  |              Level of play              |     Program to achieve level     |
| :--------: | :-------------------------------------: | :------------------------------: |
|  Checkers  |                 Perfect                 |             Chinook             |
|   Chess   | Superhuman `<br>`International Master | Deep Blue `<br>`KnightCap/Meep |
|  Othello  |               Superhuman               |            Logistello            |
| Backgammon |               Superhuman               |            TD-Gammon            |
|  Scrabble  |               Superhuman               |              Maven              |
|     Go     |               Grandmaster               |      MoGo, Crazy Stone, Zen      |
|   Poker   |               Superhuman               |          Polaris/SmooCT          |

### Game Theory

#### Optimality in Games

1. What is the optimal policy $\pi^i$ for $i\text{-th}$ player?
2. If all other players fix their policies $\pi^{-i}$
3. Best response $\pi_*^i(\pi^{-i} )$ is optimal policy against those policies
4. Nash equilibrium is a joint policy for all players $\pi^i = \pi_*^i(\pi^{-i})$
5. such that every player’s policy is a best response
6. i.e. no player would choose to deviate from Nash

#### Single-Agent and Self-Play Reinforcement Learning

1. Best response is solution to single-agent RL problem
   1. Other players become part of the environment
   2. Game is reduced to an MDP
   3. Best response is optimal policy for this MDP
2. Nash equilibrium is fixed-point of self-play RL
   1. Experience is generated by playing games between agents $a_1 \sim\pi^1, a_2\sim \pi^2, \ldots$
   2. Each agent learns best response to other players
   3. One player’s policy determines another player’s environment
   4. All players are adapting to each other

#### Two-Player Zero-Sum Games

We will focus on a special class of games:

1. A two-player game has two (alternating) players
   We will name player 1 white and player 2 black
2. A zero sum game has equal and opposite rewards for black and white

$$
R^1 + R^2 = 0
$$

We consider methods for finding Nash equilibria in these games

1. Game tree search (i.e. planning)
2. Self-play reinforcement learning

#### Perfect and Imperfect Information Games

1. A perfect information or Markov game is fully observed
   1. Chess
   2. Checkers
   3. Othello
   4. Backgammon
   5. Go
2. An imperfect information game is partially observed
   1. Scrabble
   2. Poker
3. We focus first on perfect information games

### Minimax Search

#### Minimax

1. A value function defines the expected total reward given joint policies $\pi = <\pi^1, \pi^2>$ $V_\pi(s) = \mathbb{E}_\pi [Gt |\mathcal{S}_t = s]$
2. A minimax value function maximizes white’s expected return while minimizing black’s expected return $V_*(s) = \underset{\pi^1}{max}\,\underset{\pi^2}{min}\,V_\pi(s)$
3. A minimax policy is a joint policy $\pi = <\pi^1, \pi^2>$ that achieves the minimax values
4. There is a unique minimax value function
5. A minimax policy is a Nash equilibrium

#### Minimax Search

1. Minimax values can be found by depth-first game-tree search
2. Introduced by Claude Shannon: Programming a Computer for Playing Chess

#### Value Function in Minimax Search

1. Search tree grows exponentially
2. Impractical to search to the end of the game
3. Instead use value function approximator $V(s, w) \approx V_*(s)$
   aka evaluation function, heuristic function
4. Use value function to estimate minimax value at leaf nodes
5. Minimax search run to fixed depth with respect to leaf values

#### Binary-Linear Value Function

1. Binary feature vector $x(s)$: e.g. one feature per piece
2. Weight vector $w$: e.g. value of each piece
3. Position is evaluated by summing weights of active features

#### Deep Blue

1. Knowledge
   1. 8000 handcrafted chess features
   2. Binary-linear value function
   3. Weights largely hand-tuned by human experts
2. Search
   1. High performance parallel alpha-beta search
   2. 480 special-purpose VLSI chess processors
   3. Searched 200 million positions/second
   4. Looked ahead 16-40 ply
3. Results
   1. Defeated human champion Garry Kasparov 4-2 (1997)
   2. Most watched event in internet history

#### Chinook

1. Knowledge
   1. Binary-linear value function
   2. 21 knowledge-based features (position, mobility, ...)
   3. x4 phases of the game
2. Search
   1. High performance alpha-beta search
   2. Retrograde analysis
      1. Search backward from won positions
      2. Store all winning positions in lookup tables
      3. Plays perfectly from last n checkers
3. Results
   1. Defeated Marion Tinsley in world championship 1994
      won 2 games but Tinsley withdrew for health reasons
   2. Chinook solved Checkers in 2007
      perfect play against God

### Self-Play Reinforcement Learning

#### Self-Play Temporal-Di↵erence Learning

1. Apply value-based RL algorithms to games of self-play
2. MC: update value function towards the return $G_t$

$$
\Delta_w = \alpha(G_t - V(\mathcal{S}_t , w))\nabla_wV (\mathcal{S}_t , w)
$$

3. TD(0): update value function towards successor value $V(\mathcal{S}_{t+1})$

$$
\Delta_w = \alpha(V(\mathcal{S}_{t+1} , w) - V(\mathcal{S}_t , w))\nabla_wV (\mathcal{S}_t , w)
$$

4. TD($\lambda$): update value function towards the $\lambda$-return $G_t^\lambda$

$$
\Delta_w = \alpha(G_t^\lambda - V(\mathcal{S}_t , w))\nabla_wV (\mathcal{S}_t , w)
$$

#### Policy Improvement with Afterstates

1. For deterministic games it is sufficient to estimate $V_*(s)$
2. This is because we can efficiently evaluate the afterstate $Q_*(s, a) = V_*(succ(s, a))$
3. Rules of the game define the successor state $succ(s, a)$
4. Actions are selected e.g. by min/maximising afterstate value

$$
\begin{align*}
A_t = arg\,max_a \, V_*(succ(\mathcal{S}_t , a)) \qquad \text{for white}\\
A_t = arg\,min_a \, V_*(succ(\mathcal{S}_t , a)) \qquad \text{for black}\\
\end{align*}
$$

5. This improves joint policy for both players

#### Reinforcement Learning in Logistello

Logistello used generalised policy iteration

1. Generate batch of self-play games from current policy
2. Evaluate policies using Monte-Carlo (regress to outcomes)
3. Greedy policy improvement to generate new players
   Results
   Defeated World Champion Takeshi Murukami 6-0

#### Self-Play TD in Backgammon: TD-Gammon

1. Initialised with random weights
2. Trained by games of self-play
3. Using non-linear temporal-di↵erence learning

$$
\begin{align*}
\delta_t = V(\mathcal{S}_{t+1}, w)-V(\mathcal{S}_t , w)\\
\Delta w = \alpha \delta_t \nabla_wV (\mathcal{S}_t , w)\\
\end{align*}
$$

4. Greedy policy improvement (no exploration)
5. Algorithm always converged in practice
6. Not true for other games

### Combining Reinforcement Learning and Minimax Search

#### Simple TD

1. TD: update value towards successor value
2. Value function approximator $V(s, w)$ with parameters $w$
3. Value function backed up from raw value at next state

$$
V (\mathcal{S}_t , w) \gets V(\mathcal{S}_{t+1}, w)
$$

4. First learn value function by TD learning
5. Then use value function in minimax search (no learning)

$$
V_+(\mathcal{S}_t,w) = \underset{s \in leaves(\mathcal{S_t})}{minimax}\,V(s,w)
$$

#### TD Root in Checkers: Samuel’s Player

1. First ever TD learning algorithm (Samuel 1959)
2. Applied to a Checkers program that learned by self-play
3. Defeated an amateur human player
4. Also used other ideas we might now consider strange

#### TD Leaf

1. TD leaf: update search value towards successor search value
2. Search value computed at current and next step

$$
\begin{align*}
V_+(\mathcal{S}_t,w) = \underset{s \in leaves(\mathcal{S_t})}{minimax}\,V(s,w)\\
V_+(\mathcal{S}_{t+1},w) = \underset{s \in leaves(\mathcal{S_{t+1}})}{minimax}\,V(s,w)
\end{align*}
$$

3. Search value at step $t$ backed up from search value at $t + 1$

$$
\begin{align*}
V_+(\mathcal{S}_t,w) &\gets V_+(\mathcal{S}_{t+1},w)\\
\implies V(l_+(\mathcal{S}_t),w) &\gets  V(l_+(\mathcal{S}_{t+1}),w) \\
\end{align*}
$$

#### TD leaf in Chess: Knightcap

1. Learning
   1. Knightcap trained against expert opponent
   2. Starting from standard piece values only
   3. Learnt weights using TD leaf
2. Search
   Alpha-beta search with standard enhancements
3. Results
   1. Achieved master level play after a small number of games
   2. Was not effective in self-play
   3. Was not effective without starting from good weights

#### TD leaf in Checkers: Chinook

1. Original Chinook used hand-tuned weights
2. Later version was trained by self-play
3. Using TD leaf to adjust weights
   Except material weights which were kept fixed
4. Self-play weights performed $\geq$ hand-tuned weights
5. i.e. learning to play at superhuman level

#### TreeStrap

1. TreeStrap: update search values towards deeper search values
2. Minimax search value computed at all nodes $s \in nodes(\mathcal{S}_t )$
3. Value backed up from search value, at same step, for all nodes

$$
V(s,w) \gets V_+(s,w) \implies V(s,w) \gets V(I_+(s),w)
$$

#### Treestrap in Chess: Meep

1. Binary linear value function with 2000 features
2. Starting from random initial weights (no prior knowledge)
3. Weights adjusted by TreeStrap
4. Won 13/15 vs. international masters
5. Effective in self-play
6. Effective from random initial weight

#### Simulation-Based Search

1. Self-play reinforcement learning can replace search
2. Simulate games of self-play from root state $\mathcal{S}_t$
3. Apply RL to simulated experience
   1. Monte-Carlo Control $\implies$ Monte-Carlo Tree Search
   2. Most effective variant is UCT algorithm
      Balance exploration/exploitation in each node using UCB
   3. Self-play UCT converges on minimax values
   4. Perfect information, zero-sum, 2-player games
   5. Imperfect information: see next section

### Reinforcement Learning in Imperfect-Information Games

#### Game-Tree Search in Imperfect Information Games

1. Players have different information states and therefore separate search trees
2. There is one node for each information state
   1. summarising what a player knows
   2. e.g. the cards they have seen
3. Many real states may share the same information state
4. May also aggregate states e.g. with similar value

#### Solution Methods for Imperfect Information Games

Information-state game tree may be solved by:

1. Iterative forward-search methods
   1. e.g. Counterfactual regret minimization
   2. “Perfect” play in Poker (heads-up limit Hold’em)
2. Self-play reinforcement learning
3. e.g. Smooth UCT
   1. 3 silver medals in two- and three-player Poker (limit Hold’em)
   2. Outperformed massive-scale forward-search agent

#### Smooth UCT Search

1. Apply MCTS to information-state game tree
2. Variant of UCT, inspired by game-theoretic Fictitious Play
   Agents learn against and respond to opponents’ average behaviour
3. Extract average strategy from nodes’ action counts, $\pi_{avg} (a|s) = \frac{N(s,a)}{N(s)}$
4. At each node, pick actions according to

$$
A \sim 
\begin{cases} 
\text{UCT}(\mathcal{S}), & \text{with probability } \eta \\ 
\pi_{\text{avg}}(\cdot | \mathcal{S}), & \text{with probability } 1 - \eta 
\end{cases}
$$

5. Empirically, in variants of Poker:
   1. Naive MCTS diverged
   2. Smooth UCT converged to Nash equilibrium

#### RL in Games: A Successful Recipe

|           Program           |          Input features          |    Value Fn    |    RL    |          Training          |      Search      |
| :--------------------------: | :-------------------------------: | :------------: | :-------: | :------------------------: | :---------------: |
|      Chess `<br>`Meep      | Binary `<br>`Pieces, pawns, ... |     Linear     | TreeStrap | Self-Play `<br>`/ Expert |       αβ       |
|   Checkers `<br>`Chinook   |    Binary `<br>`Pieces, ...    |     Linear     |  TD leaf  |         Self-Play         |       αβ       |
|  Othello `<br>`Logistello  |    Binary `<br>`Disc configs    |     Linear     |    MC    |         Self-Play         |       αβ       |
| Backgammon `<br>`TD Gammon |    Binary `<br>`Num checkers    | Neural network |  TD(λ)  |         Self-Play         | αβ `<br>`/ MC |
|       Go `<br>`MoGo       |   Binary `<br>`Stone patterns   |     Linear     |    TD    |         Self-Play         |       MCTS       |
|    Scrabble `<br>`Maven    |  Binary `<br>`Letters on rack  |     Linear     |    MC    |         Self-Play         |     MC search     |
|     Limit Hold'em SmooCT     |  Binary `<br>`Card abstraction  |     Linear     |   MCTS   |         Self-Play         |        —        |
