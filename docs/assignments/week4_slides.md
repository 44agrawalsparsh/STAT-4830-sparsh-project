---
marp: true
title: "Week 4 Submission"
subtitle: "Optimizing a Fantasy Hockey Auction"
author: "Your Name"
date: "2025-02-07"
theme: default
style: |
  /* This will apply a smaller font size to all slides */
  section {
    font-size: 22px;
  }
math: true
---

# Week 4 Submission

- **Project:** Optimizing decision-making in a fantasy hockey auction
- **Focus:** Using reinforcement learning and Monte Carlo Tree Search (MCTS) to improve bidding strategies

---

## Project Statement

- **Context:** Fantasy hockey auction in a league of *n* members
- **Resources:** Each member gets *t* tokens
- **Team Composition:** Each team consists of:
  - *o* forwards
  - *d* defensemen
  - *g* goalies

---

## Auction Setup

- **Player Valuation:**
  - Each player’s value is an independent random variable
  - Known means & variances (with potential future extensions to differing priors)
- **Auction Mechanics:**
  - Players are randomly nominated each round
  - Bidding is cyclical; actions: **bid up** (increase price by 1 token) or **pass**
  - A passed member cannot bid again for that player
  - Auction ends when only one member remains, awarding the player at the final price

---

## Objective & Optimization

- **Goal:** Maximize league ranking by optimizing the cumulative realized value of drafted players
- **Decision Function:**  
  $$
  f: S \rightarrow \{0,1\}
  $$  
  Where **S** represents the current auction state and the output indicates **bid** (1) or **pass** (0)
- **Mathematical Formulation:**  
  $$
  \max_{\pi} \mathbb{E}\left[\sum_{t=0}^{T} V(s_t) \mid \pi \right]
  $$
  - Expected rank/value at state: $V(s_t)$
  - Policy guiding the agent’s bidding decisions: $\pi$

---

## Why This Problem Matters

- **Immediate Benefit:** Improve your own fantasy draft strategy!
- **Wider Applications:**
  - **Finance:** Tailor bidding strategies based on risk profiles
  - **Project Management:** Efficient resource allocation
  - **Digital Marketing:** Optimized ad placement strategies
  - **Healthcare:** Equitable allocation of critical resources

---

## Data Requirements

- **Initial Stage:** Use intuitive, generated distributions for player values
- **Future Stage:** Incorporate:
  - Historical NHL data
  - Common fantasy hockey scoring systems
- **Goal:** Ensure the model generalizes well despite simplifying assumptions

---

## Measuring Success

- **Benchmarking:** Compare against dummy agents with naive valuation functions
- **Simulation:** Run multiple auction simulations to verify that the agent’s choices lead to higher cumulative value
- **Validation:** Cross-validation and A/B testing with various network architectures and hyperparameters

---

## Potential Risks & Mitigations

- **Technical Risks:**
  - Limited RL experience
  - Underestimating the problem’s complexity (both technically and computationally)
- **Mitigation Strategy:**
  - Simplify auction settings as needed
  - Start with a dummy model to gain insights before scaling up

---

## Planned Extensions

- **Disagreements over Valuations:**
  - Allow for differing priors (players’ means & variances as random variables)
  - Incorporate learning from others’ bidding behavior
- **Asymmetric Information:**
  - Model scenarios where some players have more or less information
  - Investigate information leakage from bidding patterns

---

## Technical Approach Overview

- **Finite Game Model:**  
  - Auction is modeled as a finite game with a large, but countable, state space.
- **Value Function:**  
  $V(s)$  represents the expected rank from state $s$
- **RL Framework:**  
  - Use reinforcement learning to approximate $V(s)$  
  and derive the policy $\pi$

---

## Algorithm & Approach

- **Inspiration:** AlphaGo Zero style self-play combined with MCTS
- **Why This Approach:**
  - Leverages self-play to learn effective strategies
  - Avoids heavy reliance on human-guided heuristics
- **Notes:**
  - Although classical methods (Q-learning, SARSA) were considered, the state space complexity favors an MCTS-based strategy

---

## PyTorch Implementation Strategy

1. **Environment Definition:**
   - Custom Gymnasium environment to simulate the auction
2. **Neural Network:**
   - Joint policy & value network: inputs state, outputs:
     - Action probabilities (bid or pass)
     - Expected rank/value: $V(s)$
3. **Training Loop:**
   - Use self-play to generate data and update the network
4. **Loss Functions:**
   - **Policy Loss:** Cross-entropy between predicted & actual actions
   - **Value Loss:** Mean squared error for $V(s)$
5. **Optimization:**
   - Use Adam optimizer for weight updates
6. **Evaluation:**
   - Benchmark against naive strategies periodically

---

## Validation & Resource Considerations

- **Validation Methods:**
  - Simulation, cross-validation, and A/B testing
- **Resource Needs:**
  - Access to GPUs for efficient training
  - Sufficient compute for self-play simulations (noting that even simplified drafts may take up to 15 minutes)
- **Constraints:**
  - Balancing model complexity with available computational power

---

## Initial Results

- **Simulation Setup:**
  - Simplified draft with 6 teams:
    - 8 forwards, 4 defensemen, 2 goalies per team
- **Observations:**
  - Basic MCTS agents often outperform naive strategies
  - Resource usage: ~50% CPU, ~80 MB per process
  - Some drafts take up to 15 minutes to simulate
- **Insight:**
  - Variance in player performance may add unnecessary complexity—consider focusing on mean values

---

## Unexpected Challenges

- **Simulation Time:**
  - A single draft episode can be time-consuming
- **Complexity:**
  - The game tree is very complex, affecting performance
- **Potential Adjustment:**
  - Simplify by ignoring variance in player outcomes to focus on raw mean scores and information asymmetry

---

## Self-Critique: OBSERVE

- **Strengths:**
  - Clear problem statement with real-world applications
  - Detailed technical approach with step-by-step strategy
  - Initial results show promise with MCTS outperforming naive benchmarks
- **Areas for Improvement:**
  - Mathematical formulation could be more rigorous
  - Technical details need more conciseness
  - Resource and computational requirements must be more specific

---

## Self-Critique: ORIENT & DECIDE

- **Key Decisions:**
  - Refine the optimization objective with stricter mathematical notation
  - Simplify technical approach details
  - Specify computational resources and test with small datasets
  - Enhance quantitative analysis in results
- **Open Questions:**
  - How feasible is the project with current compute resources?
  - Are there alternative RL techniques better suited for this problem?
  - How can the auction be more rigorously formulated?
  - How to effectively simulate hidden information in an AlphaGo Zero–style system?

---

## Self-Critique: ACT

- **Immediate Next Steps:**
  - Refactor code to focus on mean outcomes (ignoring variance)
  - Develop a simplified AlphaGo Zero–like self-play pipeline
  - Test and validate with smaller data sets before scaling
  - Explore alternative auction models (e.g., Vickrey auctions) for efficiency

---

## Final Thoughts & Questions

- **Final Considerations:**
  - Balancing model complexity with simulation time
  - Incorporating hidden information without excessive rigidity in assumptions
- **Questions for Feedback:**
  1. How feasible is the current project with regard to compute requirements?
  2. What other RL techniques could be explored for this type of auction?
  3. Are there better ways to mathematically formulate this problem?
  4. How best to model hidden information in an AlphaGo Zero–esque system?

---

# Thank You!

- **Questions?**
- **Feedback & Suggestions Welcome!**
