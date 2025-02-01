# Week 3 Submission

### Project Statement

**What am I optimizing**

My project idea focuses on solving a dynamic decision-making problem in the context of a simplified fantasy hockey auction. The setup for the auction is as follows:

League Structure: The league consists of $n$ members, each provided with $t$ tokens to spend in an auction.

Team Composition: Each member must build a team of $o + d + g$  players: $o$ forwards, $d$ defensemen, and $g$ goalies.

Player Valuation: For simplicity's sake each player’s value is represented as independent random variables with a known mean and variance - all members agree upon these distributions. These values are realized after the auction is completed.

Later on I may try to model scenarios where there is disagreement amount player distributions.

Auction Mechanics:

Players are randomly nominated for auction each round.

The bidding proceeds cyclically, where each member decides whether to bid up or pass.

Bidding up increases the current price by one token and temporarily assigns the player to the bidder.

If a member passes, they are no longer allowed to bid on the player for that round.

The auction ends for a player when only one member remains in the bidding, and the player is awarded to them for the final price.

Objective: My overarching goal is to maximize rank, determined by the cumulative realized value of the drafted players compared to other members. This creates a complex strategic environment where decisions need to account for budget constraints, remaining roster spots, and the uncertainty in player valuations.

That is I want to create some function $f: S \rightarrow \{0,1\}$ where $S$ represents the state of the current auction. $f$ informs the current player whether they should raise or fold given the current auction state.

**Why this problem matters**

The honest answer here is because it would help me draft my fantasy team next year. However, this problem has many other more interesting applications.

I envision the true application of this project in other areas is to identify the valuation of different objects. For example, if we ran identical versions of the agent I developed against each other, the final bid values for each player could be a valuation for the players.

Modified versions of this project could be used in, for example:

1. **Finance**: Different market participants have different risk limits. The returns in equilibirum needed to fit these risk profiles could potentially be modelled by a similar auction.

2. **Resource Allocation in Project Management**: In project management, resources (such as time, money, and personnel) must be allocated to various tasks. The optimization problem can help in making decisions on how to allocate resources efficiently to maximize project success. The valulation of different employees and resources could be determined by a similar auction technique.

5. **Ad Placement in Digital Marketing**: Advertisers must decide how to allocate their budget across different ad placements to maximize the return on investment. The optimization problem can help in determining the best strategy for ad placement.

6. **Healthcare Resource Allocation**: In healthcare, resources such as medical staff, equipment, and medications need to be allocated efficiently to maximize patient outcomes. The optimization problem can aid in making these critical decisions in an equitable manner.

I would caution that the exact formualtion of the auction described above is unlikely to be of use in these fields. However, modified auctions will likely be able to be solved using similar techniques as I use.

**What data might I need**

For now, I am using my intuition to generate distributions of players for the auction. Later on I may scrape historical NHL data as well as common fantasy hockey scoring systems to generate better distributions.

**How will I measure success?**

I will use dummy agents that use very naive valuation functions to guide their valulations as a benchmark for my model.

**What could go wrong?**

I don't have much experience with RL - I might not be able to identify the needed solutions imemdiately or could be vastly underestimating the difficulty of this problem both in terms of technical complexity and compute needed.

For both of these my solution will be to simplify the auction set up I use. A dummy model might still yield relevant insights that are applicable to the actual auction.

**Extensions I'm hoping to grow to**

1. **Disagreements over players' outcomes**: A more realistic model for fantasy hockey drafts has players disagreeing on athlete valuations. Something along the lines of players means, variances being random variables themselves and each player's priors heading in being samples from those could be interesting. Players would need to learn how to incorporate information from how others are bidding into their priors during the auction.

2. **Assymetric Information**: An even more realistic model is one where certain players have absolutely no information on player valuations. Maybe because they joined the league last minute. I want to model a scenario where players either have information or no information and must behave optimally. Ideally I would also like to have degrees of information for players - and also specializations. Perhaps, players have some information on some players and not others?

I think one consideration for this latter approach is how we model hidden information we are simulating. For example, players prior bidding patterns could reveal information about how informed they are. Do we need to model this into our MCTS?

I wonder if as a baseline we could ignore that information. Perhaps we assume that information can't be deduced? Or we could try to have the RL agent also try to predict informativeness - I'm afraid that this might be too complicated, however.

### Technical Approach (1/2 page)
- **Mathematical formulation** 

Note that the auction described above is a finite game. Bids are in integer quantities with hard constraints and there are a finite amount of states that can occur. While infeasible to compute, each version of this game is indeed solvable.

For each state we could then identify if all players played optimally, what our expected rank is. This would be the value function of a given state - we could use this value function to then decide whether we bid up or down.

This optimization problem seeks to fit this value function $V$.

Let $S$ be the state space representing all possible states of the auction, and $A$ be the action space representing the possible actions (bid up or pass). The value function $V(s)$ represents the expected rank given the state $s \in S$.

The objective is to find a policy $\pi: S \rightarrow A$ that maximizes the expected rank. The policy $\pi$ determines the action to take in each state to maximize the cumulative realized value of the drafted players.

Mathematically, the optimization problem can be formulated as:

$$
\max_{\pi} \mathbb{E}\left[\sum_{t=0}^{T} V(s_t) \mid \pi \right]
$$

where $T$ is the total number of rounds in the auction, $s_t$ is the state at round $t$, and $\pi$ is the policy.

The value function $V(s)$ can be estimated using reinforcement learning techniques, where the agent learns to approximate the value of each state through interactions with the environment.


- **Algorithm/approach choice and justification** 

My plan is to use techniques similar to those employed in Alpha Go Zero (https://www.nature.com/articles/nature24270)

I am indexing highly on self play to learn and minimal human guidance on what a good strategy is.

While the paper's training took a significant amount of time - my game set up is simpler and GPUs are significantly more powerful today than they were at the time of writing.

A lot of LLMs suggested looking into stochastic programming for this exercise. This made little sense to me as I don't believe this is a stochastic problem.

There is no stochasticity during the actual auction - all of it is in how the players realize their values post auction. As such our scoring at the end can be determined by just simulating the seasons according to the apriori information on player distributions.

Each player basically is building a normal distribution with some mean and variance and the goal is for their distribution to generally rank well.

Other classical techniques such as Q-learning and SARSA made sense, but they may not be as effective for this problem due to the complexity and large state space.

Heuristic techniques would not really solve the problem as I want.

- **PyTorch implementation strategy** 

Note to start I have not set anything up in PyTorch. Rather I created a Gymnasium environment, a naive monte carlo tree search system and have evaluated that against benchmark agents.

I wanted to discuss this approach before diving into implementing the AlphaGo Zero approach.

However, eventially I will do the following:

I will implement the reinforcement learning model using PyTorch. The strategy involves the following steps:

1. **Define the Environment**: Create a custom environment that simulates the auction process, including state representation, action space, and reward function. Done via a Gymnasium wrapper.

2. **Policy and Value Network**: Implement a neural network to approximate the policy function $\pi(s)$ and value function $V(s)$. This network will take the current state as input and output the probability of each action (bid up or pass) and the expected rank given the current state.
4. **Training Loop**: Use self-play to generate training data. The agent will play multiple games against itself, updating the policy and value networks based on the outcomes.
5. **Loss Functions**: Define appropriate loss functions for the policy and value networks. The policy loss will be based on the cross-entropy between the predicted and actual actions, while the value loss will be based on the mean squared error between the predicted and actual values.
6. **Optimization**: Use an optimizer like Adam to update the network weights based on the loss functions.
7. **Evaluation**: Periodically evaluate the agent's performance against benchmark agents to measure progress.

- **Validation methods** 

To validate the approach, I will:

1. **Benchmarking**: Compare the performance of the trained agent against dummy agents with naive strategies.
2. **Simulation**: Run multiple simulations of the auction to ensure the agent's decisions lead to higher cumulative realized values.
3. **Cross-Validation**: Use cross-validation techniques to ensure the model generalizes well to different auction scenarios.
4. **A/B Testing**: Conduct A/B testing with different hyperparameters and network architectures to find the optimal configuration.

- **Resource requirements and constraints** 

The primary resources required include:

1. **Computational Resources**: Access to GPUs for training the neural networks efficiently.
2. **Data**: Historical NHL data and fantasy hockey scoring systems for generating realistic player distributions. This is less of a priority as I want the model to generalize well and this problem isn't **that** difficult.
3. **Time**: Sufficient time to train the model and conduct thorough validation.

Constraints to consider:

1. **Compute Time**: Training the model may take a significant amount of time, especially with self-play.
2. **Data Quality**: The accuracy of the model depends on the quality and representativeness of the data used for training.
3. **Model Complexity**: Balancing model complexity with computational feasibility to ensure the model can be trained within a reasonable timeframe.

### Initial Results (1/2 page)
- **Evidence your implementation works** 

My initial work simulates an auction draft with chosen strategies. To make the MCTS run faster, I have chosen a fairly small draft structure of 6 teams, 8 forwards, 4 defensemen and 2 goalies.

Users can try a few basic strategies including an MCTS. While the MCTS does not always "win" the draft - it is usually one of the better strategies.

Some 21 tests are shown in test_outputs - the monte carlo method appears to win in every game it is present in. However there are situations where when multiple monte carlo agents are present one is beaten by another naive strategy.

The makes me more confident that when we use self-play RL with MCTS which will improve the assumptions that we make when simulating the game, we will generate more optimal strategies.

- **Current limitations** 

Currently, I have not implemented any AlphaGo Zero esque intelligent model. I am also assuming very simplified conditions. 

- **Resource usage measurements** 

My MCTS runs 100 simulations per move split over four subprocesses. Combined in general they seem to be using about 50% of CPU with around 80 mb per process, so even for a simplified version it is non trivial.

![BTOP Panel while running](./images/btop_week_3.png)

- **Unexpected challenges** 

I wasn't expecting simulating a single episode (draft) to take this long. I am a little bit worried about how much compute I will need for this game. It can take up to 15 minutes to simulate even a simplified draft.

I wonder if I eliminate the variance component of the game and focus solely on the raw mean score - with more complexities surrounding information assymetry - I will get a faster game to simulate that better mimics the real world scenario. In my tests, variance seems to play very little role in affecting the order of ranks across the agents. In the real world the distributions of performances for players are not normal either so the value add of modelling variances may be unneccessary.

### Next Steps (1/2 page)
- **Immediate improvements needed** 

I think I will refactor my code to ignore variances. The end goal is to start modelling uncertainty around playe realizations with noise in the priors agents enter with as well as differing levels of information around players.

Then it is time to start building a simplified Alpha Go Zero esque pipeline.

- **Technical challenges to address** 

I need to make sure the pipeline for a self-play reinforcement learning with MCTS working.

- **Questions you need help with** 

1. How feasible is my current project vis a vis compute
2. What are some other RL techniques to explore? I couldn't find other systems that would work for my problem well.
3. Better ways to formulate my problem?
4. What is the best way to apply an Alpha Go Zero esque system with hidden information. Is making rigid assumptions about things such as the distribution of players in general ok? I think I would need these to simulate potential hidden states.


- **Alternative approaches to try** 
Maybe if I can define a class of bidding strategies an evolutionary algorithm might work. I am skeptical of this though.

Maybe there is a different way to formulate this auction. Another idea is to model each player's auction as a Vickrey auction. So each player simotaneously makes a bid from 1 - 100 and the winner (highest bid) pays the second persons price plus 1. This might be a lot quicker to simulate and scale better while being essentially the same.

Downsides are that we lose information on which players have been bidding during the process and our action space is more complex. I don't think the set up with assymetric information yields a lot if you have to bid simotaneously as informed players have less information leakage.
- **What you've learned so far** 
1. Variance and uncertainty in player relaizations by treating them as a random variable matters little. This just adds unneccessary complexity to the game.
2. The game tree is very complex and simulating it takes time.


## Self-Critique

### OBSERVE
- Initial reactions: The report is comprehensive but could benefit from more clarity in certain sections.
- Questions: Are the mathematical formulations and technical approaches clearly explained? Is the problem statement compelling enough?

### ORIENT
- **Strengths**
  - Clear problem statement with concrete real-world impact. I think I have a good idea of what problem I want to solve. 
  - Detailed technical approach with step-by-step implementation strategy. Currently, I have a pretty set path to follow.
  - Consideration of various applications and potential extensions. I think I've thought about the trajectory of this project and how I want it to expand.
  - Initial results section provides evidence of implementation and resource usage. While I have not used PyTorch yet, I have set up a basic system with the game environment.

- **Areas for Improvement**
  - Mathematical formulation needs more rigor - currently just intuitive description. I wasn't sure how to best formulate this.
  - The technical approach section could be more concise and focused. I did use this document a lot to think of ideas and some of that lack of clarity is visible.
  - Resource requirements and constraints need more specific details. I don't have a lot of experience with this and wasn't sure how to approach it.
  - Initial results section could include more quantitative data and analysis. I need to have a better way to measure model success.

- **Critical Risks/Assumptions**
  - Assuming the computational resources will be sufficient for training the model. Need to test with realistic data size and compute time.
  - Assuming the simplified auction model will generalize well to more complex scenarios.

### DECIDE
- **Concrete Next Actions**
  - Write out the optimization objective function with constraints in more detail, including mathematical rigor.
  - Simplify the technical approach section to focus on key points and remove redundant information.
  - Specify the exact computational resources required and test with a small dataset to validate feasibility.
  - Include more quantitative data and analysis in the initial results section to provide stronger evidence of implementation.

### ACT
- **Resource Needs**
  - Access to a GPU for testing the computational feasibility of the model.
  - Guidance on refining the mathematical formulation from a subject matter expert.
