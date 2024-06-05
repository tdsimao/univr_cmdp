# CMDPs

## modules

- `agents`: model based RL agents that interact with the environment.
- `planners`: the planners used by the RL agents to compute the policy in each episode.
- `scripts`: each file is related to one of the experiments from the paper.
- `tests`: mostly unittest scripts.
- `util`: contains common scripts to train an RL agent and evaluate a policy.


## lp solver

By default, the code uses [`gurobipy`](https://www.gurobi.com/) if found, otherwise it uses [`cvxpy`](https://www.cvxpy.org/).


## usage


1. create a virtualenv and activate it
```bash
cd univr_cmdp/
python3 -m venv .venv
source .venv/bin/activate
```

2. install the dependencies
```bash
pip install -r requirements.txt
```

3. test the installation
    ```bash
    python test.py 
    ```
    ```
    state [0, 4, 0, 3, 8]   action 2    reward=-1   cost=0
    state [0, 3, 0, 3, 7]   action 3    reward=-1   cost=0
    state [0, 3, 0, 3, 6]   action 6    reward=-10  cost=0
    state [0, 4, 0, 3, 5]   action 2    reward=-1   cost=0
    state [0, 4, 0, 3, 4]   action 1    reward=-1   cost=0
    state [0, 4, 0, 3, 3]   action 5    reward=-10  cost=0
    state [1, 4, 0, 3, 2]   action 0    reward=-1   cost=0
    state [2, 4, 0, 3, 1]   action 0    reward=-1   cost=0
    state [2, 4, 0, 3, 0]   action 5    reward=-20  cost=1
    ```


## Exercises

The file `lp.py` contains the code necessary to solve a MDP using a linear programming approach.
The goal is to adapt it to solve a CMDP and extract a stochastic policy from the solution.

1. Implement the function `set_policy` to extract the policy in a state at a given time step.
2. Implement the missing code to compute the expected accumulated cost in the function `instantiate_lp` 
3. Compare the occupancy measure of the cliff walk environment given three cost bounds (0, 2, and ∞). Suggestion: use a heatmap to vizualize the results.
4. Evaluate the impact of the cost bound on the expected return. Suggestion: make a line plot with the cost bound in the x-axis and the expected return in the y-axis.



# run the main experiments

```bash
python main.py --verbose
```
