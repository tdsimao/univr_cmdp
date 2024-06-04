import unittest
import gym

import numpy as np

from lp import LinearProgrammingPlanner
# from lp_solution import LinearProgrammingPlanner
from util.mdp import monte_carlo_evaluation


def _solve(env, cost_bound, horizon=20):
    lp_agent = LinearProgrammingPlanner.from_discrete_env(env, cost_bound=cost_bound, horizon=horizon)
    np.random.seed(42)
    lp_agent.solve()
    expected_value = lp_agent.expected_value(env.isd)
    expected_cost = lp_agent.get_expected_cost()
    return lp_agent, expected_value, expected_cost


class TestTrainingAndEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("gym_factored:cliff_walking_cost-v0", num_rows=4, num_cols=6)

    def test_evaluation_unbounded_agent(self):
        agent, expected_value, expected_cost = _solve(self.env, cost_bound=None, horizon=20)
        ret, cost, length, fail = monte_carlo_evaluation(self.env, agent, agent.horizon, 1, 3)
        self.assertEqual(ret, -7)
        self.assertEqual(length, 7)
        self.assertEqual(cost, 8)
        self.assertEqual(fail, 0)

    def test_evaluation_bounded_agent_tight(self):
        agent, expected_value, expected_cost = _solve(self.env, cost_bound=0, horizon=20)
        ret, cost, length, fail = monte_carlo_evaluation(self.env, agent, agent.horizon, 1, 3)
        self.assertEqual(ret, -11)
        self.assertEqual(length, 11)
        self.assertEqual(cost, 0)
        self.assertEqual(fail, 0)

    def test_evaluation_bounded_agent(self):
        agent, expected_value, expected_cost = _solve(self.env, cost_bound=2, horizon=20)   
        ret, cost, length, fail = monte_carlo_evaluation(self.env, agent, agent.horizon, 1, 1000)
        self.assertAlmostEqual(cost, 2, delta=0.1)
        self.assertAlmostEqual(ret, -10, delta=1)
        self.assertAlmostEqual(length, 10, delta=1)
        self.assertAlmostEqual(fail, 0, delta=0)


class TestLPAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("gym_factored:chain2d-v0")

    def test_lp_agent_horizon_zero(self):
        _, expected_value, expected_cost = _solve(self.env, cost_bound=None, horizon=0)
        self.assertAlmostEqual(expected_value, 0, places=2)
        self.assertAlmostEqual(expected_cost, 0)

    def test_lp_agent_horizon_one(self):
        _, expected_value, expected_cost = _solve(self.env, cost_bound=None, horizon=1)
        self.assertAlmostEqual(expected_value, 4, places=2)
        self.assertAlmostEqual(expected_cost, 1)

    def test_lp_agent_horizon_two(self):
        _, expected_value, expected_cost = _solve(self.env, cost_bound=None, horizon=2)
        self.assertAlmostEqual(expected_value, 8.25, places=2)
        self.assertAlmostEqual(expected_cost, 1.75)

    def test_lp_agent_unbounded(self):
        _, expected_value, expected_cost = _solve(self.env, cost_bound=None, horizon=3)
        self.assertAlmostEqual(expected_value, 10, places=2)
        self.assertAlmostEqual(expected_cost, 2)

    def test_lp_agent_bounded(self):
        _, expected_value, expected_cost = _solve(self.env, cost_bound=0, horizon=3)
        self.assertAlmostEqual(expected_value, 1, places=2)
        self.assertAlmostEqual(expected_cost, 0, places=2)


class TestLPAgentCliff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("gym_factored:cliff_walking_cost-v0", num_rows=3, num_cols=4)

    def test_lp_agent_unbounded(self):
        _, expected_value, expected_cost = _solve(self.env, cost_bound=None)
        self.assertAlmostEqual(expected_value, -5, places=2)
        self.assertAlmostEqual(expected_cost, 2, places=2)

    def test_lp_agent_bounded_0(self):
        _, expected_value, expected_cost = _solve(self.env, cost_bound=0)
        self.assertAlmostEqual(expected_value, -7, places=2)
        self.assertAlmostEqual(expected_cost, 0, places=2)

    def test_lp_agent_bounded_1(self):
        _, expected_value, expected_cost = _solve(self.env, cost_bound=1)
        self.assertAlmostEqual(expected_value, -6, places=2)
        self.assertAlmostEqual(expected_cost, 1, places=2)

    def test_lp_agent_bounded_2(self):
        _, expected_value, expected_cost = _solve(self.env, cost_bound=2)
        self.assertAlmostEqual(expected_value, -5, places=2)
        self.assertAlmostEqual(expected_cost, 2, places=2)


class TestLPAgentCliffLarge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("gym_factored:cliff_walking_cost-v0")

    def test_lp_agent_unbounded(self):
        _, expected_value, expected_cost = _solve(self.env, cost_bound=None)
        self.assertAlmostEqual(expected_value, -13, places=2)
        self.assertAlmostEqual(expected_cost, 20, places=2)

    def test_lp_agent_bounded(self):
        _, expected_value, expected_cost = _solve(self.env, cost_bound=0, horizon=20)
        self.assertAlmostEqual(expected_value, -17, places=2)
        self.assertAlmostEqual(expected_cost, 0, places=2)



if __name__ == "__main__":
    unittest.main()
