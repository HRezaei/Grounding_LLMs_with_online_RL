import json
import random
import unittest

import gym
import babyai_text


class TestBabyAISeeding(unittest.TestCase):

    def test_env_reproducibility(self):
        """
        This function tests whether the env and mission remain the same for a specific seed?
        So to make sure we end up with the same game if we use the same seed
        It also makes sure that during a single run, resetting the env generates the same game.
        """
        with open("seed_samples.json", "r") as fp:
            samples = json.load(fp)

        name_env = 'BabyAI-MixedTrainLocal-v0'
        for seed, game in samples.items():
            seed = int(seed)
            env = gym.make(name_env, seed=seed)  # , scenario='goto')
            env.seed(seed)
            env.reset()
            mission = env.mission
            board = str(env.env.env)
            assert mission == game["mission"], f"{mission=}, {game['mission']=}, {seed=}"
            assert board == game["board"]

            # In the same run, try to reset the env again, and it must generate the same game:
            for _ in range(100):
                env.seed(seed)
                env.reset()
                assert env.mission == mission
                assert str(env.env.env) == board


if __name__ == '__main__':
    unittest.main()
