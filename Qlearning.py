import numpy as np
import random


class qLearner:
    def __init__(self, epsilon, learning_rate, gamma, policy=[0, 0]):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q = np.array(policy)

    def take_action(self):
        if random.choices([0, 1], weights=(self.epsilon * 100, (1 - self.epsilon) * 100))[0] == 0:
            return random.choice([0, 1])
        else:
            return np.argmax(self.Q)

    def update_policy(self):
        pass


class environment:
    def __init__(self,
                 num_of_agents,
                 agents_graph,
                 num_of_train_episodes=2000,
                 num_of_eval_episodes=500):
        self.agents = [qLearner() for i in range(num_of_agents)]
        self.graph = agents_graph
        self.num_of_train_episodes = num_of_train_episodes
        self.num_of_eval_episodes = num_of_eval_episodes

    def get_neighborhood(self, agent_num):
        return list(np.where(self.graph[agent_num] == 1)[0])

    def observe(self, agent_action, neighborhood_actions):
        reward = neighborhood_actions.count(agent_action)
        return reward

    def learn(self, train = True):

        if train:
            num_of_episodes = self.num_of_train_episodes #TODO: set training params
        else:
            num_of_episodes = self.num_of_eval_episodes #TODO: set after-training params

        for _ in num_of_episodes:
            actions = [agent.take_action() for agent in self.agents]

            rewards = [self.observe(actions[i], list(map(actions.__getitem__, self.get_neighborhood(i))))
                       for i in range(len(self.agents))]

            for agent, r in self.agents, rewards :
                agent.update_policy(r)


    def plot(self):
        pass


graph = [[0 for i in range(7)] for i in range(7)]

graph[0][3] = 1
graph[3][0] = 1

graph[0][2] = 1
graph[2][0] = 1

graph[1][2] = 1
graph[2][1] = 1

graph[0][3] = 1
graph[3][0] = 1

graph[2][4] = 1
graph[4][2] = 1

graph[3][4] = 1
graph[4][3] = 1

graph[3][5] = 1
graph[5][3] = 1

graph[3][6] = 1
graph[6][3] = 1

print(graph)
