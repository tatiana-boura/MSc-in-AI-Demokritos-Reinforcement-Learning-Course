import numpy as np
import random


class qLearner:
    def __init__(self, learning_rate, gamma, policy=[0., 0.], epsilon=1):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q = np.array(policy)

    def take_action(self):
        if random.choices([0, 1], weights=(self.epsilon * 100, (1 - self.epsilon) * 100))[0] == 0:
            return random.choice([0, 1])
        else:
            return np.argmax(self.Q)

    def update_policy(self, reward, action):
        self.Q[action] = self.Q[action] + self.learning_rate * ( reward + (self.gamma * np.max(self.Q))-self.Q[action])

        #print(f'Q= {self.Q}, reward={reward}')


class environment:
    def __init__(self,
                 num_of_agents,
                 agents_graph,
                 num_of_train_episodes=2000,
                 num_of_eval_episodes=500):
        self.agents = [qLearner(learning_rate=0.9,gamma=0.9) for i in range(num_of_agents)]
        self.graph = agents_graph
        self.num_of_train_episodes = num_of_train_episodes
        self.num_of_eval_episodes = num_of_eval_episodes
        self.num_of_agents = num_of_agents

    def get_neighborhood(self, agent_num):
        neighborhood = list(np.where(np.array(self.graph[agent_num]) == 1)[0])
        return neighborhood

    def observe(self, agent_action, neighborhood_actions):
        reward = neighborhood_actions.count(agent_action)
        return reward

    def learn(self, train = True):

        if train:
            num_of_episodes = self.num_of_train_episodes #TODO: set training params
        else:
            for agent in self.agents:
                agent.epsilon = 0.
            num_of_episodes = self.num_of_eval_episodes #TODO: set after-training params

        for curr_episode in range(num_of_episodes):

            actions = [agent.take_action() for agent in self.agents]

            rewards = [self.observe(actions[i], list(map(actions.__getitem__, self.get_neighborhood(i))))
                       for i in range(len(self.agents))]

            for agent, r, a in zip(self.agents, rewards, actions) :
                agent.update_policy(r,a)

            if train and curr_episode % 100 == 0 and curr_episode != 0:
                for agent in self.agents:
                    #agent.learning_rate *= 1.0
                    if agent.epsilon >= 0.01:
                        agent.epsilon -= 0.01

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
'''
graph = [[0 for i in range(2)] for i in range(2)]

graph[0][1] = 1
graph[1][0] = 1

game_env = environment(2,graph,num_of_train_episodes=5,num_of_eval_episodes=2)
print(graph)
'''

game_env = environment(7,graph)

'''
neigh = game_env.get_neighborhood(0)
print(neigh)
'''
game_env.learn(train=True)
game_env.learn(train=False)


for agent in game_env.agents:
    print(agent.Q)