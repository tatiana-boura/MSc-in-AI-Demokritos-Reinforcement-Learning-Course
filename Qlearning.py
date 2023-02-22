import numpy as np
import random
import matplotlib.pyplot as plt


class qLearner:
    def __init__(self, learning_rate, gamma, epsilon=1.):
        self.agent_type = None
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q = np.zeros((2, 1))
        self.Q_history = []
        self.R_history=[]

    def take_action(self):
        if random.choices([0, 1], weights=(self.epsilon * 100, (1 - self.epsilon) * 100))[0] == 0:
            return random.choice([0, 1])
        else:
            return np.argmax(self.Q)

    def update_policy(self, reward, action):
        self.Q[action] = self.Q[action] + self.learning_rate * (reward + (self.gamma * np.max(self.Q)) - self.Q[action])

    def set_type(self, agent_type):
        self.agent_type = agent_type
        if self.agent_type == 'X':
            self.Q[0][0] = 1.
        else:
            self.Q[1][0] = 1.

    def get_policy(self):
        return np.argmax(self.Q)


class environment:
    def __init__(self,
                 num_of_agents,
                 agents_graph,
                 num_of_train_episodes=2000,
                 num_of_eval_episodes=500,
                 num_of_timesteps=100):
        self.agents = [qLearner(learning_rate=0.8, gamma=0.9) for i in range(num_of_agents)]
        self.graph = agents_graph
        self.num_of_train_episodes = num_of_train_episodes
        self.num_of_eval_episodes = num_of_eval_episodes
        self.num_of_agents = num_of_agents
        self.num_of_timesteps = num_of_timesteps

    def get_neighborhood(self, agent_num):
        neighborhood = list(np.where(np.array(self.graph[agent_num]) == 1)[0])
        return neighborhood

    def observe(self, agent_action, neighborhood_actions):
        reward = neighborhood_actions.count(agent_action)
        return reward

    def learn(self, train=True):

        if train:
            num_of_episodes = self.num_of_train_episodes
        else:
            for agent in self.agents:
                agent.epsilon = 0.
            num_of_episodes = self.num_of_eval_episodes

        for curr_episode in range(num_of_episodes):
            for t in range(self.num_of_timesteps):
                actions = [agent.take_action() for agent in self.agents]  # joint action

                rewards = [self.observe(actions[i], list(map(actions.__getitem__, self.get_neighborhood(i))))
                           for i in range(len(self.agents))]

                if curr_episode == 100 and train:
                    for agent, r,i in zip(game_env.agents, rewards, range(self.num_of_agents)):
                        agent.R_history.append(r/(len(self.get_neighborhood(i))))

                for agent, r, a in zip(self.agents, rewards, actions):
                    agent.update_policy(r, a)

            if train and curr_episode % 10 == 0 and curr_episode != 0:
                for agent in self.agents:
                    if agent.epsilon >= 0.01:
                        agent.epsilon -= 0.01

            for agent in self.agents:
                agent.Q_history.append([agent.Q[0][0], agent.Q[1][0]])

    def plot(self):
        for agent, i in zip(game_env.agents, range(len(game_env.agents))):
            action_1 = [item[0] for item in agent.Q_history]
            action_2 = [item[1] for item in agent.Q_history]

            plt.plot(list(range(self.num_of_train_episodes + self.num_of_eval_episodes)), action_1, label="action1")
            plt.plot(list(range(self.num_of_train_episodes + self.num_of_eval_episodes)), action_2, label="action2")
            plt.legend()
            plt.title(f"Q values of agent {i}, type = '{agent.agent_type}'")
            plt.savefig(f"Q_values_agent_{i}_type_{agent.agent_type}")
            plt.show()

            plt.plot(list(range(self.num_of_timesteps)) , agent.R_history)

            plt.title(f"Rewards {i}, type = '{agent.agent_type}'")
            plt.savefig(f"Rewards_{i}_type_{agent.agent_type}")
            plt.show()


    def __repr__(self):
        result = ""
        for i in range(self.num_of_agents):
            result += "\n"
            result += f"Agent {i} is connected with {self.get_neighborhood(i)} and is type of '{self.agents[i].agent_type}'"
        return result


graph = [[0 for i in range(7)] for i in range(7)]

graph[1][3] = 1
graph[3][1] = 1

graph[0][3] = 1
graph[3][0] = 1

graph[0][2] = 1
graph[2][0] = 1

graph[1][2] = 1
graph[2][1] = 1

graph[2][4] = 1
graph[4][2] = 1

graph[3][4] = 1
graph[4][3] = 1

graph[3][5] = 1
graph[5][3] = 1

graph[3][6] = 1
graph[6][3] = 1

game_env = environment(num_of_agents=7, agents_graph=graph, num_of_train_episodes=2000,num_of_timesteps=100, num_of_eval_episodes=500)
game_env.agents[0].set_type('Y')
game_env.agents[1].set_type('X')
game_env.agents[2].set_type('X')
game_env.agents[3].set_type('Y')
game_env.agents[4].set_type('X')
game_env.agents[5].set_type('Y')
game_env.agents[6].set_type('X')

print("\nEnvironment:\n")

print(game_env)

print("\nBefore Training:\n")

for agent, i in zip(game_env.agents, range(len(game_env.agents))):
    print(f"Agent's {i} policy: {agent.get_policy()}")

game_env.learn(train=True)
game_env.learn(train=False)

print("\nAfter Training:\n")

for agent, i in zip(game_env.agents, range(len(game_env.agents))):
    print(f"Agent's {i} policy: {agent.get_policy()}")

game_env.plot()
