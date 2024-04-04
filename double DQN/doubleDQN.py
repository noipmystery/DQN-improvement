import torch
import torch.optim as optim
import torch.nn.functional as F
from utils.network import Network


class DoubleDQN:
    """
    Double DQN
    """
    def __init__(self, in_channels, num_actions, lr, gamma, epsilon, alpha):
        self.gamma = gamma
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.q_network = Network(in_channels=in_channels, num_actions=num_actions)
        self.target_q_network = Network(in_channels=in_channels, num_actions=num_actions)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=lr, eps=epsilon, alpha=alpha)

    def calculate_loss(self, states, actions, rewards, next_states, dones):
        """
        y(state) = reward if done
                 = reward + gamma * target(next_state, argmax_a q(next_state, a))
        loss = (y(state) - q(state, action)) ^ 2
        """
        tmp = self.q_network(states)
        nxt = self.target_q_network(next_states)
        rewards = rewards.to(self.device)
        q_values = tmp[range(states.shape[0]), actions.long()]
        q_action = torch.argmax(tmp, dim=1).to(self.device)
        target_values = nxt[range(states.shape[0]), q_action.long()]
        default = rewards + self.gamma * target_values
        target = torch.where(dones.to(self.device), rewards, default).to(self.device).detach()
        return F.mse_loss(target, q_values)

    def update(self, states, actions, rewards, next_states, dones):
        loss = self.calculate_loss(states, actions, rewards, next_states, dones)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
