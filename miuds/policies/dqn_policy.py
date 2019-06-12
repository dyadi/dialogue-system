from .policy import RLPolicy
from .networks import MLP
from .replay_buffer import ReplayBuffer
from .schedule import LinearSchedule
import miuds.config
from miuds.torch_utils import to_tensor
import numpy as np
import torch
import torch.nn.functional as F
from collections import namedtuple


Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward', 'done'),
    defaults=(None, ) * 5)


class DQNPolicy(RLPolicy):
    """Deep-Q-Network Policy
    """
    def __init__(self, intent_set, transitive_intent_set, slot_set,
                 hidden_size=128):
        super(DQNPolicy, self).__init__(
                intent_set, transitive_intent_set, slot_set)
        self.network = MLP(
                input_size=self.input_size,
                output_size=self.output_size,
                hidden_size=hidden_size)
        self.device = miuds.config.device

    def set_train_mode(self, batch_size=32, gamma=0.9, replay_buffer_size=2000,
                       network_update_freq=1, target_update_freq=10,
                       optimizer='Adam', optimizer_args={'lr': 1e-2},
                       eps_greedy=('Linear', (2000, 0.0, 1.0))):
        """Set policy to train mode and set hyperparamters
        """
        super(DQNPolicy, self).set_train_mode()
        self.replay_buffer = ReplayBuffer(
                capacity=replay_buffer_size,
                transition=Transition)
        self.batch_size = batch_size
        self.gamma = gamma
        self.network_update_freq = network_update_freq
        self.target_update_freq = target_update_freq
        if eps_greedy is not None:
            self.eps_greedy = LinearSchedule(*eps_greedy[1])
        else:
            eps_greedy = eps_greedy

        if isinstance(optimizer, str):
            self.optimizer = getattr(torch.optim, optimizer)(
                    self.network.parameters(), **optimizer_args)
        else:
            self.optimizer = optimizer(self.network.parameters(),
                                       **optimizer_args)

        self.target_network = MLP(
                input_size=self.input_size,
                output_size=self.output_size,
                hidden_size=self.hidden_size)
        self.target_network.load_state_dict(self.network.state_dict())
        self.num_step = 0
        self.last_transition = Transition()

    @staticmethod
    def train_episode(dialog_manager, user):
        """Train episode
        ### WARNING ###
        ### THIS METHOD IS ONLY FOR TESTING ###
        """
        # TODO: make this work for high-level api
        policy = dialog_manager.agent.policy
        if not policy.train_mode:
            self.set_train_mode()
        episode_reward = 0
        user_action = user.initial_episode()
        done = False
        while not done:
            action = dialog_manager(user_action)
            user_action, reward, done, info = user.step(action)
            self.last_transition.reward = reward
            self.last_transition.done = done
            episode_reward += reward
            policy.num_step += 1
            if policy.num_step % policy.network_update_freq == 0:
                self._update()
            if policy.num_step % policy.target_update_freq == 0:
                policy.target_network.load_state_dict(
                        self.network.state_dict())
        return episode_reward

    def _update(self):
        assert (self.train_mode)
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = self.replay_buffer.sample(self.batch_size)

        state = to_tensor(batch.state)
        next_state = to_tensor(batch.next_state)
        action = to_tensor(batch.action)
        reward = to_tensor(batch.reward)
        done = to_tensor(batch.done.astype(np.float32))

        q = self.network(state)
        with torch.no_grad():
            q_prime = self.target_network(next_state)
            q_prime = q_prime.max(1)[0].unsqueeze(1)
            q_target = reward + self.gamma * q_prime * (1 - done)
        q = torch.gather(q, 1, action)
        loss = F.mse_loss(q, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def make_action(self, state):
        state = self.encode_dialog_state(state)

        if self.train_mode and self.eps_greedy is not None and \
           torch.rand(1).item() < self.eps_greedy(self.num_step):
            action = np.random.randint(self.output_size)
        else:
            state = to_tensor(state).unsqueeze(0)
            q_values = self.network(state)
            action = q_values.argmax().item()

        # Store transition
        # TODO: This not work now
        if self.train_mode:
            self.last_transition.next_state = state
            if self.last_transition.state is not None:
                self.replay_buffer.push(self.last_transition)
            self.last_transition.state = state
            self.last_transition.action = action

        action = self.decode_dialog_action(action)
        return action

    def save(self, path):
        state_dict = {k: v.cpu() for k, v in self.network.state_dict().items()}
        model = {
                    'state_dict': state_dict
                }
        torch.save(model, path)

    def load(self, path):
        model = torch.load(path)
        self.network.load_state_dict(model['state_dict'])

    @property
    def replay_buffer_size(self):
        return len(self.replay_buffer)

    @property
    def hidden_size(self):
        return self.network.hidden_size
