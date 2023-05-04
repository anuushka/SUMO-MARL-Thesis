from re import A
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import numpy as np
import random
import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


class Net(nn.Module):
    def __init__(self, state_size, action_size):
        # модел нь torch.nn.Module - оос удамшиж байгаа учир заавал эх object - ийг эхлүүлнэ
        super(Net, self).__init__()
        # оролтонд төлөвийн хэмжээс байх ба сургалтын явцад төлөвийн мэдээллүүдийг өгнө
        self.fc1 = nn.Linear(state_size, 120)
        # гүн давхрагын хэсэг байх ба 2 болон түүнээс дээш байж болно
        self.fc2 = nn.Linear(120, 50)
        # гаралт нь гүн давхрагын гаралтыг оролтонд авч үйлдлийн хэмжээгээр гаргахаар тохируулж өгнө.
        self.fc3 = nn.Linear(50, action_size)

    # сургалтын шууд алхам буюу модел нь мөр мөрөөр дарааллуулан биелэгдэхээр бичнэ
    def forward(self, state):
        # сургалтын явцад моделийн оролтонд тухайн төлөвийн мэдээллийг эхний давхрагад оруулна
        x = self.fc1(state)
        x = F.relu(x)       # гүн давхрагын хэсгийг энд бичнэ
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)  # гаралтын үйлдлийн магадлалыг буцаана


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


# reinforcement сургалтын явцад санах ойд өмнөх үйлдлүүдийн үр дүнг бичиж хадгалах шаардлагатай бөгөөд энэ санах ойгоос модел суралцдаг
BUFFER_SIZE = int(1e5)
# санах ойгоос хэчнээн үйлдлүүдийг зэрэг модел сургалтанд ашиглахыг заана
BATCH_SIZE = 64
GAMMA = 0.99            #
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
device = 'cpu'


class Agent():
    """Орчинтой хамтран ажиллах боломжтой агент."""

    def __init__(self, state_size, action_size, seed, alpha, gamma):
        """Агентийг эхлүүлэх

        Params
        ======
            state_size (int): төлөвийн хэмжээс
            action_size (int): үйлдлийн хэмжээс
            seed (int): санамсаргүй утга авах
        """
        self.state_size = state_size
        # print(state_size)
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.alpha = alpha
        self.gamma = gamma

        # Моделийг зарлах
        # энд шууд сургалтын моделийг зарлана уу
        self.qnetwork_local = Net(self.state_size, self.action_size).to(device)
        # энд туршлагын моделийг зарлана уу
        self.qnetwork_target = Net(self.state_size, self.action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # санах ойг зарлах
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # хугацааг эхлүүлэх (мөн алхам бүрд хугацааг ахиулах)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # үйлдэл бүрийг санах ойд хадгалах хэсэг
        self.memory.add(state, action, reward, next_state, done)

        # хугацааны алхам нэмэгдүүлэх
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            #  хэрэв санах ойд хангалттай туршлага хуримтлагдсан бол санамсаргүй бүлгийг сонгон сургах
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()  # санах ойгоос санамсаргүй бүлгийг сонгох
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """ өгөгдсөн төлөвт тохирох үйлдлийг буцааж илгээх

        Params
        ======
            state (array_like): одоогийн төлөв
            eps (float): туршлага хуримтлуулах утгыг хэмжээ
        """
        state = torch.tensor(
            np.array(state))  # одоогийн төлөвийг сургалтанд бэлтгэх
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # сурсан моделоос үйлдэл сонгох эсвэл санамсаргүй үйлдэл сонгох
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # санамсаргүй үйлдлийг үйлдлийн хүснэгтээс сонгох
            return random.choice(range(self.action_size))

    def learn(self, experiences, gamma):
        """өгөгдсөн туршлагад сургалт хийх.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples хэлбэрийн туршлагууд өгөгдөнө
            gamma (float): буурах фактор
        """
        # print('learning working')
        # Санамсаргүй утгуудыг туршлагаас сонгон авах
        states, actions, rewards, next_states, dones = experiences
        # states = states.to(device)
        # actions = actions.to(device)
        # Алдааг тооцоолох болон багасгах
        # Туршлагын моделоос дараагийн төлөвийн таамаглалыг гарган авах
        q_targets_next = self.qnetwork_target(
            next_states).detach().max(1)[0].unsqueeze(1)
        # Туршлагын моделоос хүрэх үр дүнгийн утгыг bellman тэгшитгэл ашиглан авах
        q_targets = rewards + self.gamma * q_targets_next * (1 - dones)
        # шууд сургалтын моделоос таамаглалыг авах
        q_expected = self.qnetwork_local(states).gather(1, actions)

        # Алдааг тооцоолох моделийг шинэчлэх
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- сурсан шууд сургалтын моделийг ашиглан туршлагын моделийг шинэчлэх ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """турлагийн моделыг шинэчлэх.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): моделийн жингүүдийг хуулах
            target_model (PyTorch model): моделийн жингүүдийг бичих
            tau (float): туршлагын моделийг шинэчлэх хэмжээ
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)
