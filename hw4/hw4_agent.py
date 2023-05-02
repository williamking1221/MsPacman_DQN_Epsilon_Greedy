import argparse
import logging
from re import S
import timeit

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
import numpy as np

import gymnasium as gym

import hw4_utils as utils

from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt


torch.set_num_threads(4)


logging.basicConfig(format=(
        "[%(levelname)s:%(asctime)s] " "%(message)s"), level=logging.INFO)


class DQNNetwork(nn.Module):
    def __init__(self, naction, args):
        super().__init__()
        self.iH, self.iW, self.iC = 210, 160, 3
        self.conv1 = nn.Conv2d(self.iC, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # the flattened size is 8960 assuming dims and convs above
        self.fc1 = nn.Linear(8960*2, 8960)
        self.fc2 = nn.Linear(8960, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, naction)
        self.naction = naction

    def forward(self, X, prev_state=None):
        """
        X - bsz x T x iC x iH x iW observations (in order)
        returns:
          bsz x T x naction action logits, prev_state
        """
        bsz, T = X.size()[:2]

        Z = F.gelu(self.conv3( # bsz*T x hidden_dim x H3 x W3
              F.gelu(self.conv2(
                F.gelu(self.conv1(X.view(-1, self.iC, self.iH, self.iW)))))))

        if prev_state is None:
            # flatten with MLP
            Z = F.gelu(self.fc2(Z.reshape(bsz*T, -1))) # bsz*T x hidden_dim
            Z = Z.view(bsz, T, -1)
            return self.fc3(Z), prev_state
        else:
            if prev_state.shape != X.shape:
                print("Prev State Shape: ")
                print(prev_state.shape)
                print(prev_state)
            Z_prev = F.gelu(self.conv3( # bsz*T x hidden_dim x H3 x W3
                        F.gelu(self.conv2(
                            F.gelu(self.conv1(prev_state.view(-1, self.iC, self.iH, self.iW)))))))
            
            Z = Z.reshape(bsz*T, -1)
            Z_prev = Z_prev.reshape(bsz*T, -1)

            Z_new = torch.cat((Z, Z_prev), 1)
            # print(Z_new.shape)
            Z_new = F.gelu(self.fc2(F.gelu(self.fc1(Z_new)).reshape(bsz*T, -1))) # bsz*T x hidden_dim
            Z_new = Z_new.view(bsz, T, -1)
            return self.fc3(Z_new), prev_state
    
    def get_action(self, x, epsilon, prev_state):
        """
        x - 1 x 1 x ic x iH x iW
        returns:
          int index of action
        """
        # DQN Epsilon-Greedy Mixture -- Half Epsilon
        if random.random() > epsilon / 2:
            logits, prev_state = self.forward(x, prev_state)
            # take highest scoring action
            action = logits.argmax(-1).squeeze().item()
            return action, prev_state
        else:
            action = random.choice(np.arange(self.naction))
        return action, prev_state


class ReplayMemory:
    def __init__(self, args):
        """
        Initialize ReplayMemory Object

        Params:
        args -- arguments passed onto CLI
        """

        self.capacity = args.memory_len
        self.memory = deque(maxlen=self.capacity)
        self.bsz = args.batch_size
        self.experiences = namedtuple("Experience", field_names=["state",
                                                                 "prev_state",
                                                                 "action",
                                                                 "reward",
                                                                 "next_state",
                                                                 "done"])

    def push(self, state, prev_state, action, reward, next_state, done):
        """
        Push an experience to memory
        """
        exp = self.experiences(state, prev_state, action, reward, next_state, done)
        self.memory.append(exp)

    def sample(self):
        """
        Randomly sample a batch of experiences from memory
        """
        experiences = random.sample(self.memory, k=self.bsz)
        
        states = torch.from_numpy(np.vstack([double_expand_dims(e.state) for e in experiences if e is not None])).float()
        prev_states = torch.from_numpy(np.vstack([double_expand_dims(e.prev_state) for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([np.expand_dims(e.action, 0) for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([double_expand_dims(e.next_state) for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([np.expand_dims(e.done, 0) for e in experiences if e is not None]).astype(np.uint8)).float()
        
        return (states, prev_states, actions, rewards, next_states, dones)

    def __len__(self):
        """
        Return the length of memory
        """
        return len(self.memory)


def double_expand_dims(x):
    return np.expand_dims(np.expand_dims(x, 0), 0)


def pg_step(stepidx, model, optimizer, scheduler, envs, observations, prev_state, bsz=4):
    if envs is None:
        envs = [gym.make(args.env) for _ in range(bsz)]
        observations = [env.reset(seed=i)[0] for i, env in enumerate(envs)]
        observations = torch.stack( # bsz x ic x iH x iW -> bsz x 1 x ic x iH x iW
            [utils.preprocess_observation(obs) for obs in observations]).unsqueeze(1)
        prev_state = None

    logits, rewards, actions = [], [], []
    not_terminated = torch.ones(bsz) # agent is still alive
    for t in range(args.unroll_length):
        logits_t, prev_state = model(observations, prev_state) # logits are bsz x 1 x naction
        logits.append(logits_t)
        # sample actions
        actions_t = Categorical(logits=logits_t.squeeze(1)).sample()
        actions.append(actions_t.view(-1, 1)) # bsz x 1
        # get outputs for each env, which are (observation, reward, terminated, truncated, info) tuples
        env_outputs = [env.step(actions_t[b].item()) for b, env in enumerate(envs)]
        rewards_t = torch.tensor([eo[1] for eo in env_outputs])
        # if we lose a life, zero out all subsequent rewards
        still_alive = torch.tensor([env.ale.lives() == args.start_nlives for env in envs])
        not_terminated.mul_(still_alive.float())
        rewards.append(rewards_t*not_terminated)
        observations = torch.stack([utils.preprocess_observation(eo[0]) for eo in env_outputs]).unsqueeze(1)

    # calculate reward-to-go
    r2g = torch.zeros(bsz, args.unroll_length)
    curr_r = 0
    for r in range(args.unroll_length-1, -1, -1):
        curr_r = rewards[r] + args.discounting * curr_r
        r2g[:, r].copy_(curr_r)

    adv = (r2g - r2g.mean()) / (r2g.std() + 1e-7) # biased, but people do it
    logits = torch.cat(logits, dim=1) # bsz x T x naction
    actions = torch.cat(actions, dim=1) # bsz x T 
    cross_entropy = F.cross_entropy(
        logits.view(-1, logits.size(2)), actions.view(-1), reduction='none')
    pg_loss = (cross_entropy.view_as(actions) * adv).mean()
    total_loss = pg_loss

    stats = {"mean_return": sum(r.mean() for r in rewards)/len(rewards),
             "pg_loss": pg_loss.item()}
    optimizer.zero_grad()
    total_loss.backward()
    # nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clipping)
    optimizer.step()
    scheduler.step()

    # reset any environments that have ended
    for b in range(bsz):
        if not_terminated[b].item() == 0:
            obs = envs[b].reset(seed=stepidx+b)[0]
            observations[b].copy_(utils.preprocess_observation(obs))

    return stats, envs, observations, prev_state


def act(model_loc, prev_state, state, epsilon):
    """
    Either take the best action from the observation or return a random one based on epsilon greedy

    Params:
    model_loc - model, supposedly the qnetwork_local
    state - state at a specific time
    epsilon - epsilon used for epsilon-greedy

    returns:
        action
    """
    state = state.float().unsqueeze(0).unsqueeze(0)
    if prev_state is not None:
        prev_state = prev_state.float().unsqueeze(0).unsqueeze(0)

    # Epsilon Greedy
    if random.random() > epsilon:
        action = model_loc.get_action(state, epsilon, prev_state)[0]
        return action
    else:
        action = random.choice(np.arange(model_loc.naction))
        return action


def soft_update(model_loc, model_target, tau):
    
    for target_param, local_param in zip(model_target.parameters(),
                                    model_loc.parameters()):
        target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)
    
    # model_loc_param = 0
    # for param in model_loc.parameters():
    #     if param.grad is not None:
    #         model_loc_param+= torch.sum(torch.abs(param.grad)).item()
    # print(model_loc_param)
    return


def learn(model_loc, model_target, experiences, gamma, optimizer, tau):
    """
    model_loc - Local model for training
    model_target - Target final model that we periodically update based on model_loc params
    experiences - Sampled from replay buffer
    gamma - Discounting rate
    optimizer - Adam optimizer used to optimize params in the DQN
    tau - parameter for transferring learned params from model_loc to model_target
    """
    states, prev_states, actions, rewards, next_states, dones = experiences

    model_loc.train()
    model_target.eval()
    predicted_targets = model_loc(states, prev_states)[0].squeeze(1).max(1)[0].unsqueeze(1)         # 8x1 -- bsz x 1
    # print("Predicted Targets")
    # print(predicted_targets)

    with torch.no_grad():
        labels_next = model_target(next_states, states)[0].squeeze(1).max(1)[0].unsqueeze(1)       # 8x1 -- bsz x 1
        # print(labels_next)

    labels = rewards + (gamma * labels_next * (1 - dones))
    # print("Labels")
    # print(labels)
    loss = F.mse_loss(predicted_targets,labels, reduction='mean')
    # print("Loss")
    # print(loss)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model_loc.parameters(), args.grad_norm_clipping)
    optimizer.step()

    # ------------------- update target network ------------------- #
    soft_update(model_loc, model_target, tau)
    return


def dqn_step(t, model_loc, model_target, prev_state, state, action, reward, next_state, done, memory, bsz, gamma, optimizer, tau, update=False):
    """
    DQN Step

    Params:
    model_loc - model, supposedly the qnetwork_local
    model_target - model, supposedly the actual model

    From env params:
    state - state
    action - action
    reward - reward
    next_state - next_state
    done - done

    memory - replay buffer
    bsz - batch size (requirement for minimum size for training Q Network)
    gamma - discounting factor
    update - whether we update the model or not (learn)
    
    """
    # Save experience in replay memory -- Not prev_state == None
    if t > 0:
        memory.push(state, prev_state, action, reward, next_state, done)

    if update:
        # Enough samples in memory
        if len(memory) > bsz:
            experience = memory.sample()
            learn(model_loc, model_target, experience, gamma, optimizer, tau)
    return


def train(args):
    # T = args.unroll_length
    B = args.batch_size
    T = args.train_time
    nepisodes = args.nepisodes
    epsilon = args.epsilon
    epsilon_min = args.epsilon_min
    epsilon_decay = args.epsilon_decay
    episode_eval_rate = args.eval_every_ep
    model_update_rate = args.update_every
    gamma = args.discounting
    tau = args.tau

    args.device = torch.device("cpu")
    env = gym.make(args.env)
    naction = env.action_space.n
    args.start_nlives = env.ale.lives()
    del env

    model = DQNNetwork(naction, args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    def lr_lambda(epoch): # multiplies learning rate by value returned; can be used to decay lr
        return 1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def checkpoint():
        if args.save_path is None:
            return
        logging.info("Saving checkpoint to {}".format(args.save_path))
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "args": args}, args.save_path)


    timer = timeit.default_timer
    last_checkpoint_time = timer()
    envs, observations, prev_state = None, None, None
    frame = 0

    # DQN Changes
    scores = []
    scores_window = deque(maxlen=100)               # Last 100 scores
    model_loc = DQNNetwork(naction, args)           # Local model
    replay_buffer = ReplayMemory(args)              # Memory    
    t_step = 0                                      # Tracking for updates


    for i_episode in range(1, nepisodes+1):
        env = gym.make(args.env)
        state = env.reset()[0]
        score = 0
        for t in range(T):
            preprocessed_state = utils.preprocess_observation(state, prev_state)
            preprocessed_prevstate = utils.preprocess_observation(prev_state, None)
            action = act(model_loc, preprocessed_prevstate, preprocessed_state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)
            preprocessed_nextstate = utils.preprocess_observation(next_state, state)
            if(t_step+1) % model_update_rate == 0:
                dqn_step(t, model_loc, model, preprocessed_prevstate, preprocessed_state, action, reward, preprocessed_nextstate, done, replay_buffer, B, gamma, optimizer, tau, update=True)
            else: 
                dqn_step(t, model_loc, model, preprocessed_prevstate, preprocessed_state, action, reward, preprocessed_nextstate, done, replay_buffer, B, gamma, optimizer, tau, update=False)
            prev_state = state
            state = next_state
            score += reward
            if done: 
                break
            t_step += 1

        # Save most recent score    
        scores_window.append(score)
        scores.append(score)
        epsilon = max(epsilon_min, epsilon_decay * epsilon)         # Epsilon decay

        # Check if Checkpoint 
        if timer() - last_checkpoint_time > args.min_to_save * 60:
            checkpoint()
            last_checkpoint_time = timer()
        
        print('\rEpisode {}\tScore {:.2f}'.format(i_episode, score, end=""))

        if i_episode % episode_eval_rate == 0:
            print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)))

        if i_episode > 0 and i_episode % episode_eval_rate == 0:
            utils.validate(model, epsilon, render=args.render)

        t_step = 0

    plt.plot(scores)
    plt.show()
    plt.close()

            
    # while frame < args.total_frames:
    #     start_time = timer()
    #     start_frame = frame
    #     stats, envs, observations, prev_state = dqn_step(
    #         frame, model, optimizer, scheduler, envs, observations, prev_state, bsz=B)
    #     frame += T*B # here steps means number of observations
    #     if timer() - last_checkpoint_time > args.min_to_save * 60:
    #         checkpoint()
    #         last_checkpoint_time = timer()

    #     sps = (frame - start_frame) / (timer() - start_time)
    #     logging.info("Frame {:d} @ {:.1f} FPS: pg_loss {:.3f} | mean_ret {:.3f}".format(
    #       frame, sps, stats['pg_loss'], stats["mean_return"]))

    #     if frame > 0 and frame % (args.eval_every*T*B) == 0:
    #         utils.validate(model, render=args.render)
    #         model.train()


parser = argparse.ArgumentParser()

parser.add_argument("--env", type=str, default="ALE/MsPacman-v5", help="gym environment")
parser.add_argument("--mode", default="train", choices=["train", "valid",], 
                    help="training or validation mode")
parser.add_argument("--total_frames", default=1000000, type=int, 
                    help="total environment frames to train for")
parser.add_argument("--batch_size", default=8, type=int, help="learner batch size.")
parser.add_argument("--unroll_length", default=80, type=int, 
                    help="unroll length (time dimension)")
parser.add_argument("--hidden_dim", default=256, type=int, help="policy net hidden dim")
parser.add_argument("--discounting", default=0.99, type=float, help="discounting factor")
parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate")
parser.add_argument("--grad_norm_clipping", default=10.0, type=float,
                    help="Global gradient norm clip.")
parser.add_argument("--save_path", type=str, default=None, help="save model here")
parser.add_argument("--load_path", type=str, default=None, help="load model from here")
parser.add_argument("--min_to_save", default=5, type=int, help="save every this many minutes")
parser.add_argument("--eval_every", default=50, type=int, help="eval every this many updates")
parser.add_argument("--render", action="store_true", help="render game-play at validation time")

# Deep Q Network
parser.add_argument("--memory_len", type=int, default=2000, help="length of the memory for Deep Q Learning")
parser.add_argument("--epsilon", type=float, default=1.0, help="exploration rate")
parser.add_argument("--epsilon_min", type=float, default=0.6, help="minimum epsilon")
parser.add_argument("--epsilon_decay", type=float, default=0.999, help="decay rate for epsilon")
parser.add_argument("--nepisodes", type=int, default=500, help="number of episodes to train from")
parser.add_argument("--train_time", type=int, default=10000, help="training time")
parser.add_argument("--eval_every_ep", type=int, default=10, help="eval every this many episodes")
parser.add_argument("--update_every", type=int, default=50, help="how often to update / learn the network")
parser.add_argument("--tau", type=float, default=5e-3, help="parameter for soft update rate")


if __name__ == "__main__":
    torch.manual_seed(59006)
    np.random.seed(59006)
    args = parser.parse_args()
    logging.info(args)
    if args.mode == "train":
        train(args)
    else:
        assert args.load_path is not None
        checkpoint = torch.load(args.load_path)
        saved_args = checkpoint["args"]
        env = gym.make(args.env)
        naction = env.action_space.n
        saved_args.start_nlives = env.ale.lives()
        del env        
        model = DQNNetwork(naction, saved_args)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model
        args = saved_args

        utils.validate(model, args.epsilon_min, args)
