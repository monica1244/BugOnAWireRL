import random
import os
import time
import numpy as np
import torch, torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple
import torch.nn.functional as F

from dqn import DQN
from replay_memory import ReplayMemory
import cv2

from environment import init_env, start_new_game, get_reward_and_next_state,N_ACTIONS,RESOLUTION

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


num_episodes = 500
episode_durations = []
res_path = "results/"
model_checkpoint_path = "model-chk.pt"

steps_done = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(RESOLUTION, RESOLUTION, N_ACTIONS).to(device)
target_net = DQN(RESOLUTION, RESOLUTION, N_ACTIONS).to(device)

# If loading a saved model
# policy_net.load_state_dict(torch.load(model_checkpoint_path))
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


def plot_durations():
    """Function to plot and store the durations of each episode

    Args: i_episode: episode number used in the image name

    """

    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if not os.path.exists(res_path):
        print("doesnt existtt")
        os.makedirs(res_path)
    try:
        plt.savefig(res_path + "duration_plot_{}.png".format(time.strftime("%Y%m%d-%H%M%S")))
    except:
        print("Unable to save plot")
    # Take 100 episode averages and plot them too
    # if len(durations_t) >= 100:
    #     means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())
    #     plt.savefig(res_path+"duration_plot_{}.png".format(i_episode))

    plt.pause(0.001)  # pause a bit so that plots are updated
    time.sleep(1)

def optimize_model():
    """Function for gradient updates

    In this function we sample from memory(ReplayMemory), use the policy_net to
    get the state_action_values and the target net and the next_states to compute
    the expected_state_action_values and use huber loss to update the weights.

    """
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def select_action(state, eps=None):
    """Function for select an action givena state

    In this function we select a random sample variable and compare it with epsilon
    threshold to decide whether to explore or exploit.

    """
    global steps_done
    sample = random.random()
    if eps is not None:
        eps_threshold = eps
    else:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(N_ACTIONS)]], device=device, dtype=torch.long)
        # return torch.tensor([[3]], device=device, dtype=torch.long)


def main():
    """Main Training Loop"""
    # init_env()

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        time.sleep(1)
        state = start_new_game()

        last_time = time.time()
        old_frame, next_frame = None, None
        print("Game Start Time: {}".format(last_time))
        for t in count():
            print("t:{} time:{}".format(t, time.time() - last_time))
            last_time = time.time()
            action = select_action(state, 0) # Select and perform an action
            ##########################################
            #TODO Intract with the environment to get the reward and next_state
            old_frame = next_frame
            reward, next_state, next_frame = get_reward_and_next_state(action)

            ##########################################
            reward = torch.tensor([reward], device=device)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()
            if next_state is None:
                episode_durations.append(t + 1)
                break
            # else:
            #     if t > 20 and action == 3:
            #         print("Action: {}".format(action))
            #         cv2.imshow('old_frame', old_frame)
            #         cv2.imshow('next_frame', next_frame)
            #         cv2.waitKey(0)
            #
            #         break



        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            # torch.save(policy_net.state_dict(), model_checkpoint_path)


        print("Episode {}/{} -- Duration: {}".format(i_episode, num_episodes, t+1))


def eval():
    time.sleep(1)
    state = start_new_game()
    with torch.no_grad():
        for t in count():
            action = select_action(state, eps=0) # Choose best action
            ##########################################
            #TODO Intract with the environment to get the reward and next_state
            reward, next_state = get_reward_and_next_state(action)
            # print("frame {}".format(t))

            ##########################################
            reward = torch.tensor([reward], device=device)

            # Move to the next state
            state = next_state

            if next_state is None:
                break
        print("Eval Duration: {}".format(t))

if __name__ == "__main__":
    main()
    eval()
    plot_durations()
