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
import pickle
import signal

from environment import init_env, start_new_game, get_reward_and_next_state,N_ACTIONS,RESOLUTION

np.random.seed(111)
torch.manual_seed(111)

BATCH_SIZE = 512
TRAIN_ITERATIONS = 100
LEARNING_RATE = 1e-4
GAMMA = 0.999
EPS_START = 0.25
EPS_END = 0.05
EPS_DECAY = 100
TARGET_UPDATE = 1
SAVE_MODEL = 10
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


num_episodes = 500
episode_durations = []
res_path = "results/"
model_checkpoint_path = "model-egreedy-chk.pt"
replay_buffer_path = "replay-egreedy.pkl"

steps_done = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(RESOLUTION, RESOLUTION, N_ACTIONS).to(device)
target_net = DQN(RESOLUTION, RESOLUTION, N_ACTIONS).to(device)

memory = ReplayMemory(50000)

# If loading a saved model
# print("Loading Model and Replay buffer")
# if os.path.exists(model_checkpoint_path):
#     policy_net.load_state_dict(torch.load(model_checkpoint_path))
# with open(replay_buffer_path, 'rb') as input:
#     memory.set_memory(pickle.load(input))

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


def keyboardInterruptHandler(signal, frame):
    print("KeyboardInterrupt has been caught. Saving plot...")
    plot_durations()
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)


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
        return 0
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([torch.Tensor(s) for s in batch.next_state
                                                if s is not None]).to(device)
    state_batch = torch.cat([torch.Tensor(s) for s in batch.state]).to(device)
    action_batch = torch.cat([torch.LongTensor([[s]]) for s in batch.action]).to(device)
    reward_batch = torch.cat([torch.Tensor([s]) for s in batch.reward]).to(device)

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
    return loss.item()


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
    if sample > eps_threshold:
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action = policy_net(state.to(device)).max(1)[1].view(1, 1).item()
            return action, (sample, eps_threshold)
    else:
        action = random.randrange(N_ACTIONS)
        return action, None


def main():
    """Main Training Loop"""
    # init_env()

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        time.sleep(1)
        state = start_new_game()
        time.sleep(0.2)

        last_time = time.time()
        print("Game Start Time: {}".format(last_time))
        for t in count():
            last_time = time.time()
            action, network_action = select_action(state)

            reward, next_state = get_reward_and_next_state(action)

            if network_action is not None:
                print("t:{}\t time:{:.2f}\t reward:{}\t action:{}\t NETWORK {:.2f}/{:.2f}".format(t, time.time() - last_time, reward, action, network_action[0], network_action[1]))
            else:
                print("t:{}\t time:{:.2f}\t reward:{}\t action:{}\t".format(t, time.time() - last_time, reward, action))
            # if reward > 1:
            #     print("REWARD -- {}".format(reward))
            # Store the transition in memory
            # only if frame number > 60, don't consider empty frames
            if t > 60:
                memory.push(state,
                    action,
                    next_state,
                    reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            # optimize_model()
            if next_state is None:
                episode_durations.append(t + 1)
                break



        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if i_episode % SAVE_MODEL == 0:
            torch.save(policy_net.state_dict(), model_checkpoint_path)
            print("Saving replays at episode {}".format(i_episode))
            with open(replay_buffer_path, 'wb') as output:
                pickle.dump(memory.get_memory(), output)
            eval()

        print("Episode {}/{} -- Duration: {}".format(i_episode, num_episodes, t+1))
        print("Memory Size: {}".format(len(memory.get_memory())))

        print("Optimizing model")
        global steps_done
        steps_done += 1
        optim_time = time.time()

        loss = 0
        for _ in range(TRAIN_ITERATIONS):
            loss += optimize_model()
        print("BATCH_SIZE:{} TRAIN_ITERATIONS:{} Optimization time:{:.2f} Avg Loss:{:.2f}".format(
            BATCH_SIZE, TRAIN_ITERATIONS, time.time() - optim_time, loss/TRAIN_ITERATIONS))


def eval():
    time.sleep(1)
    state = start_new_game()
    with torch.no_grad():
        last_time = time.time()
        for t in count():
            last_time = time.time()
            action, network_action = select_action(state, eps=0) # Choose best action
            reward, next_state = get_reward_and_next_state(action)
            print("t:{}\t time:{:.2f}\t reward:{}\t action:{}\t NETWORK {:.2f}/{:.2f}".format(t, time.time() - last_time, reward, action, network_action[0], network_action[1]))

            # Move to the next state
            state = next_state

            if next_state is None:
                break
        print("Eval Duration: {}".format(t))

if __name__ == "__main__":
    main()
    eval()
    plot_durations()
