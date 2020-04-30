import numpy as np
import matplotlib.pyplot as plt


episode_durations = np.loadtxt("results/episode-durations_20200430-0205.out")
episode_rewards = np.loadtxt("results/episode-rewards_20200430-0205.out")
eval_scores = np.loadtxt("results/eval-scores_20200430-0205.out")
print(eval_scores)

plt.clf()
fig, ax1 = plt.subplots()
ax1.set_title('Training...')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Duration', color= 'tab:blue')
# ax1.plot(episode_durations,color='tab:blue')
ax2 = ax1.twinx()
ax2.set_ylabel('Episode Rewards', color= 'tab:green')
ax2.plot(episode_rewards,color='tab:green')
plt.show()

# plt.title('Training...')
# plt.xlabel('Episode')
# plt.ylabel('Duration')
# plt.plot(episode_rewards)
# plt.show()
# print(episode_rewards)
