"""
How to train the Policy Network :
    python train.py
        --lr 1e-4
        --cv_dir checkpoint directory
        --batch_size 512 (more is better)
        --data_dir directory to contain csv file
        --alpha 0.6
"""
import os
import torch
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn
from collections import deque
import pickle
import pylab
cudnn.benchmark = True
import argparse
from torch.autograd import Variable
# from tensorboard_logger import configure, log_value
from torch.distributions import Bernoulli

from utils import utils, utils_detector
from constants import base_dir_metric_cd, base_dir_metric_fd
from constants import num_actions

import warnings
warnings.simplefilter("ignore")


# GPU Device
gpu_id = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
use_cuda = torch.cuda.is_available()
print("GPU device " , use_cuda)


parser = argparse.ArgumentParser(description='PolicyNetworkTraining')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--data_dir', default='data/', help='data directory')
parser.add_argument('--load', default=None, help='checkpoint to load agent from')
parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--img_size', type=int, default=256, help='PN Image Size')
parser.add_argument('--epoch_step', type=int, default=10000, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=10000, help='total epochs to run')
parser.add_argument('--num_workers', type=int, default=8, help='Number of Workers')
parser.add_argument('--test_epoch', type=int, default=5, help='At every N epoch test the network')
parser.add_argument('--parallel', action='store_true', default=False, help='use multiple GPUs for training')
parser.add_argument('--alpha', type=float, default=0.8, help='probability bounding factor')
parser.add_argument('--beta', type=float, default=0.1, help='Coarse detector increment')
parser.add_argument('--sigma', type=float, default=0.5, help='cost for patch use')
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.makedirs(args.cv_dir)
utils.save_args(__file__, args)

def train(epoch):
    agent.train()
    rewards, rewards_baseline, policies = [], [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
        if not args.parallel:
            inputs = inputs.cuda()

        # Actions by the Agent
        probs = F.sigmoid(agent.forward(inputs))
        alpha_hp = np.clip(args.alpha + epoch * 0.001, 0.6, 0.95)
        probs = probs*alpha_hp + (1-alpha_hp) * (1-probs)

        # Sample the policies from the Bernoulli distribution characterized by agent
        distr = Bernoulli(probs)
        policy_sample = distr.sample()

        # Test time policy - used as baseline policy in the training step
        policy_map = probs.data.clone()
        policy_map[policy_map<0.5] = 0.0
        policy_map[policy_map>=0.5] = 1.0
        policy_map = Variable(policy_map)

        # Get the batch wise metrics
        offset_fd, offset_cd = utils.read_offsets(targets, num_actions)

        # Find the reward for baseline and sampled policy
        reward_map = utils.compute_reward(offset_fd, offset_cd, policy_map.data, args.beta, args.sigma)
        reward_sample = utils.compute_reward(offset_fd, offset_cd, policy_sample.data, args.beta, args.sigma)
        advantage = reward_sample.cuda().float() - reward_map.cuda().float()

        # Find the loss for only the policy network
        loss = distr.log_prob(policy_sample)
        loss = loss * Variable(advantage).expand_as(policy_sample)
        loss = loss.mean()

        loss = loss + F.smooth_l1_loss(sum(critic(inputs)), sum(reward_map))

        optimizer_agent.zero_grad()
        optimizer_critic.zero_grad()
        loss.backward()
        optimizer_agent.step()
        optimizer_critic.step()

        rewards.append(reward_sample.cpu())
        rewards_baseline.append(reward_map.cpu())
        policies.append(policy_sample.data.cpu())

    reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards)
    train_rewards.append(reward)
    avg_reward.append(reward)
    train_loss.append(loss)

    print('Train: %d | Rw: %.6f | S: %.3f | V: %.3f | #: %d' % (epoch, reward, sparsity, variance, len(policy_set)))

    # log_value('train_reward', reward, epoch)
    # log_value('train_sparsity', sparsity, epoch)
    # log_value('train_variance', variance, epoch)
    # log_value('train_baseline_reward', torch.cat(rewards_baseline, 0).mean(), epoch)
    # log_value('train_unique_policies', len(policy_set), epoch)

def test(epoch):
    agent.eval()
    rewards, metrics, policies, set_labels = [], [], [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        inputs = Variable(inputs, volatile=True)
        if not args.parallel:
            inputs = inputs.cuda()

        # Actions by the Policy Network
        probs = F.sigmoid(agent(inputs))

        # Sample the policy from the agents output
        policy = probs.data.clone()
        policy[policy<0.5] = 0.0
        policy[policy>=0.5] = 1.0
        policy = Variable(policy)

        offset_fd, offset_cd = utils.read_offsets(targets, num_actions)

        reward = utils.compute_reward(offset_fd, offset_cd, policy.data, args.beta, args.sigma)
        metrics, set_labels = utils.get_detected_boxes(policy, targets, metrics, set_labels)

        rewards.append(reward)
        policies.append(policy.data)

    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*metrics))]
    precision, recall, AP, f1, ap_class = utils_detector.ap_per_class(true_positives, pred_scores, pred_labels, set_labels)
    reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards)
    ap.append(AP.mean())
    ar.append(recall.mean())
    test_epochs.append(epoch)
    test_actions.append(policy_set)
    print('Test - AP: {} | AR : {}'.format(AP.mean(), recall.mean()))
    print('Test - Rw: %.4f | S: %.3f | V: %.3f | #: %d' % (reward, sparsity, variance, len(policy_set)))

    # log_value('test_reward', reward, epoch)
    # log_value('test_AP', AP[0], epoch)
    # log_value('test_AR', recall.mean(), epoch)
    # log_value('test_sparsity', sparsity, epoch)
    # log_value('test_variance', variance, epoch)
    # log_value('test_unique_policies', len(policy_set), epoch)

    # save the model --- agent
    agent_state_dict = agent.module.state_dict() if args.parallel else agent.state_dict()
    state = {
      'agent': agent_state_dict,
      'epoch': epoch,
      'reward': reward,
    }
    if epoch % 100 == 0:
        torch.save(state, args.cv_dir + '/ckpt_E_%d_R_%.2E' % (epoch, reward))

#--------------------------------------------------------------------------------------------------------#
trainset, testset = utils.get_dataset(args.img_size, args.data_dir)
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

agent = utils.get_model(num_actions)
critic = utils.critic_model(1)
critic = utils.critic_model(1)

# ---- Load the pre-trained model ----------------------
start_epoch = 0
if args.load is not None:
    checkpoint = torch.load(args.load)
    agent.load_state_dict(checkpoint['agent'])
    start_epoch = checkpoint['epoch'] + 1
    print('loaded agent from %s' % args.load)

# Parallelize the models if multiple GPUs available - Important for Large Batch Size to Reduce Variance
if args.parallel:
    agent = nn.DataParallel(agent)
agent.cuda()
critic.cuda()

# Update the parameters of the policy network
optimizer_agent = optim.Adam(agent.parameters(), lr=args.lr)

# Update the parameters of the policy network
optimizer_critic = optim.Adam(critic.parameters(), lr=args.lr)

# Save the args to the checkpoint directory
# configure(args.cv_dir+'/log', flush_secs=5)

# Start training and testing
epochs = []
test_epochs = []
train_rewards = []
avg_rewards = []
ap = []
ar = []
test_actions = []
avg_reward = deque(maxlen=5)
train_loss = []

for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    train(epoch)
    epochs.append(epoch)
    avg_rewards.append(np.mean(avg_reward))

    with open('./save_graph/800_A2C/epochs.txt', 'wb') as f:
        pickle.dump(epochs, f)
    with open('./save_graph/800_A2C/rewards.txt', 'wb') as f:
        pickle.dump(train_rewards, f)
    with open('./save_graph/800_A2C/avg_rewards.txt', 'wb') as f:
        pickle.dump(avg_rewards, f)
    with open('./save_graph/800_A2C/ap.txt', 'wb') as f:
        pickle.dump(ap, f)
    with open('./save_graph/800_A2C/ar.txt', 'wb') as f:
        pickle.dump(ar, f)
    with open('./save_graph/800_A2C/actions.txt', 'wb') as f:
        pickle.dump(test_actions, f)
    with open('./save_graph/800/losses.txt', 'wb') as f:
        pickle.dump(train_loss, f)

    pylab.cla()

    pylab.plot(epochs, avg_rewards, 'b')
    pylab.xlabel("epoch")
    pylab.ylabel("avg_reward")
    pylab.savefig("./save_graph/800_A2C/graph_avg_r.png")

    pylab.cla()

    pylab.plot(test_epochs, ap, 'b')
    pylab.xlabel("epoch")
    pylab.ylabel("precision")
    pylab.savefig("./save_graph/800_A2C/graph_precision.png")

    pylab.cla()

    pylab.plot(test_epochs, ar, 'b')
    pylab.xlabel("epoch")
    pylab.ylabel("recall")
    pylab.savefig("./save_graph/800_A2C/graph_recall.png")

    pylab.cla()

    pylab.plot(epochs, avg_rewards, 'b', label='average reward')
    pylab.plot(test_epochs, ap, 'g', label='average precision')
    pylab.plot(test_epochs, ar, 'r', label='average recall')
#     pylab.plot(epochs, train_loss, 'b', label='train loss')

    pylab.xlabel("epoch")
    pylab.legend(loc='best')

    pylab.savefig("./save_graph/800_A2C/graph_summary.png")

    if (epoch) % args.test_epoch == 0:
        test(epoch)

