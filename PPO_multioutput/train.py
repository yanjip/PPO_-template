# time: 2023/10/30 14:57
# author: YanJP

import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_discrete import PPO_discrete
import envs
from tqdm import tqdm
from Draw_pic import *
import time
seed = para.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def evaluate_policy(args, env, agent, state_norm):
    times = 1
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:  # During the evaluating,update=False
            s = state_norm(s, update=False)
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            s_, r, done, _ = env.step(a)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += sum(env.res_p)
    print("reward:",episode_reward)
    # return evaluate_reward / times
    return np.array(env.res_p),env.carrier,env.res_birate,env.VQcheck

def write_power(powers,bitrates,carrier_allocation,VQcheck,powersum):
    with open('runs/sum_power.txt', 'a+') as F:
        F.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n')
        # F.write("----Power:" + str(para.maxPower) + '\n')
        F.write("PowerSum:" + str(powersum) + "       power_list:" + str(powers) + "\n")
        F.write("carrier_allocation:"+str(carrier_allocation)+"\n")
        F.write("VQ_check:" + str(VQcheck) + "\n\n")
        F.write("Bitrate_list:" + str(bitrates) + "\n\n")
def get_file_model():
    folder_path = "runs/model/"  # 需要修改成你想要操作的文件夹路径
    file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    sorted_files = sorted(file_list, key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)
    latest_file = sorted_files[0]
    return latest_file
def main(args, time, seed):
    # env = gym.make(env_name)
    # env_evaluate = gym.make(env_name)  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    # env.seed(seed)
    # env.action_space.seed(seed)
    # env_evaluate.seed(seed)
    # env_evaluate.seed(seed)
    # env_evaluate.action_space.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    env = envs.env_()
    env_evaluate = envs.env_()

    args.state_dim = env.observation_space[0]
    # args.action_dim = env.action_space.n
    args.action_dim = env.action_dim
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    # print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_episode_steps={}".format(args.max_episode_steps))

    rewards=[]
    ma_rewards = []  # 记录所有回合的滑动平均奖励

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    # total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    agent = PPO_discrete(args)

    # Build a tensorboard
    writer = SummaryWriter(log_dir='runs/PPO_discrete/time_{}_seed_{}'.format(time, seed))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    for total_steps in tqdm(range(1,args.max_train_steps+1)):
        # para.h=para.generate_h()
    # while total_steps < args.max_train_steps:
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        episode_rewards=[]
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            s_, r, done, _ = env.step(a)
            if len(s_)==9:
                pass
            episode_rewards.append(r)
            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True  #有next state
            else:
                dw = False

            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            # total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0
        ep_reward=sum(episode_rewards)
        print("ep_reward:",ep_reward)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
            # Evaluate the policy every 'evaluate_freq' steps
        if total_steps % args.evaluate_freq == 0:
            evaluate_num += 1
            evaluate_power,carrier_allocation,bitrates,VQcheck = evaluate_policy(args, env_evaluate, agent, state_norm)
            # evaluate_rewards.append(evaluate_reward)
            # np.set_printoptions(precision=3)
            # powersum= sum([a * b for a, b in zip(evaluate_power, carrier_allocation)])
            powersum= sum(evaluate_power)

            print("num:{} \t Power_sum:{:.3f} \t _power_list:{} \tcarrier_allocation:{}\tbirate_seleciton:{}".format(evaluate_num,powersum,evaluate_power,carrier_allocation,bitrates))
            # writer.add_scalar('step_rewards', evaluate_rewards[-1], global_step=total_steps)
            # Save the rewards
            # if evaluate_num % args.save_freq == 0:
            #     np.save('./data_train/PPO_discrete_time_{}_seed_{}.npy'.format(time, seed), np.array(evaluate_rewards))
        # if (total_steps + 1) % 2 == 0:
        #     print(f'train_step:{total_steps},power_all:{env.res_p}',)
    write_power(evaluate_power,bitrates,carrier_allocation,VQcheck,powersum)
    path='runs/model/ppo_'+time+'.pth'
    torch.save(agent.actor.state_dict(), path)
    return {'episodes': range(len(rewards)), 'rewards': rewards, 'ma_rewards': ma_rewards}

def test():
    env = envs.env_()
    args.state_dim = env.observation_space[0]
    # args.action_dim = env.action_space.n
    args.action_dim = env.action_dim
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    agent = PPO_discrete(args)
    model=get_file_model()
    path='runs/model/'+model
    agent.load_model(path)
    rewards = []
    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization

    for total_steps in range(1,args.max_test_steps+1):
        # s = env.reset()
        # episode_steps = 0
        # done = False
        # episode_rewards=[]
        # while not done:
        #     episode_steps += 1
        #     a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
        #     s_, r, done, _ = env.step(a)
        #     episode_rewards.append(r)
        #     if done and episode_steps != args.max_episode_steps:
        #         dw = True  #有next state
        #     else:
        #         dw = False
        #     s = s_
        evaluate_power, carrier_allocation, bitrates, VQcheck = evaluate_policy(args, env, agent, state_norm)
        print("num:{} \n Power_sum:{:.3f} \n power_list:{} \ncarrier_allocation:{}\nbirate_seleciton:{}\nVQ_check:{}".format(
            evaluate_num, sum(evaluate_power), evaluate_power, carrier_allocation, bitrates,VQcheck))

        # ep_reward=sum(episode_rewards)
        # print("ep_reward:",ep_reward)
        # rewards.append(ep_reward)



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(1.0e3), help=" Maximum number of training steps")
    parser.add_argument("--max_test_steps", type=int, default=int(1), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=10, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()
    curr_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    # train=True
    train=False

    if train:
        res_dic=main(args, curr_time, seed=seed)
        plot_rewards(res_dic['rewards'],curr_time,path='runs/pic')
    else:
        test()
    # test()

