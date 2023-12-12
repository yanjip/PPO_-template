'''
@Author  ：Yan JP
@Created on Date：2023/6/5 17:20 
'''
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random
# import torch
import pandas as pd
import para
import datetime
import pickle

from matplotlib.font_manager import FontProperties  # 导入字体模块


# 设置中文字体，注意需要根据自己电脑情况更改字体路径，否则还是默认的字体
def chinese_font():
    try:
        font = FontProperties(
            # 系统字体路径
            fname='C:\\Windows\\Fonts\\方正粗黑宋简体.ttf', size=14)
    except:
        font = None
    return font


# 中文画图
def plot_rewards_cn(rewards, cfg, path=None, tag='train'):
    sns.set()
    plt.figure()
    plt.title(u"{}环境下{}算法的学习曲线".format(cfg['env_name'],
                                       cfg['algo_name']), fontproperties=chinese_font())
    plt.xlabel(u'回合数', fontproperties=chinese_font())
    plt.plot(rewards)
    plt.plot(smooth(rewards))
    plt.legend(('奖励', '滑动平均奖励',), loc="best", prop=chinese_font())
    if cfg['save_fig']:
        plt.savefig(f"{path}/{tag}ing_curve_cn.png")
    if cfg['show_fig']:
        plt.show()


# 用于平滑曲线，类似于Tensorboard中的smooth
def smooth(data, weight=0.9):
    '''
    Args:
        data (List):输入数据
        weight (Float): 平滑权重，处于0-1之间，数值越高说明越平滑，一般取0.9

    Returns:
        smoothed (List): 平滑后的数据
    '''
    last = data[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_rewards_tile(rewards, cfg, path=None):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title(f"{tag}ing curve on {cfg['device']} ")
    plt.title("Tile Choose Solved By DQN")
    plt.xlabel('Epsiodes')
    plt.ylabel('Reward')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()
    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if cfg['save_fig']:
        plt.savefig(f"{path}/{a}train_curve.png")
    if cfg['show_fig']:
        plt.show()

def plot_rewards(rewards, cfg, path=None, tag='train'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title(f"{tag}ing curve on {cfg['device']} ")
    plt.title("Beamforming Solved By DDPG")
    plt.xlabel('Epsiodes')
    plt.ylabel('Reward')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()
    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if cfg['save_fig']:
        plt.savefig(f"{path}/{a}_beamforing.png")
    if cfg['show_fig']:
        plt.show()


def plot_losses(losses, algo="DQN", save=True, path='./'):
    sns.set()
    plt.figure()
    plt.title("loss curve of {}".format(algo))
    plt.xlabel('epsiodes')
    plt.plot(losses, label='rewards')
    plt.legend()
    if save:
        plt.savefig(path + "losses_curve")
    plt.show()


# 保存奖励
def save_results(res_dic, tag='train', path=None):
    '''
    '''
    Path(path).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(res_dic)
    df.to_csv(f"{path}/{tag}ing_results.csv", index=None)
    print('结果已保存: ' + f"{path}/{tag}ing_results.csv")


# 创建文件夹
def make_dir(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


# 删除目录下所有空文件夹
def del_empty_dir(*paths):
    for path in paths:
        dirs = os.listdir(path)
        for dir in dirs:
            if not os.listdir(os.path.join(path, dir)):
                os.removedirs(os.path.join(path, dir))


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# 保存参数
def save_args(args, path=None):
    Path(path).mkdir(parents=True, exist_ok=True)
    with open(f"{path}/params.json", 'w') as fp:
        json.dump(args, fp, cls=NpEncoder)
    print("参数已保存: " + f"{path}/params.json")


import time


def write_sinr(sinr, SE, random_SE):
    with open('H_W/sinr_SE.txt', 'a+') as F:
        F.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n')
        F.write("----Power:" + str(para.maxPower) + '\n')
        F.write("SINR:" + str(sinr) + "       SE_all:" + str(SE) + "\n")
        F.write("Random_SE:" + str(random_SE) + "\n\n")


def draw_evaluate(fov_id, proposed, uncompress, greedy, coarsness):
    # 绘制散点图
    plt.scatter(fov_id, proposed, c='red', label='Proposed')
    plt.scatter(fov_id, uncompress, c='blue', label='Uncompress')
    plt.scatter(fov_id, greedy, c='green', label='Greedy')
    plt.scatter(fov_id, coarsness, c='orange', label='coarsness')

    # 添加图例
    plt.legend()

    # 添加坐标轴标签
    plt.ylabel('QoE')
    plt.xlabel('FoV_ID')
    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    plt.savefig("runs/baseline/performance_compare" + a, dpi=600)

    # 显示图形
    plt.show()


def plot_Q_t(ts, proposed, uncompress, greedy, coarsness):
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # 设置字体属性
    saveExcel([ts, proposed, uncompress, greedy, coarsness], 'H_W/TC_Q_Slot.xlsx')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.plot(ts, proposed, marker='o', label='Proposed')  ## 使用圆形节点
    plt.plot(ts, uncompress, marker='s', label='Uncompress')  # 使用方形节点
    plt.plot(ts, greedy, marker='^', label='Greedy')  # 使用三角形节点
    plt.plot(ts, coarsness, marker='*', label='Coarsness')

    plt.legend(loc='best')
    plt.xlabel('T_slot (s)')
    plt.ylabel('Quality of Video')
    plt.title('Playback Quality with Different Time-Slot Length')

    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig("runs/baseline/T_Quality" + a, dpi=600)
    # 显示图形
    plt.show()


def plot_QoE_F(Fs, proposed, uncompress, greedy, coarsness):
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # 设置字体属性
    # saveExcel([Fs, proposed, uncompress, greedy, coarsness], 'H_W/TC_Q_F4.xlsx')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.plot(Fs, proposed, marker='o', label='Proposed')  ## 使用圆形节点
    plt.plot(Fs, uncompress, marker='s', label='Uncompress')  # 使用方形节点
    plt.plot(Fs, greedy, marker='^', label='Greedy')  # 使用三角形节点
    plt.plot(Fs, coarsness, marker='*', label='Coarsness')

    plt.legend(loc='best')
    plt.ylim(6)
    plt.xlabel('Computation Capacity (GHz)')
    plt.ylabel('Quality of Video')
    plt.title('Playback Quality with Different Computation Capabilities')
    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig("runs/baseline/F_QoE-" + a, dpi=600)
    # 显示图形
    plt.show()


def plot_QoE_Sinr(Fs, proposed, uncompress, greedy, coarsness):
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # 设置字体属性
    saveExcel([Fs, proposed, uncompress, greedy, coarsness], 'H_W/TC_Q_SINR.xlsx')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.plot(Fs, proposed, marker='o', label='Proposed')  ## 使用圆形节点
    plt.plot(Fs, uncompress, marker='s', label='Uncompress')  # 使用方形节点
    plt.plot(Fs, greedy, marker='^', label='Greedy')  # 使用三角形节点
    plt.plot(Fs, coarsness, marker='*', label='Coarsness')

    plt.legend(loc='best')
    plt.xlabel('SINR (dB)')
    plt.ylabel('Quality of Video')
    plt.title('Playback Quality with Different SINR')
    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig("runs/baseline/Sinr_QoE-" + a, dpi=600)
    # 显示图形
    plt.show()


def plot_QoE_BW(Fs, proposed, uncompress, greedy, coarsness):
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # 设置字体属性
    saveExcel([Fs, proposed, uncompress, greedy, coarsness], 'H_W/TC_Q_BW.xlsx')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.plot(Fs, proposed, marker='o', label='Proposed')  ## 使用圆形节点
    plt.plot(Fs, uncompress, marker='s', label='Uncompress')  # 使用方形节点
    plt.plot(Fs, greedy, marker='^', label='Greedy')  # 使用三角形节点
    plt.plot(Fs, coarsness, marker='*', label='Coarsness')

    plt.legend(loc='best')
    plt.xlabel('Bandwith (MHz)')
    plt.ylabel('Quality of Video')
    plt.title('Playback Quality with Different Bandwith')
    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig("runs/baseline/Bandwith_QoE-" + a, dpi=600)
    # 显示图形
    plt.show()


def plot_SE(powers, ddpg, dqn, randomSE):
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # 设置字体属性
    saveExcel_beam([powers, ddpg, dqn, randomSE], path='H_W/Power_SE2.xlsx')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.plot(powers, ddpg, marker='o', label='DDPG')  ## 使用圆形节点
    plt.plot(powers, dqn, marker='s', label='DQN')  # 使用方形节点
    plt.plot(powers, randomSE, marker='^', label='RANDOM')  # 使用三角形节点
    # plt.plot(Fs, coarsness, marker='*', label='coarsness')

    plt.legend(loc='best')
    plt.xlabel('The Maximum Transmitting Power of the AP (dBm)')
    plt.ylabel('Spectrum Effectiveness (bps/Hz)')
    plt.title('Spectrum Effectiveness with Different Transmiting Power')
    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig("runs/baseline/Power-SE" + a, dpi=600)
    # 显示图形
    plt.show()


def plot_SINR_bar(user1, user2, user3):
    saveExcel_beam([user1, user2, user3, 0], path='H_W/SE_Bar_2.xlsx')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    # 设置柱状图的宽度比例
    bar_width = 0.5
    categories = ['ddpg', 'dqn', 'random']
    # total_values = np.add(user1,user2,user3)
    # 绘制堆积柱状图    默认的柱状图宽度是 0.8
    plt.bar(categories, user1, width=bar_width, label='User1')
    plt.bar(categories, user2, width=bar_width, bottom=user1, label='User2')
    plt.bar(categories, user3, width=bar_width, bottom=np.add(user1, user2), label='User3')

    plt.xlabel('Different Algorithms')
    plt.ylabel('Spectrum Effectiveness Of Each Users(bps/Hz)')
    # plt.title('Stacked Bar Chart')
    plt.legend()
    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig("runs/baseline/SINR-Bar-" + a, dpi=600)
    plt.show()


def plot_QoE_bar(proposed, uncompress, greedy, coarsness, dqn_pro):
    data = [proposed, uncompress, greedy, coarsness, dqn_pro]
    # saveNpz(data, path='H_W/QoE_bar_Users.npz')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'

    # 设置柱状图的宽度比例
    bar_width = 0.5
    # categories = ['Proposed', 'Uncompress', 'Greedy', 'Coarsness', 'BeamDQN']
    # categories = ['JB-BRC', 'JB-Uncompress', 'JB-Greedy', 'JB-Coarsness', 'BQ-BRC']
    # user1 = [proposed[0], uncompress[0], greedy[0], coarsness[0], dqn_pro[0]]
    # user2 = [proposed[1], uncompress[1], greedy[1], coarsness[1], dqn_pro[1]]
    # user3 = [proposed[2], uncompress[2], greedy[2], coarsness[2], dqn_pro[2]]
    categories = ['Proposed', 'Baseline 1', 'Baseline 2', 'Baseline 3']
    user1 = [proposed[0], uncompress[0], greedy[0], coarsness[0]]
    user2 = [proposed[1], uncompress[1], greedy[1], coarsness[1]]
    user3 = [proposed[2], uncompress[2], greedy[2], coarsness[2]]
    bar_width = 0.2
    x = np.arange(len(categories))

    # 绘制堆积柱状图    默认的柱状图宽度是 0.8
    plt.bar(x, user1, width=bar_width, label='User1')
    plt.bar(x + bar_width, user2, width=bar_width, label='User2')
    plt.bar(x + 2 * bar_width, user3, width=bar_width, label='User3')

    plt.xticks(x + bar_width, categories)
    # plt.xlabel('Different Algorithms')
    plt.ylabel('The Average QoE In One Time Slot')
    # plt.title('Stacked Bar Chart')
    plt.legend(loc='upper center', ncol=3)
    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig("runs/baseline/QoE-Bar-" + a, dpi=600,bbox_inches='tight', pad_inches=0.1)
    plt.show()


# 和上面那个函数相比，这个是堆叠的
def plot_QoE_bar2(proposed, uncompress, greedy, coarsness, dqn_pro):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'

    # 设置柱状图的宽度比例
    bar_width = 0.5
    # categories = ['Proposed', 'Uncompress', 'Greedy', 'Coarsness', 'BeamDQN']
    categories = ['JB-BRC', 'JB-Uncompress', 'JB-Greedy', 'JB-Coarsness', 'BQ-BRC']
    user1 = [proposed[0], uncompress[0], greedy[0], coarsness[0], dqn_pro[0]]
    user2 = [proposed[1], uncompress[1], greedy[1], coarsness[1], dqn_pro[1]]
    user3 = [proposed[2], uncompress[2], greedy[2], coarsness[2], dqn_pro[2]]
    # 绘制堆积柱状图    默认的柱状图宽度是 0.8
    plt.bar(categories, user1, width=bar_width, label='User1')
    plt.bar(categories, user2, width=bar_width, bottom=user1, label='User2')
    plt.bar(categories, user3, width=bar_width, bottom=np.add(user1, user2), label='User3')

    plt.xlabel('Different Algorithms')
    plt.ylabel('QoE of Average Time slot')
    # plt.title('Stacked Bar Chart')
    plt.legend(ncol=3)
    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig("runs/baseline/QoE-Bar-Users-" + a, dpi=600,bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_QoE_T(QoEs):
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # 设置字体属性
    saveExcel([QoEs, 0, 0, 0, 0], 'H_W/QoE_pro_stable_single.xlsx')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    x = [i + 1 for i in range(len(QoEs))]
    plt.plot(x, QoEs, marker='>', label='Proposed')  # 使用三角形节点
    # plt.plot(Fs, coarsness, marker='*', label='coarsness')
    # plt.xticks(x, map(int, x))
    plt.ylim(0)
    plt.legend(loc='best')
    plt.xlabel('Time (s)')
    plt.ylabel('QoE')
    plt.title('single QoE with Growing Time')
    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig("runs/baseline/QoE-Time" + a, dpi=600,bbox_inches='tight', pad_inches=0.1)
    # 显示图形
    plt.show()


def plot_QoE_T_all(QoEs, uncompress, greedy, coarsness, dqn):
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # categories = ['JB-BRC', 'JB-Uncompress', 'JB-Greedy', 'JB-Coarsness', 'BQ-BRC']
    # 设置字体属性
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    x = [i + 1 for i in range(len(QoEs))]
    # plt.plot(x, QoEs, marker='.', markersize=9, label='JB-BRC')  # 使用三角形节点
    # plt.plot(x, uncompress, marker='s', markersize=4, label='JB-Uncompress')  # 使用方形节点
    # plt.plot(x, greedy, marker='^', markersize=4, label='JB-Greedy')  # 使用三角形节点
    # plt.plot(x, coarsness, marker='*', markersize=7, label='JB-Coarsness')
    # plt.plot(x, dqn, marker='d', markersize=4, label='BQ-BRC')
    # plt.xticks(x, map(int, x))

    plt.plot(x, QoEs, marker='.', markersize=9, label='Proposed')  # 使用三角形节点
    plt.plot(x, uncompress, marker='s', markersize=4, label='Baseline 1')  # 使用方形节点
    plt.plot(x, greedy, marker='^', markersize=4, label='Baseline 2')  # 使用三角形节点
    plt.plot(x, coarsness, marker='*', markersize=7, label='Baseline 3')
    plt.ylim(0.3)
    plt.legend(loc='lower center', ncol=4)
    plt.xlabel('Time Slot')
    plt.ylabel('$\mathrm{QoE}(t)$')
    # plt.title('Users\'s QoE at Every Slot')
    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig("runs/baseline/QoE-Time-Stable" + a, dpi=600,bbox_inches='tight', pad_inches=0.1)
    # 显示图形
    plt.show()


def plot_QoE_sum_increasing(QoEs, uncompress, greedy, coarsness, dqn):
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # categories = ['JB-BRC', 'JB-Uncompress', 'JB-Greedy', 'JB-Coarsness', 'BQ-BRC']
    # 设置字体属性
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    x = [i + 1 for i in range(len(QoEs))]
    # plt.plot(x, QoEs, marker='>', markersize=4, label='JB-BRC')  # 使用三角形节点
    # plt.plot(x, uncompress, marker='s', markersize=4, label='JB-Uncompress')  # 使用方形节点
    # plt.plot(x, greedy, marker='^', markersize=4, label='JB-Greedy')  # 使用三角形节点
    # plt.plot(x, coarsness, marker='*', markersize=4, label='JB-Coarsness')
    plt.plot(x, QoEs, marker='>', markersize=4, label='Proposed')  # 使用三角形节点
    plt.plot(x, uncompress, marker='s', markersize=4, label='Baseline 1')  # 使用方形节点
    plt.plot(x, greedy, marker='^', markersize=4, label='Baseline 2')  # 使用三角形节点
    plt.plot(x, coarsness, marker='*', markersize=4, label='Baseline 3')
    # plt.plot(x, dqn, marker='d', markersize=4, label='BQ-BRC')
    # plt.xticks(x, map(int, x))
    plt.ylim(0)
    plt.legend(loc='upper left', ncol=1)
    plt.xlabel('Time Slot')
    plt.ylabel('The Cumulative QoE')
    # plt.title('Total QoE at Time Slot')
    a = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig("runs/baseline/11_22_QoE-TimeSum_increasing" + a, dpi=600,bbox_inches='tight', pad_inches=0.1)
    # 显示图形
    plt.show()


def saveNpz(data: list, path):
    file_path = path  # file_path = 'H_W/res.npz'
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def saveExcel(data: list, path):
    data = {
        'a': data[0],
        'b': data[1],
        'c': data[2],
        'd': data[3],
        'e': data[4],
    }
    # 创建 DataFrame
    df = pd.DataFrame(data)
    # 保存为 Excel 文件
    file_path = path
    df.to_excel(file_path, index=False)


def saveExcel_beam(data: list, path):
    data = {
        'x': data[0],
        'ddpg': data[1],
        'dqn': data[2],
        'random': data[3],
    }
    # 创建 DataFrame
    df = pd.DataFrame(data)
    # 保存为 Excel 文件
    file_path = path
    df.to_excel(file_path, index=False)


def get_TC_data(file_path):
    df = pd.read_excel(file_path)
    res = df.to_dict(orient='list')
    return res


if __name__ == '__main__':
    # sinr = [11.379501, 4.522286, 41.942398]
    # se = 11.519477844238281
    # write_sinr(sinr, se)
    # draw_evaluate(1,2,3)
    # x = [5.84, 5.9, 6.32]
    # y = [4.84, 4.87, 5.62]
    # z = [4.68, 4.68, 5.85]
    # w = [5.33, 5.33, 6.22]
    # m = [4.2, 6.74, 4.2]
    # plot_QoE_bar(x, y, z, w, m)
    file = 'H_W/TC_Q_F4.xlsx'
    data = get_TC_data(file)
    plot_QoE_F(data['a'], data['b'], data['c'], data['d'], data['e'])
    pass
