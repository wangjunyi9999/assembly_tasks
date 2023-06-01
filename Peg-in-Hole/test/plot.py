import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
'''
e.g.
    python plot.py --algo=ppo --info=right_force --task=insert
'''
DOF_NUMS = 7
PICK_END_NUMS = 98
LONG_END_NUMS = 198
SUB_PLOT_COLUMNS = 7
SUB_PLOT_LINES = 1

parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, default='ppo')  # ppo, sac
parser.add_argument("--task", type=str, default='pick')  # pick; place; insert
# dof_state; left_force; right_force
parser.add_argument("--info", type=str, default='mid_pos')
args = parser.parse_args()

root_path = f'./{args.algo}_extract_info/'
file_name = f'{args.info}_{args.task}.txt'
file_path = os.path.join(root_path, file_name)
data = np.loadtxt(file_path)
end = PICK_END_NUMS  # avoid extrem values
if 'pick' not in args.task:
    end = LONG_END_NUMS


def main():
    if 'mid' in args.info:
        ax = plt.figure().add_subplot(projection='3d')
        pos_x = data[:end, 0]
        pos_y = data[:end, 1]
        pos_z = data[:end, 2]
        ax.scatter(pos_x, pos_y, pos_z)

    elif 'dof' in args.info:
        idx = np.arange(0, end)
        for i in range(DOF_NUMS):
            q = data[:end, i]
            plt.subplot(SUB_PLOT_COLUMNS, SUB_PLOT_LINES, i+1)
            plt.plot(idx, q)
            plt.title(f'dof {i}')

    elif 'force' in args.info:
        idx = np.arange(0, end)
        data_r = np.loadtxt(f'{args.algo}_extract_info/right_force_insert.txt')
        data_l = np.loadtxt(f'{args.algo}_extract_info/left_force_insert.txt')
        label_force = ['X', 'Y', 'Z']
        for i in range(len(label_force)):
            if i < 2:
                # q=data[:end,i]
                q = data_l[:end, i]+data_r[:end, i]
            else:
                # q=data[:end,i]*100
                q = (data_l[:end, i]+data_r[:end, i])*1000
            plt.subplot(len(label_force), 1, i+1)
            plt.plot(idx, q)
            plt.title(f'force_{label_force[i]}')

    plt.show()


if __name__ == "__main__":
    main()
