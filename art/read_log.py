import os
from pathlib import Path
import matplotlib.pyplot as plt

RES_DIR = Path(__file__).resolve().parent.parent / 'results' / 'mnist' / 'repair' / 'debug'



if __name__ == '__main__':
    lr = [0.01, 0.005]
    weight_decay = [0.0001, 0.00005]
    kcoeff = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    support_loss = ['SmoothL1', 'L1', 'L2']
    accuracy_loss = ['CE', 'MSE']
    repair_radius = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
    record_dict = {}
    for l in lr:
        record_dict[l] = {}
        for w in weight_decay:
            record_dict[l][w] = {}
            for k in kcoeff:
                record_dict[l][w][k] = {}
                for s in support_loss:
                    record_dict[l][w][k][s] = {}
                    for a in accuracy_loss:
                        record_dict[l][w][k][s][a] = {}
                        for r in repair_radius:
                            record_dict[l][w][k][s][a][r] = 0
                        

    for file in os.listdir(RES_DIR):
        if file.endswith('log'):
            log_file_path = RES_DIR / file
            # file_name = file.split('-')
            if '5e-05' in file:
                file = file.replace('5e-05', '0.00005')
            lr, weight_decay, kcoeff, support_loss, accuracy_loss, repair_radius = file.split('-')[:-6]
            lr = lr[2:]; weight_decay = weight_decay[12:]; kcoeff = kcoeff[6:]; support_loss = support_loss[12:]; accuracy_loss = accuracy_loss[13:]; repair_radius = repair_radius[13:]
            with open(log_file_path, 'r') as f:
                lines = f.readlines()
                test_acc = lines[-10].split(' ')[-1][:-2]
                record_dict[float(lr)][float(weight_decay)][float(kcoeff)][support_loss][accuracy_loss][float(repair_radius)] = float(test_acc)
                # for line in lines:
                #     if line.startswith('INFO:root:======== for network:5_2 and data:'):
                #         strings = line.split(' ')
                #         data = strings[4].split(':')[1]
                #         random = strings[7]

                #     if line.startswith('INFO:root:Marabou result'):
                #         tuples = line.split(':')[-1]
                #         result,time = tuples[1:-1].split(',')
                #         record_dict[data][random]['Marabou']['time'] = int(time[2:-2])
                #         record_dict[data][random]['Marabou']['result'] = result[1:-1]
                #     elif line.startswith('INFO:root:Delta-Marabou result'):
                #         tuples = line.split(':')[-1]
                #         result,time = tuples[1:-1].split(',')
                #         record_dict[data][random]['Delta']['time'] = int(time[2:-2])
                #         record_dict[data][random]['Delta']['result'] = result[1:-1]
    f.close()

    print(record_dict)

# plot accucary according to repair radius,and draw multiple lines on a graph

    for l in lr:
        for w in weight_decay:
            for k in kcoeff:
                for s in support_loss:
                    for a in accuracy_loss:
                        plt.figure()
                        plt.title('lr:{}, weight_decay:{}, kcoeff:{}, support_loss:{}, accuracy_loss:{}'.format(l, w, k, s, a))
                        plt.xlabel('repair_radius')
                        plt.ylabel('test_accuracy')
                        for r in repair_radius:
                            plt.plot(r, record_dict[l][w][k][s][a][r], 'o')
                        plt.savefig('lr_{}_weight_decay_{}_kcoeff_{}_support_loss_{}_accuracy_loss_{}.png'.format(l, w, k, s, a))
                        plt.close()
