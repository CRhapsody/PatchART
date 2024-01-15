from pathlib import Path
import os
import json
import csv

def read_art_result():
    FILE_PATH = '/home/chizm/PatchART/tools/acasxu/art.log'
    record_dict = {}
    net_list = [f'N{i}_{j}'       for i in range(2, 6) for j in range(1,10) ]
    for net in net_list:
        # if net == 'N4_2':
        #     continue
        record_dict[net] = {}
        for item in ['rsr', 'rgr', 'fdd', 'time']:
            if item == 'rsr' or item == 'rgr':
                record_dict[net][item] = '100.0'
            else:
                record_dict[net][item] = '0.0'


    with open(FILE_PATH, 'r') as f:
        lines = f.readlines()
        for line in lines:
            choose = line.split(' ')
            if len(choose) < 4:
                continue
            if choose[3].startswith('AcasNetID'):
                line = line.split('_')[:-1]
                net_id = f'N{line[-2]}_{line[-1]}'
                continue
            if choose[2].startswith('Test'):
                line = line.split(' ')
                fdd = f'{(1-float(line[-1][:-2]))*100.0:.1f}'
                continue
            if choose[2].startswith('After'):
                line = line.split(' ')
                time = f'{float(line[8][1:]):.1f}'                
                record_dict[net_id]['fdd'] = fdd
                record_dict[net_id]['time'] = time
                continue
    with open('art_acasxu_result.json', 'w') as f:
        json.dump(record_dict, f, indent=4)
    return record_dict


def read_care_result():
    record_dict = {}
    net_list = [f'N{i}_{j}' for i in range(2, 6) for j in range(1,10) ]
    for net in net_list:
        if net == 'N4_2':
            continue
        record_dict[net] = {}
        for item in ['rsr', 'rgr', 'fdd', 'time']:
            record_dict[net][item] = '0.0'
    
    DIR_PATH = '/home/chizm/PatchART/tools/acasxu/CARE-acasxu'
    for root, dirs, files in os.walk(DIR_PATH):
        for file in files:
            if file.endswith('.log'):
                net_id = file.split('.')[0]
                net_id = f'N{net_id[1]}_{net_id[2]}'
                with open(os.path.join(root, file), 'r') as f:
                    lines = f.readlines()
                    lines = lines[:-2]

                    line = lines[-3]

                    rsr = line.split(' ')[-3][:-1]
                    rsr = f'{(1- float(rsr))*100:.1f}'

                    line = lines[-2]
                    rgr = line.split(' ')[-3][:-1]
                    rgr = f'{(1- float(rgr))*100:.1f}'
                    fdd = line.split(' ')[-1][:-1]
                    fdd = f'{(1-float(fdd))*100:.1f}'

                    line = lines[-1]
                    time = line.split(' ')[-1][:-2]
                    time = f'{float(time):.1f}'
                    record_dict[net_id]['rsr'] = rsr
                    record_dict[net_id]['rgr'] = rgr
                    record_dict[net_id]['fdd'] = fdd
                    record_dict[net_id]['time'] = time
    with open('care_acasxu_result.json', 'w') as f:
        json.dump(record_dict, f, indent=4)
    return record_dict


def read_prdnn_result():
    record_dict = {}
    net_list = [f'N{i}_{j}' for i in range(2, 6) for j in range(1,10) ]
    for net in net_list:
        if net == 'N4_2':
            continue
        record_dict[net] = {}
        for item in ['rsr', 'rgr', 'fdd', 'time']:
            record_dict[net][item] = '0.0'
    DIR_PATH = '/home/chizm/PatchART/tools/acasxu/prdnn-result'
    for root, dirs, files in os.walk(DIR_PATH):
        for file in files:
            if file.endswith('.log'):
                net_id = file.split('.')[0]
                net_id = f'N{net_id[-2]}_{net_id[-1]}'
                with open(os.path.join(root, file), 'r') as f:
                    lines = f.readlines()
                    lines = lines[:-2]
                    if len(lines) < 5:
                        print(net_id)
                        continue
                    line = lines[-4]

                    rsr = line.split(' ')[-1][:-1]
                    rsr = f'{(float(rsr))*100:.1f}'

                    line = lines[-2]
                    rgr = line.split(' ')[-1][:-2]
                    rgr = f'{100 - (float(rgr)):.1f}'

                    line = lines[-1]
                    fdd = line.split(' ')[-1][:-1]
                    fdd = f'{(100-float(fdd)):.1f}'

                    line = lines[-5]
                    time = line.split(':')[-2][1:]
                    time = time.split(',')[0]
                    time = f'{float(time):.1f}'
                    record_dict[net_id]['rsr'] = rsr
                    record_dict[net_id]['rgr'] = rgr
                    record_dict[net_id]['fdd'] = fdd
                    record_dict[net_id]['time'] = time
    with open('prdnn_acasxu_result.json', 'w') as f:
        json.dump(record_dict, f, indent=4)
    return record_dict


def read_reassure_result():
    record_dict = {}
    net_list = [f'N{i}_{j}' for i in range(2, 6) for j in range(1,10) ]
    for net in net_list:
        if net == 'N4_2':
            continue
        record_dict[net] = {}
        for item in ['rsr', 'rgr', 'fdd', 'time']:
            if item == 'rsr':
                record_dict[net][item] = '100.0'
            else:
                record_dict[net][item] = '0.0'

    FILE_PATH = '/home/chizm/PatchART/tools/acasxu/reassure-acas.txt'

    with open(FILE_PATH, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            line = line[2:]
            # convert the line to string
            line = ' '.join(line)
            if line.startswith('N'):
                net_id = line.split(' ')[0]
                net_id = f'N{net_id[-2]}_{net_id[-1]}'
                continue

            if line.startswith('fidelity'):
                line = line.split(' ')
                fdd = line[-1][:-1]
                fdd = fdd = f'{(100-float(fdd)):.1f}'
                continue
            if line.startswith('Generalization'):
                line = line.split(' ')
                rgr = line[-1][:-1]
                rgr = f'{float(rgr):.1f}'
                continue
            if line.startswith('Time'):
                line = line.split(' ')
                time = line[-1][:-1]
                time = f'{float(time):.1f}'
                record_dict[net_id]['fdd'] = fdd
                record_dict[net_id]['rgr'] = rgr
                record_dict[net_id]['time'] = time
                continue
    with open('reassure_acasxu_result.json', 'w') as f:
        json.dump(record_dict, f, indent=4)
    return record_dict


def read_patchrepair_result():
    FILE_PATH = '/home/chizm/PatchART/tools/acasxu/patchrepair.log'
    record_dict = {}
    net_list = [f'N{i}_{j}'       for i in range(2, 6) for j in range(1,10) ]
    for net in net_list:
        # if net == 'N4_2':
        #     continue
        record_dict[net] = {}
        for item in ['rsr', 'rgr', 'fdd', 'time']:
            if item == 'rsr' or item == 'rgr':
                record_dict[net][item] = '100.0'
            else:
                record_dict[net][item] = '0.0'


    with open(FILE_PATH, 'r') as f:
        lines = f.readlines()
        for line in lines:
            choose = line.split(' ')
            if len(choose) < 4:
                continue
            if choose[3].startswith('AcasNetID'):
                line = line.split('_')[:-1]
                net_id = f'N{line[-2]}_{line[-1]}'
                continue
            if choose[2].startswith('Test'):
                line = line.split(' ')
                fdd = f'{(1-float(line[-1][:-2]))*100.0:.1f}'
                continue
            if choose[2].startswith('After'):
                line = line.split(' ')
                time = f'{float(line[8][1:]):.1f}'                
                record_dict[net_id]['fdd'] = fdd
                record_dict[net_id]['time'] = time
                continue
    with open('patchrepair_acasxu_result.json', 'w') as f:
        json.dump(record_dict, f, indent=4)

    return record_dict





def convert_to_csv_rq():
    patchrepair_result = read_patchrepair_result()
    care_result = read_care_result()
    prdnn_result = read_prdnn_result()
    reassure_result = read_reassure_result()
    art_result = read_art_result()

    net_list = [f'N{i}_{j}' for i in range(2, 6) for j in range(1,10) ]
    with open('rq_acasxu_result.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Model', 'RSR/%', 'RGR/%', 'FDD/%', 'time/s'])
        writer.writerow(['', 'CARE', 'PRDNN', 'REASS', 'ART', 'Ours', 'CARE', 'PRDNN', 'REASS', 'ART', 'Ours', 'CARE', 'PRDNN', 'REASS', 'ART', 'Ours', 'CARE', 'PRDNN', 'REASS', 'ART', 'Ours'])
        for net in net_list:
            if net == 'N4_2':
                continue
            writer.writerow(['$N_{'+f'{net[1]},{net[3]}'+'}$', care_result[net]['rsr'] + '[]', prdnn_result[net]['rsr']+ '~[]', reassure_result[net]['rsr']+ '~~[]', art_result[net]['rsr']+ '~[]', patchrepair_result[net]['rsr']+ '[]', care_result[net]['rgr']+ '[]', prdnn_result[net]['rgr']+ '~[]', reassure_result[net]['rgr']+ '~~[]', art_result[net]['rgr']+ '~[]', patchrepair_result[net]['rgr']+ '[]', care_result[net]['fdd']+ '[]', prdnn_result[net]['fdd']+ '~[]', reassure_result[net]['fdd']+ '~~[]', art_result[net]['fdd']+ '~[]', patchrepair_result[net]['fdd']+ '[]', care_result[net]['time']+ '[]', prdnn_result[net]['time']+ '~~~[]', reassure_result[net]['time']+ '~~[]', art_result[net]['time']+ '~[]', patchrepair_result[net]['time']+ '[]'])
    print('1st Done!')
    # avg
    avg = {}
    for item in ['rsr', 'rgr', 'fdd', 'time']:
        avg[item] = {}
        for method in ['care', 'prdnn', 'reassure', 'art', 'Ours']:
            avg[item][method] = 0.0
    for net in net_list:
        if net == 'N4_2':
            continue
        for item in ['rsr', 'rgr', 'fdd', 'time']:
            avg[item]['care'] += float(care_result[net][item])
            avg[item]['prdnn'] += float(prdnn_result[net][item])
            avg[item]['reassure'] += float(reassure_result[net][item])
            avg[item]['art'] += float(art_result[net][item])
            avg[item]['Ours'] += float(patchrepair_result[net][item])
    for item in ['rsr', 'rgr', 'fdd', 'time']:
        for method in ['care', 'prdnn', 'reassure', 'art', 'Ours']:
            avg[item][method] /= 35
            if method == 'reaasure':
                avg[item][method] = f'{avg[item][method]:.1f}' + '~~[]'
            elif method == 'art':
                avg[item][method] = f'{avg[item][method]:.1f}' + '~[]'
            elif method == 'prdnn' and item != 'time':
                avg[item][method] = f'{avg[item][method]:.1f}' + '~[]'
            elif method == 'prdnn' and item == 'time':
                avg[item][method] = f'{avg[item][method]:.1f}' + '~~~[]'
            else:
                avg[item][method] = f'{avg[item][method]:.1f}' + '[]'
    with open('rq_acasxu_result.csv', 'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Average', avg['rsr']['care'], avg['rsr']['prdnn'], avg['rsr']['reassure'], avg['rsr']['art'], avg['rsr']['Ours'], avg['rgr']['care'], avg['rgr']['prdnn'], avg['rgr']['reassure'], avg['rgr']['art'], avg['rgr']['Ours'], avg['fdd']['care'], avg['fdd']['prdnn'], avg['fdd']['reassure'], avg['fdd']['art'], avg['fdd']['Ours'], avg['time']['care'], avg['time']['prdnn'], avg['time']['reassure'], avg['time']['art'], avg['time']['Ours']])
    print('2nd Done!')
        
        

if __name__ == '__main__':
    # read_patchrepair_result()
    # read_care_result()
    # read_prdnn_result()
    # read_reassure_result()
    # read_art_result()
    convert_to_csv_rq()