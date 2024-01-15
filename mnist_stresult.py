from pathlib import Path
import os
import json
import csv
FNN_SMALL_ACC = 0.9658
FNN_BIG_ACC = 0.9718
CNN_SMALL_ACC = 0.9827

# construct the function to read log from every tool
def read_patchrepair_result():
    DIR_PATH = '/home/chizm/PatchART/results/mnist/repair/debug'
    
    # construct the dictionary to store the result
    result_dict = {}
    for net in ['FNN_small', 'FNN_big', 'CNN_small']:
        result_dict[net] = {}
        for radius in [0.05, 0.1, 0.3]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [50,100,200,500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}
                for patch_format in ['small', 'big']:
                    result_dict[net][radius][data_number][patch_format] = {}
                    for item in ['drawdown', 'rs', 'rgr', 'dsr', 'dgsr', 'time']:
                        result_dict[net][radius][data_number][patch_format][item] = 0
    
    
    
    
    # traverse all the files
    for root, dirs, files in os.walk(DIR_PATH):
        for file in files:
            if file.endswith('.log'):
                name =  file.split('-')
                net = name[0]
                data_number = str(name[1])
                patch_format = name[3]



                file_path = os.path.join(root, file)

                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    # remove the white line
                    lines = lines[:-3]

                    # for line in lines:
                    #     # test the begin of the line
                    #     if line.startswith('  repair_radius:'):
                            # split the line
                    # radius
                    line = lines[24]
                    line = line.split(' ')
                    # remove the \n
                    # get the radius
                    radius = float(line[-1])
                    radius = str(radius)
                    # get the drawdown
                    line = lines[-8]
                    line = line.split(' ')
                    drawdown = float(line[-1])
                    if net == 'FNN_small':
                        drawdown = FNN_SMALL_ACC - drawdown
                    elif net == 'FNN_big':
                        drawdown = FNN_BIG_ACC - drawdown
                    elif net == 'CNN_small':
                        drawdown = CNN_SMALL_ACC - drawdown
                    drawdown = f'{drawdown*100:.1f}'
                    # get the rs
                    line = lines[-7]
                    line = line.split(' ')
                    rs = f'{float(line[-1])*100:.1f}'
                    #rs = round(rs, 2)
                    # get the rgr
                    line = lines[-5]
                    line = line.split(' ')
                    rgr = f'{float(line[-1])*100:.1f}'
                    #rgr = round(rgr, 2)
                    # get the time
                    line = lines[-1]
                    line = line.split(' ')

                    time = line[-1]
                    time = time[:-3]
                    time = float(time)
                    time = f'{time:.1f}'

                    # record the result
                    result_dict[net][radius][data_number][patch_format]['drawdown'] = drawdown
                    result_dict[net][radius][data_number][patch_format]['rs'] = rs
                    result_dict[net][radius][data_number][patch_format]['rgr'] = rgr
                    result_dict[net][radius][data_number][patch_format]['time'] = time

    # save the result to json file
    with open('patchrepair_mnist_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    return result_dict   

def read_patchrepair_label_result():
    DIR_PATH = '/home/chizm/PatchART/results/mnist/repair/label'
    
    # construct the dictionary to store the result
    result_dict = {}
    for net in ['FNN_small', 'FNN_big', 'CNN_small']:
        result_dict[net] = {}
        for radius in [0.05, 0.1, 0.3]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [50,100,200,500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}
                for patch_format in ['small', 'big']:
                    result_dict[net][radius][data_number][patch_format] = {}
                    for item in ['drawdown', 'rs', 'rgr', 'dsr', 'dgsr', 'time']:
                        result_dict[net][radius][data_number][patch_format][item] = 0

    
    
    
    # traverse all the files
    for root, dirs, files in os.walk(DIR_PATH):
        for file in files:
            if file.endswith('.log'):
                name =  file.split('-')
                net = name[0]
                radius = name[1]
                data_number = str(name[2])
                patch_format = name[3]



                file_path = os.path.join(root, file)

                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    # remove the white line
                    lines = lines[:-3]

                    # for line in lines:
                    #     # test the begin of the line
                    #     if line.startswith('  repair_radius:'):
                            # split the line
                    # radius
                    # line = lines[24]
                    # line = line.split(' ')
                    # # remove the \n
                    # # get the radius
                    # radius = float(line[-1])
                    # radius = str(radius)
                    # get the drawdown
                    line = lines[-7]
                    line = line.split(' ')
                    drawdown = float(line[-1])
                    if net == 'FNN_small':
                        drawdown = FNN_SMALL_ACC - drawdown
                    elif net == 'FNN_big':
                        drawdown = FNN_BIG_ACC - drawdown
                    elif net == 'CNN_small':
                        drawdown = CNN_SMALL_ACC - drawdown
                    drawdown = f'{drawdown*100:.1f}'
                    # get the rs
                    line = lines[-6]
                    line = line.split(' ')
                    rs = f'{float(line[-1])*100:.1f}'
                    #rs = round(rs, 2)
                    # get the rgr
                    line = lines[-5]
                    line = line.split(' ')
                    rgr = f'{float(line[-1])*100:.1f}'
                    #rgr = round(rgr, 2)
                    # get the time
                    line = lines[-1]
                    line = line.split(' ')

                    time = line[-1]
                    time = time[:-3]
                    time = float(time)
                    time = f'{time:.1f}'
                    # record the result
                    result_dict[net][radius][data_number][patch_format]['drawdown'] = drawdown
                    result_dict[net][radius][data_number][patch_format]['rs'] = rs
                    result_dict[net][radius][data_number][patch_format]['rgr'] = rgr
                    result_dict[net][radius][data_number][patch_format]['time'] = time

    # save the result to json file
    with open('patchrepair_label_mnist_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    return result_dict  

def read_care_result():
    DIR_PATH = '/home/chizm/PatchART/tools/mnist/care-mnist'
    
    # construct the dictionary to store the result
    result_dict = {}
    for net in ['FNN_small', 'FNN_big', 'CNN_small']:
        result_dict[net] = {}
        for radius in [0.05, 0.1, 0.3]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [50,100,200,500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}
                for item in ['drawdown', 'rs', 'rgr', 'dsr', 'dgsr', 'time']:
                    result_dict[net][radius][data_number][item] = 0
    
    
    
    
    # traverse all the files
    for root, dirs, files in os.walk(DIR_PATH):
        for file in files:
            if file.endswith('.log'):
                name =  file.split('-')
                net = name[0]
                # judge the net
                if net == 'fnnsmall':
                    net = 'FNN_small'
                elif net == 'fnnbig':
                    net = 'FNN_big'
                elif net == 'cnnsmall':
                    net = 'CNN_small'

                data_number = str(name[1])
                radius = name[2]
                # remove the .log
                radius = radius[:-4]



                file_path = os.path.join(root, file)

                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    # remove the white line
                    lines = lines[:-2]

                    # for line in lines:
                    #     # test the begin of the line
                    #     if line.startswith('  repair_radius:'):
                            # split the line
                    
                    # get the drawdown
                    line = lines[2]
                    line = line.split(' ')
                    drawdown = float(line[-1])
                    if net == 'FNN_small':
                        drawdown = FNN_SMALL_ACC - drawdown
                    elif net == 'FNN_big':
                        drawdown = FNN_BIG_ACC - drawdown
                    elif net == 'CNN_small':
                        drawdown = CNN_SMALL_ACC - drawdown
                    drawdown = f'{drawdown*100:.1f}'
                    # get the rs
                    line = lines[1]
                    line = line.split(' ')
                    rs = 1. - float(line[-3][:-1])
                    rs = f'{rs*100:.1f}'
                    #rs = round(rs, 2)
                    # get the rgr
                    line = lines[2]
                    line = line.split(' ')
                    rgr = 1. - float(line[-3][:-1])
                    rgr = f'{rgr*100:.1f}'
                    #rgr = round(rgr, 2)
                    # get the time
                    line = lines[3]
                    line = line.split(' ')
                    time1 = line[-1]
                    time1 = time1[:-2]
                    time1 = float(time1)

                    line = lines[4]
                    line = line.split(' ')
                    time2 = line[-1]
                    time2 = time2[:-2]
                    time2 = float(time2)
                    time = time1 + time2
                    time = f'{time:.1f}'

                    # record the result
                    result_dict[net][radius][data_number]['drawdown'] = drawdown
                    result_dict[net][radius][data_number]['rs'] = rs
                    result_dict[net][radius][data_number]['rgr'] = rgr
                    result_dict[net][radius][data_number]['time'] = time

    # save the result to json file
    with open('care_mnist_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    return result_dict     



def read_prdnn_result():
    DIR_PATH = '/home/chizm/PatchART/tools/mnist/prdnn-mnist'
    
    # construct the dictionary to store the result
    result_dict = {}
    for net in ['FNN_small', 'FNN_big', 'CNN_small']:
        result_dict[net] = {}
        for radius in [0.05, 0.1, 0.3]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [50,100,200,500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}
                for item in ['drawdown', 'rs', 'rgr', 'dsr', 'dgsr', 'time']:
                    if item != 'time':
                        result_dict[net][radius][data_number][item] = '--'
                    else:
                        result_dict[net][radius][data_number][item] = '--'
    
    
    
    
    # traverse all the files
    for root, dirs, files in os.walk(DIR_PATH):
        for file in files:
            if file.endswith('.log'):
                name =  file.split('-')
                net = name[0]
                net = net[:-4]
                # judge the net


                # data_number = str(name[1])
                # radius = name[2]
                # # remove the .log
                # radius = radius[:-4]



                file_path = os.path.join(root, file)

                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    # remove the white line
                    lines = lines[:-2]

                    for line in lines:
                        # test the begin of the line
                        if line.startswith('Repair model:'):
                            # split the line
                            line = line.split(' ')
                            radius = line[5]
                            data_number = line[8][:-1]
                            continue
                        if line.startswith('acc after repair:'):
                            line = line.split(' ')
                            line = line[-1].split(':')
                            drawdown = float(line[-1])
                            if net == 'FNN_small':
                                drawdown = FNN_SMALL_ACC - drawdown
                            elif net == 'FNN_big':
                                drawdown = FNN_BIG_ACC - drawdown
                            elif net == 'CNN_small':
                                drawdown = CNN_SMALL_ACC - drawdown
                            drawdown = f'{drawdown*100:.1f}'
                            result_dict[net][radius][data_number]['drawdown'] = drawdown
                            continue
                        if line.startswith('rsr after repair:'):
                            line = line.split(' ')
                            line = line[-1].split(':')
                            rs = f'{float(line[-1])*100:.1f}'
                            #rs = round(rs, 2)
                            result_dict[net][radius][data_number]['rs'] = rs
                            continue
                        if line.startswith('gene after repair:'):
                            line = line.split(' ')
                            line = line[-2].split(':')
                            rgr = f'{float(line[-1])*100:.1f}'
                            #rgr = round(rgr, 2)
                            result_dict[net][radius][data_number]['rgr'] = rgr
                            continue
                        if line.startswith('time cost:'):
                            line = line.split(' ')
                            line = line[-1].split(':')
                            time = float(line[-1][:-2])
                            time = f'{time:.1f}'
                            result_dict[net][radius][data_number]['time'] = time
                            continue
    # save the result to json file
    with open('prdnn_mnist_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    return result_dict   
                    

def read_reassure_result():
    DIR = '/home/chizm/PatchART/tools/mnist/reassure-mnist'
    result_dict = {}
    for net in ['FNN_small', 'FNN_big', 'CNN_small']:
        result_dict[net] = {}
        for radius in [0.05, 0.1, 0.3]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [50,100,200,500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}
                for item in ['drawdown', 'rs', 'rgr', 'dsr', 'dgsr', 'time']:
                    if item != 'time':
                        result_dict[net][radius][data_number][item] = '--'
                    else:
                        result_dict[net][radius][data_number][item] = '--'
    
    file = os.path.join(DIR, 'result.log')
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            line = line[2:]
            # convert the line to string
            line = ' '.join(line)
            if line.startswith('Repair MNIST'):
                line = line.split(' ')
                net = line[2]
                if net == 'fnnsmall':
                    net = 'FNN_small'
                elif net == 'fnnbig':
                    net = 'FNN_big'
                elif net == 'cnnsmall':
                    net = 'CNN_small'
                radius = line[4][:-1]
                continue
            if line.startswith('   Gamma :'):
                line  = line[3:]
                line = line.split(' ')

                data_number = line[7]
                continue
            if line.startswith('Acc after repair'):
                line = line.split(' ')
                drawdown = float(line[-1])
                if net == 'FNN_small':
                    drawdown = FNN_SMALL_ACC - drawdown
                elif net == 'FNN_big':
                    drawdown = FNN_BIG_ACC - drawdown
                elif net == 'CNN_small':
                    drawdown = CNN_SMALL_ACC - drawdown
                drawdown = f'{drawdown*100:.1f}'
                result_dict[net][radius][data_number]['drawdown'] = drawdown
                continue
            if line.startswith('Repair success rate'):
                line = line.split(' ')
                rs = f'{float(line[-1])*100:.1f}'
                #rs = round(rs, 2)
                result_dict[net][radius][data_number]['rs'] = rs
                continue
            if line.startswith('Generalization'):
                line = line.split(' ')
                rgr = f'{float(line[-1])*100:.1f}'
                #rgr = round(rgr, 2)
                result_dict[net][radius][data_number]['rgr'] = rgr
                continue
            if line.startswith('Time'):
                line = line.split(' ')
                time = float(line[-1])
                time = f'{time:.1f}'
                result_dict[net][radius][data_number]['time'] = time
                continue
    # save the result to json file
    with open('reassure_mnist_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    return result_dict

def read_aprnn_result():
    DIR_PATH = '/home/chizm/PatchART/tools/mnist/aprnn-mnist'
    
    # construct the dictionary to store the result
    result_dict = {}
    for net in ['FNN_small', 'FNN_big', 'CNN_small']:
        result_dict[net] = {}
        for radius in [0.05, 0.1, 0.3]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [50,100,200,500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}
                for item in ['drawdown', 'rs', 'rgr', 'dsr', 'dgsr', 'time']:
                    if item != 'time':
                        result_dict[net][radius][data_number][item] = '--'
                    else:
                        result_dict[net][radius][data_number][item] = '--'



    # traverse all the files
    for root, dirs, files in os.walk(DIR_PATH):
        for file in files:
            if file.endswith('.log'):
                name =  file.split('-')
                net = name[0]
                net = net[:-4]
                # judge the net


                # data_number = str(name[1])
                # radius = name[2]
                # # remove the .log
                # radius = radius[:-4]



                file_path = os.path.join(root, file)

                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    # remove the white line
                    lines = lines[:-2]

                    for line in lines:
                        # test the begin of the line
                        if line.startswith('Repair model:'):
                            # split the line
                            line = line.split(' ')
                            radius = line[5]
                            data_number = line[8][:-1]
                            continue
                        if line.startswith('acc after repair:'):
                            line = line.split(' ')
                            line = line[-1].split(':')
                            drawdown = float(line[-1])
                            
                            if net == 'FNN_small':
                                drawdown = FNN_SMALL_ACC - drawdown
                            elif net == 'FNN_big':
                                drawdown = FNN_BIG_ACC - drawdown
                            elif net == 'CNN_small':
                                drawdown = CNN_SMALL_ACC - drawdown
                            drawdown = f'{drawdown*100:.1f}'
                            result_dict[net][radius][data_number]['drawdown'] = drawdown
                            continue
                        if line.startswith('rsr after repair:'):
                            line = line.split(' ')
                            line = line[-1].split(':')
                            rs = f'{float(line[-1])*100:.1f}'
                            #rs = round(rs, 2)
                            result_dict[net][radius][data_number]['rs'] = rs
                            continue
                        if line.startswith('gene after repair:'):
                            line = line.split(' ')
                            line = line[-2].split(':')
                            rgr = f'{float(line[-1])*100:.1f}'
                            #rgr = round(rgr, 2)
                            result_dict[net][radius][data_number]['rgr'] = rgr
                            continue


                        if line.startswith('gene of pgd:'):
                            line = line.split(' ')
                            line = line[-2].split(':')
                            pgd = f'{float(line[-1])*100:.1f}'
                            #pgd = round(pgd, 2)
                            result_dict[net][radius][data_number]['dsr'] = pgd
                            continue
                        if line.startswith('time cost:'):
                            line = line.split(' ')
                            line = line[-1].split(':')
                            time = float(line[-1][:-2])
                            time = f'{time:.1f}'
                            result_dict[net][radius][data_number]['time'] = time
                            continue
    # save the result to json file
    with open('aprnn_mnist_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    return result_dict    

def convert_dict_to_csv():
    '''
    convert the result to csv file
    
    '''

    # read the result
    patchrepair_result = read_patchrepair_result()
    art_result = read_art_result()
    care_result = read_care_result()
    prdnn_result = read_prdnn_result()
    reassure_result = read_reassure_result()
    aprnn_result = read_aprnn_result()
    trades_result = read_trades_result()
    # construct the header
    header = ['net', 'radius', 'data_number', 'patch_format', 'rs', 'rgr', 'drowdown', 'dsr', 'dgr' 'time']
    # construct the csv file
    with open('mnist_result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        # write the patchart result
        for net in ['FNN_small', 'FNN_big', 'CNN_small']:
            for radius in [0.05, 0.1, 0.3]:
                radius = str(radius)
                for data_number in [50,100,200,500,1000]:
                    data_number = str(data_number)
                    for patch_format in ['small', 'big']:
                        drawdown = patchrepair_result[net][radius][data_number][patch_format]['drawdown']
                        rs = patchrepair_result[net][radius][data_number][patch_format]['rs']
                        gene = patchrepair_result[net][radius][data_number][patch_format]['rgr']
                        dsr = patchrepair_result[net][radius][data_number][patch_format]['dsr']
                        dgsr = patchrepair_result[net][radius][data_number][patch_format]['dgsr']




                        time = patchrepair_result[net][radius][data_number][patch_format]['time']
                        row = [net, radius, data_number, patch_format, drawdown, rs, gene, dsr, dgsr, time]
                        writer.writerow(row)
        # write the art result
        for net in ['FNN_small', 'FNN_big', 'CNN_small']:
            for radius in [0.05, 0.1, 0.3]:
                radius = str(radius)
                for data_number in [50,100,200,500,1000]:
                    data_number = str(data_number)
                    drawdown = art_result[net][radius][data_number]['drawdown']
                    rs = art_result[net][radius][data_number]['rs']
                    gene = art_result[net][radius][data_number]['gene']
                    time = art_result[net][radius][data_number]['time']
                    row = [net, radius, data_number, 'art', drawdown, rs, gene, time]
                    writer.writerow(row)




        # write the care result
        for net in ['FNN_small', 'FNN_big', 'CNN_small']:
            for radius in [0.05, 0.1, 0.3]:
                radius = str(radius)
                for data_number in [50,100,200,500,1000]:
                    data_number = str(data_number)
                    drawdown = care_result[net][radius][data_number]['drawdown']
                    rs = care_result[net][radius][data_number]['rs']
                    gene = care_result[net][radius][data_number]['gene']
                    time = care_result[net][radius][data_number]['time']
                    row = [net, radius, data_number, 'care', drawdown, rs, gene, time]
                    writer.writerow(row)
        # write the prdnn result
        for net in ['FNN_small', 'FNN_big', 'CNN_small']:
            for radius in [0.05, 0.1, 0.3]:
                radius = str(radius)
                for data_number in [50,100,200,500,1000]:
                    data_number = str(data_number)
                    drawdown = prdnn_result[net][radius][data_number]['drawdown']
                    rs = prdnn_result[net][radius][data_number]['rs']
                    gene = prdnn_result[net][radius][data_number]['gene']
                    time = prdnn_result[net][radius][data_number]['time']
                    row = [net, radius, data_number, 'prdnn', drawdown, rs, gene, time]
                    writer.writerow(row)
        # write the reassure result
        for net in ['FNN_small', 'FNN_big', 'CNN_small']:
            for radius in [0.05, 0.1, 0.3]:
                radius = str(radius)
                for data_number in [50,100,200,500,1000]:
                    data_number = str(data_number)
                    drawdown = reassure_result[net][radius][data_number]['drawdown']
                    rs = reassure_result[net][radius][data_number]['rs']
                    gene = reassure_result[net][radius][data_number]['gene']
                    time = reassure_result[net][radius][data_number]['time']
                    row = [net, radius, data_number, 'reassure', drawdown, rs, gene, time]
                    writer.writerow(row)
        
     # write the aprnn result
        for net in ['FNN_small', 'FNN_big', 'CNN_small']:
            for radius in [0.05, 0.1, 0.3]:
                radius = str(radius)
                for data_number in [50,100,200,500,1000]:
                    data_number = str(data_number)
                    drawdown = aprnn_result[net][radius][data_number]['drawdown']
                    rs = aprnn_result[net][radius][data_number]['rs']
                    gene = aprnn_result[net][radius][data_number]['gene']
                    time = aprnn_result[net][radius][data_number]['time']
                    row = [net, radius, data_number, 'aprnn', drawdown, rs, gene, time]
                    writer.writerow(row)


    # write another csv file for pgd result
    header = ['net', 'radius', 'data_number', 'tool', 'autoattack_success']
    pgd_r = pgd_result()

    with open('mnist_pgd_result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for net in ['FNN_small', 'FNN_big', 'CNN_small']:
            for radius in [0.05, 0.1, 0.3]:
                radius = str(radius)
                for data_number in [50,100,200,500,1000]:
                    data_number = str(data_number)
                    for tool in ['small','big', 'ori', 'adv-train', 'art', 'care', 'prdnn', 'reassure']:
                        pgd = pgd_r[net][radius][data_number][tool]['autoattack_success']
                        row = [net, radius, data_number, tool, pgd]
                        writer.writerow(row)
        # write the aprnn result
        for net in ['FNN_small', 'FNN_big', 'CNN_small']:
            for radius in [0.05, 0.1, 0.3]:
                radius = str(radius)
                for data_number in [50,100,200,500,1000]:
                    data_number = str(data_number)
                    pgd = aprnn_result[net][radius][data_number]['autoattack_success']
                    row = [net, radius, data_number, 'aprnn', pgd]
                    writer.writerow(row)

    # write another csv file for generalization result
    header = ['net', 'radius', 'data_number', 'tool', 'defense success of apgd & apgdt']
    generalization_r = generalizaiton_result()
    with open('mnist_generalization_result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for net in ['FNN_small', 'FNN_big', 'CNN_small']:
            for radius in [0.05, 0.1, 0.3]:
                radius = str(radius)
                for data_number in [50,100,200,500,1000]:
                    data_number = str(data_number)
                    for tool in ['PatchRepair', 'original', 'adv_training', 'art', 'care', 'prdnn', 'reassure', 'aprnn']:
                        defense = generalization_r[net][radius][data_number][tool]['defense success of apgd & apgdt']
                        row = [net, radius, data_number, tool, defense]
                        writer.writerow(row)


def read_art_result():
    DIR_PATH = '/home/chizm/PatchART/tools/results/mnist/ART'
    
    # construct the dictionary to store the result
    result_dict = {}
    for net in ['FNN_small', 'FNN_big', 'CNN_small']:
        result_dict[net] = {}
        for radius in [0.05, 0.1, 0.3]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [50,100,200,500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}
                # for patch_format in ['small', 'big']:
                #     result_dict[net][radius][data_number][patch_format] = {}
                for item in ['drawdown', 'rs', 'rgr', 'dsr', 'dgsr', 'time']:
                    if item != 'time':
                        result_dict[net][radius][data_number][item] = '--'
                    else:
                        result_dict[net][radius][data_number][item] = '--'
    
    
    
    
    # traverse all the files
    for root, dirs, files in os.walk(DIR_PATH):
        for file in files:
            if file.endswith('.log'):
                name =  file.split('-')
                net = name[0]
                radius = name[1]
                data_number = str(name[2])



                file_path = os.path.join(root, file)

                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    # remove the white line
                    lines = lines[:-3]

                    # for line in lines:
                    #     # test the begin of the line
                    #     if line.startswith('  repair_radius:'):
                            # split the line
                    # radius
                    # line = lines[24]
                    # line = line.split(' ')
                    # # remove the \n
                    # # get the radius
                    # radius = float(line[-1])
                    # radius = str(radius)
                    # get the drawdown
                    line = lines[-9]
                    line = line.split(' ')
                    drawdown = float(line[-1][:-2])
                    if net == 'FNN_small':
                        drawdown = FNN_SMALL_ACC - drawdown
                    elif net == 'FNN_big':
                        drawdown = FNN_BIG_ACC - drawdown
                    elif net == 'CNN_small':
                        drawdown = CNN_SMALL_ACC - drawdown
                    drawdown = f'{drawdown*100:.1f}'
                    # get the rs
                    line = lines[-8]
                    line = line.split(' ')
                    rs = float(line[-1][:-2])
                    rs = f'{rs*100:.1f}'
                    #rs = round(rs, 2)
                    # get the gene
                    line = lines[-6]
                    line = line.split(' ')
                    gene = float(line[-1][:-2])
                    # gene = round(gene, 2)
                    gene = f'{gene*100:.1f}'
                    # get the time
                    line = lines[-1]
                    line = line.split(' ')

                    time = line[-1]
                    time = time[:-3]
                    time = float(time)
                    time = f'{time:.1f}'

                    # record the result
                    result_dict[net][radius][data_number]['drawdown'] = drawdown
                    result_dict[net][radius][data_number]['rs'] = rs
                    result_dict[net][radius][data_number]['rgr'] = gene
                    result_dict[net][radius][data_number]['time'] = time

    # save the result to json file
    with open('art_mnist_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    return result_dict     


def read_trades_result():
    file_path = '/home/chizm/PatchART/tools/mnist/trade-mnist/result.txt'
    result_dict = {}
    for net in ['FNN_small', 'FNN_big', 'CNN_small']:
        result_dict[net] = {}
        for radius in [0.05, 0.1, 0.3]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [50,100,200,500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}
                # for patch_format in ['small', 'big']:
                #     result_dict[net][radius][data_number][patch_format] = {}
                for item in ['drawdown', 'rs', 'rgr', 'dsr', 'dgsr', 'time']:
                    if item != 'time':
                        result_dict[net][radius][data_number][item] = '--'
                    else:
                        result_dict[net][radius][data_number][item] = '--'
    
    
    
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line == '\n':
                continue
            line = line.split(' ')
            net = line[1]
            radius = line[3]
            data_number = line[2]

            line = line[5:]
            if line[-1] == '\n':
                line = line[:-1]
            for a in line:
                a = a.split(':')
                item = a[0]
                value = a[1][:-1]
                value = float(value)
                if item == 'test':
                    if net == 'FNN_small':
                        value = value - FNN_SMALL_ACC
                    elif net == 'FNN_big':
                        value = value - FNN_BIG_ACC
                    elif net == 'CNN_small':
                        value = value - CNN_SMALL_ACC
                    value = f'{-value*100:.1f}'
                    item = 'drawdown'
                elif item == 'attack':
                    item = 'rgr'
                    value = f'{value*100:.1f}'
                elif item == 'repair':
                    item = 'rs'
                    value = f'{value*100:.1f}'
                elif item == 'time':
                    value = f'{value:.1f}'
                result_dict[net][radius][data_number][item] = value

        
    # save the result to json file
    with open('trade_mnist_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    return result_dict

        



def pgd_result():
    
    pgd_result_file = '/home/chizm/PatchART/results/mnist/repair/autoattack/compare_autoattack_ac.txt'
    result_dict = {}
    for net in ['FNN_small', 'FNN_big', 'CNN_small']:
        result_dict[net] = {}
        for radius in [0.05, 0.1, 0.3]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [50,100,200,500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}
                for tool in ['small','big','small_label','big_label', 'ori', 'trades', 'art', 'care', 'prdnn', 'reassure']:
                    result_dict[net][radius][data_number][tool] = {}
                    for item in ['dsr']:
                        result_dict[net][radius][data_number][tool][item] = '--'
    with open(pgd_result_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line == '\n':
                continue
            line = line.split(' ')
            net = line[1]
            radius = line[2]
            data_number = line[3]

            line = line[5:]
            if line[-1] == '\n':
                line = line[:-1]
            for a in line:
                a = a.split(':')
                tool = a[0]
                if tool == 'adv-train':
                    tool = 'trades'
                # if tool == 'reassure':
                #     print(1)
                if a[1][-1] == '\n' or a[1][-1] == ',':
                    pgd = a[1][:-1]
                else:
                    pgd = a[1]
                pgd = 1- (float(pgd)/float(data_number))
                pgd = f'{pgd*100:.1f}'
                #pgd = round(pgd, 2)
                result_dict[net][radius][data_number][tool]['dsr'] = pgd

    # save the result to json file
    with open('pgd_mnist_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    return result_dict

def generalizaiton_result():
    pgd_result_file = '/home/chizm/PatchART/results/mnist/repair/generalization/compare/compare_generalization.txt'
    result_dict = {}
    for net in ['FNN_small', 'FNN_big', 'CNN_small']:
        result_dict[net] = {}
        for radius in [0.05, 0.1, 0.3]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [50,100,200,500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}
                for tool in ['small', 'big', 'small_label', 'big_label', 'label', 'original', 'trades', 'art', 'care', 'prdnn', 'reassure', 'aprnn', ]:
                    result_dict[net][radius][data_number][tool] = {}
                    for item in ['dgsr']:
                        if tool == 'reassure' and (data_number != 1000 or data_number != 500) and net != 'FNN_small':
                            result_dict[net][radius][data_number][tool][item] = '0.0'
                        elif tool == 'reassure' and (data_number != 1000) and net == 'FNN_small':
                            result_dict[net][radius][data_number][tool][item] = '0.0'
                        else:
                            result_dict[net][radius][data_number][tool][item] = '--'
    with open(pgd_result_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line == '\n':
                continue
            line = line.split(' ')
            net = line[2][:-1]
            radius = line[4][:-1]
            data_number = line[6][:-1]

            line = line[7:]
            if line[-1] == '\n':
                line = line[:-1]
            for a in line:
                a = a.split(':')
                tool = a[0]
                if tool == 'adv_training':
                    tool = 'trades'
                elif tool == 'label':
                    tool = 'small_label'
                elif tool == 'PatchRepair':
                    tool = 'small'
                elif tool == 'P_big':
                    tool = 'big'
                pgd = a[1][:-1]
                pgd = float(pgd)
                pgd = f'{pgd*100:.1f}'
                #pgd = round(pgd, 2)
                result_dict[net][radius][data_number][tool]['dgsr'] = pgd

    # save the result to json file
    with open('pgd_generlization_mnist_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    return result_dict    

def convert_dict_to_csv_rq():
    patchrepair_result = read_patchrepair_result()
    patchrepair_label_result = read_patchrepair_label_result()
    art_result = read_art_result()
    care_result = read_care_result()
    prdnn_result = read_prdnn_result()
    reassure_result = read_reassure_result()
    aprnn_result = read_aprnn_result()
    trades_result = read_trades_result()
    pgd_r = pgd_result()
    generalization_r = generalizaiton_result()

    

    with open('mnist_rq1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')    
        writer.writerow(['Model','radius', 'num', 'RSR/%', 'DD/%' ,'Time/s'])
        writer.writerow(['', '', '', 'CARE', 'PRDNN', 'APRNN', 'REASSU', 'TRADE', 'ART', 'Ours', 'CARE', 'PRDNN', 'APRNN', 'REASS', 'TRADE', 'ART', 'Ours', 'CARE', 'PRDNN', 'APRNN', 'REASS', 'TRADE', 'ART', 'Ours'])
        for net in ['FNN_small', 'FNN_big', 'CNN_small']:
            for radius in [0.05, 0.1, 0.3]:
                radius = str(radius)
                for data_number in [50,100,200,500,1000]:
                
                    data_number = str(data_number)
                    # get the rs for every tool
                    rs_care = care_result[net][radius][data_number]['rs']+ '[]'
                    rs_prdnn = prdnn_result[net][radius][data_number]['rs']+ '[]'
                    rs_aprnn = aprnn_result[net][radius][data_number]['rs']+ '[]'
                    rs_reassure = reassure_result[net][radius][data_number]['rs']+ '[]'
                    rs_trades = trades_result[net][radius][data_number]['rs']+ '~~[]'
                    rs_art = art_result[net][radius][data_number]['rs']+ '~[]'
                    rs_patchrepair = patchrepair_result[net][radius][data_number]['small']['rs']+ '[]'

                    # round to 2 decimal
                    # rs_care = round(rs_care, 2)
                    # rs_prdnn = round(rs_prdnn, 2)
                    # rs_aprnn = round(rs_aprnn, 2)
                    # rs_reassure = round(rs_reassure, 2)
                    # rs_trades = round(rs_trades, 2)
                    # rs_art = round(rs_art, 2)
                    # rs_patchrepair = round(rs_patchrepair, 2)


                    # get the rgr for every tool and 


                    # rgr_care = round(rs_care, 2)
                    # rgr_prdnn = round(rs_prdnn, 2)
                    # rgr_aprnn = round(rs_aprnn, 2)
                    # rgr_reassure = round(rs_reassure, 2)
                    # rgr_trades = round(rs_trades, 2)
                    # rgr_art = round(rs_art, 2)
                    # rgr_patchrepair = round(rs_patchrepair, 2)

                    # get the drawdown for every tool
                    dd_care = care_result[net][radius][data_number]['drawdown'] + '[]'
                    dd_prdnn = prdnn_result[net][radius][data_number]['drawdown'] + '[]'
                    dd_aprnn = aprnn_result[net][radius][data_number]['drawdown'] + '[]'
                    dd_reassure = reassure_result[net][radius][data_number]['drawdown'] + '[]'
                    dd_trades = trades_result[net][radius][data_number]['drawdown'] + '~~[]'
                    dd_art = art_result[net][radius][data_number]['drawdown'] + '~[]'
                    dd_patchrepair = patchrepair_result[net][radius][data_number]['small']['drawdown'] + '[]'

                    # dd_care = round(dd_care, 2)
                    # dd_prdnn = round(dd_prdnn, 2)
                    # dd_aprnn = round(dd_aprnn, 2)
                    # dd_reassure = round(dd_reassure, 2)
                    # dd_trades = round(dd_trades, 2)
                    # dd_art = round(dd_art, 2)
                    # dd_patchrepair = round(dd_patchrepair, 2)

                    time_care = care_result[net][radius][data_number]['time']+ '[]'
                    time_prdnn = prdnn_result[net][radius][data_number]['time']+ '[]'
                    time_aprnn = aprnn_result[net][radius][data_number]['time']+ '[]'
                    time_reassure = reassure_result[net][radius][data_number]['time']+ '[]'
                    time_trades = trades_result[net][radius][data_number]['time']+ '~~[]'
                    time_art = art_result[net][radius][data_number]['time']+  '~[]'
                    time_patchrepair = patchrepair_result[net][radius][data_number]['small']['time']+ '[]'
                    writer.writerow([net, radius, data_number, rs_care, rs_prdnn, rs_aprnn, rs_reassure, rs_trades, rs_art, rs_patchrepair, dd_care, dd_prdnn, dd_aprnn, dd_reassure, dd_trades, dd_art, dd_patchrepair, time_care, time_prdnn, time_aprnn, time_reassure, time_trades, time_art, time_patchrepair])
    
    # for pgd and generalization
    with open('mnist_rq3.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')    
        writer.writerow(['Model','radius', 'num', 'RGR/%', 'DSR/%',  'DGSR/%'])
        writer.writerow(['', '', '', 'CARE', 'PRDNN', 'APRNN', 'REASS', 'TRADE', 'ART', 'Ours', 'CARE', 'PRDNN', 'APRNN', 'REASS', 'TRADE', 'ART', 'Ours', 'CARE', 'PRDNN', 'APRNN', 'REASS', 'TRADE', 'ART', 'Ours'])
        for net in ['FNN_small', 'FNN_big', 'CNN_small']:
            for radius in [0.05, 0.1, 0.3]:
                radius = str(radius)
                for data_number in [50,100,200,500,1000]:
                
                    data_number = str(data_number)

                    rgr_care = care_result[net][radius][data_number]['rgr'] + '[]'
                    rgr_prdnn = prdnn_result[net][radius][data_number]['rgr'] + '[]'
                    rgr_aprnn = aprnn_result[net][radius][data_number]['rgr'] + '[]'
                    rgr_reassure = reassure_result[net][radius][data_number]['rgr'] + '[]'
                    rgr_trades = trades_result[net][radius][data_number]['rgr'] + '~~[]'
                    rgr_art = art_result[net][radius][data_number]['rgr'] + '~[]'
                    rgr_patchrepair = patchrepair_result[net][radius][data_number]['small']['rgr'] + '[]'
                    # get the dsr for every tool
                    dsr_care = pgd_r[net][radius][data_number]['care']['dsr']+ '[]'
                    dsr_prdnn = pgd_r[net][radius][data_number]['prdnn']['dsr']+ '[]'
                    dsr_aprnn = aprnn_result[net][radius][data_number]['dsr']+ '[]'
                    dsr_reassure = pgd_r[net][radius][data_number]['reassure']['dsr']+ '[]'
                    dsr_trades = pgd_r[net][radius][data_number]['trades']['dsr']+ '~~[]'
                    dsr_art = pgd_r[net][radius][data_number]['art']['dsr']+ '~[]'
                    dsr_patchrepair = pgd_r[net][radius][data_number]['small']['dsr']+ '[]'

                    dgsr_care = generalization_r[net][radius][data_number]['care']['dgsr']+ '[]'
                    dgsr_prdnn = generalization_r[net][radius][data_number]['prdnn']['dgsr']+ '[]'
                    dgsr_aprnn = generalization_r[net][radius][data_number]['aprnn']['dgsr']+ '[]'
                    dgsr_reassure = generalization_r[net][radius][data_number]['reassure']['dgsr']+ '[]'
                    dgsr_trades = generalization_r[net][radius][data_number]['trades']['dgsr']+ '~~[]'
                    dgsr_art = generalization_r[net][radius][data_number]['art']['dgsr']+ '~[]'
                    dgsr_patchrepair = generalization_r[net][radius][data_number]['small']['dgsr']+ '[]'

                    writer.writerow([net, radius, data_number, rgr_care, rgr_prdnn, rgr_aprnn, rgr_reassure, rgr_trades, rgr_art, rgr_patchrepair, dsr_care, dsr_prdnn, dsr_aprnn, dsr_reassure, dsr_trades, dsr_art, dsr_patchrepair, dgsr_care, dgsr_prdnn, dgsr_aprnn, dgsr_reassure, dgsr_trades, dgsr_art, dgsr_patchrepair])
                    
    # compare the rs, rgr, dd between patchrepair and patchrepair_label(small patch and big patch),2*2 = 4 tool               
    with open('mnist_rq4.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')    
        # writer.writerow(['Model','radius', 'num', 'RSR/%', 'DD/%', 'Time/s'])
        writer.writerow(['Model','radius', 'num', '', '', 'RSR/%', '', '', 'RGR/%',  '', '', 'DSR/%', '', '',  'DGSR/%', '', '', ])
        writer.writerow(['', '', '', 'PS', 'PB', 'LS', 'LB', 'PS', 'PB', 'LS', 'LB', 'PS', 'PB', 'LS', 'LB', 'PS', 'PB', 'LS', 'LB', 'PS', 'PB', 'LS', 'LB'])
        for net in ['FNN_small', 'FNN_big', 'CNN_small']:
            for radius in [0.05, 0.1, 0.3]:
                radius = str(radius)
                for data_number in [50,100,200,500,1000]:
                
                    data_number = str(data_number)    
                    # get the rs for patchrepair and patchrepair_label
                    rs_pb_small = patchrepair_result[net][radius][data_number]['small']['rs']+ '[]'
                    rs_pb_big = patchrepair_result[net][radius][data_number]['big']['rs']+ '[]'
                    rs_la_small = patchrepair_label_result[net][radius][data_number]['small']['rs']+ '[]'
                    rs_la_big = patchrepair_label_result[net][radius][data_number]['big']['rs']+ '[]'


                    # get the dd for patchrepair and patchrepair_label
                    # dd_pb_small = patchrepair_result[net][radius][data_number]['small']['drawdown']+ '[]'
                    # dd_pb_big = patchrepair_result[net][radius][data_number]['big']['drawdown']+ '[]'
                    # dd_la_small = patchrepair_label_result[net][radius][data_number]['small']['drawdown']+ '[]'
                    # dd_la_big = patchrepair_label_result[net][radius][data_number]['big']['drawdown']+ '[]'

                    # # get the time for patchrepair and patchrepair_label
                    # time_pb_small = patchrepair_result[net][radius][data_number]['small']['time']+ '[]'
                    # time_pb_big = patchrepair_result[net][radius][data_number]['big']['time']+ '[]'
                    # time_la_small = patchrepair_label_result[net][radius][data_number]['small']['time']+ '[]'
                    # time_la_big = patchrepair_label_result[net][radius][data_number]['big']['time']+ '[]'
                    rgr_pb_small = patchrepair_result[net][radius][data_number]['small']['rgr']+ '[]'
                    rgr_pb_big = patchrepair_result[net][radius][data_number]['big']['rgr']+ '[]'
                    rgr_la_small = patchrepair_label_result[net][radius][data_number]['small']['rgr']+ '[]'
                    rgr_la_big = patchrepair_label_result[net][radius][data_number]['big']['rgr']+ '[]'

                    # get the rs for patchrepair and patchrepair_label
                    dsr_pb_small = pgd_r[net][radius][data_number]['small']['dsr']+ '[]'
                    dsr_pb_big = pgd_r[net][radius][data_number]['big']['dsr']+ '[]'
                    dsr_la_small = pgd_r[net][radius][data_number]['small_label']['dsr']+ '[]'
                    dsr_la_big = pgd_r[net][radius][data_number]['big_label']['dsr']+ '[]'

                    # get the rgr for patchrepair and patchrepair_label
                    dgsr_pb_small = generalization_r[net][radius][data_number]['small']['dgsr']+ '[]'
                    dgsr_pb_big = generalization_r[net][radius][data_number]['big']['dgsr']+ '[]'
                    dgsr_la_small = generalization_r[net][radius][data_number]['small_label']['dgsr']+ '[]'
                    dgsr_la_big = generalization_r[net][radius][data_number]['big_label']['dgsr']+ '[]'
                    
                    writer.writerow([net, radius, data_number, rs_pb_small, rs_pb_big, rs_la_big, rs_la_small, rgr_pb_small, rgr_pb_big, rgr_la_big, rgr_la_small, dsr_pb_small, dsr_pb_big, dsr_la_big, dsr_la_small, dgsr_pb_small, dgsr_pb_big, dgsr_la_big, dgsr_la_small])

                    # writer.writerow([net, radius, data_number, rs_pb_small, rs_pb_big, rs_la_big, rs_la_small, dd_pb_small, dd_pb_big, dd_la_big, dd_la_small, time_pb_small, time_pb_big, time_la_big, time_la_small])




    # compare the dsr, dgsr between patchrepair and patchrepair_label(small patch and big patch),2*2 = 4 tool               
    # with open('mnist_rq4_2.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',')    
    #     writer.writerow(['Model','radius', 'num', 'RGR/%', 'DSR/%', 'DGSR/%'])
    #     writer.writerow(['', '', '', 'PS', 'PB', 'LS', 'LB', 'PS', 'PB', 'LS', 'LB'])
    #     for net in ['FNN_small', 'FNN_big', 'CNN_small']:
    #         for radius in [0.05, 0.1, 0.3]:
    #             radius = str(radius)
    #             for data_number in [50,100,200,500,1000]:
                
    #                 data_number = str(data_number)    
    #                     # get the rgr for patchrepair and patchrepair_label
    #                 rgr_pb_small = patchrepair_result[net][radius][data_number]['small']['rgr']+ '[]'
    #                 rgr_pb_big = patchrepair_result[net][radius][data_number]['big']['rgr']+ '[]'
    #                 rgr_la_small = patchrepair_label_result[net][radius][data_number]['small']['rgr']+ '[]'
    #                 rgr_la_big = patchrepair_label_result[net][radius][data_number]['big']['rgr']+ '[]'

    #                 # get the rs for patchrepair and patchrepair_label
    #                 dsr_pb_small = pgd_r[net][radius][data_number]['small']['dsr']+ '[]'
    #                 dsr_pb_big = pgd_r[net][radius][data_number]['big']['dsr']+ '[]'
    #                 dsr_la_small = pgd_r[net][radius][data_number]['small_label']['dsr']+ '[]'
    #                 dsr_la_big = pgd_r[net][radius][data_number]['big_label']['dsr']+ '[]'

    #                 # get the rgr for patchrepair and patchrepair_label
    #                 dgsr_pb_small = generalization_r[net][radius][data_number]['small']['dgsr']+ '[]'
    #                 dgsr_pb_big = generalization_r[net][radius][data_number]['big']['dgsr']+ '[]'
    #                 dgsr_la_small = generalization_r[net][radius][data_number]['small_label']['dgsr']+ '[]'
    #                 dgsr_la_big = generalization_r[net][radius][data_number]['big_label']['dgsr']+ '[]'

    #                 writer.writerow([net, radius, data_number, rgr_pb_small, rgr_pb_big, rgr_la_big, rgr_la_small, dsr_pb_small, dsr_pb_big, dsr_la_big, dsr_la_small, dgsr_pb_small, dgsr_pb_big, dgsr_la_big, dgsr_la_small])




if __name__ == '__main__':
    # art_result = read_art_result()
    # patchrepair_result = read_patchrepair_result()
    # care_result = read_care_result()
    # prdnn_result = read_prdnn_result()
    # reassure_result = read_reassure_result()
    # aprnn_result = read_aprnn_result()
    # patch_repair_label_result = read_patchrepair_label_result()
    # convert_dict_to_csv()
    # pgd_result()
    # generalizaiton_result()
    # read_trade_result()
    # convert_dict_to_csv()

    convert_dict_to_csv_rq()