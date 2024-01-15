from pathlib import Path
import os
import json
import csv
# FNN_SMALL_ACC = 0.9658
# FNN_BIG_ACC = 0.9718
# CNN_SMALL_ACC = 0.9827
RESNET18 = 0.8833
VGG19 = 0.9342

# construct the function to read log from every tool
def read_patchrepair_result():
    DIR_PATH = '/home/chizm/PatchART/results/cifar10/repair'
    
    # construct the dictionary to store the result
    result_dict = {}
    for net in ['vgg19', 'resnet18']:
        result_dict[net] = {}
        for radius in [4, 8]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [50,100,200,500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}


                for item in ['drawdown', 'rsr', 'rgr','time']:
                    result_dict[net][radius][data_number][item] = 0
    
    
    
    
    # traverse all the files
    for root, dirs, files in os.walk(DIR_PATH):
        for file in files:
            if file.endswith('.log'):
                name =  file.split('-')
                net = name[0]
                radius = name[1]
                data_number = str(name[3])
                



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
                    line = lines[-8]
                    line = line.split(' ')
                    drawdown = float(line[-1])
                    if net == 'vgg19':
                        drawdown = VGG19 - drawdown
                    elif net == 'resnet18':
                        drawdown = RESNET18 - drawdown
                    drawdown = f'{drawdown*100:.1f}'
                    # get the rs
                    line = lines[-7]
                    line = line.split(' ')
                    rs = f'{float(line[-1])*100:.1f}'
                    # get the gene
                    line = lines[-5]
                    line = line.split(' ')
                    rgr = f'{float(line[-1])*100:.1f}'
                    # get the time
                    line = lines[-1]
                    line = line.split(' ')

                    time = line[-1]
                    time = time[:-3]
                    time = float(time)
                    time = f'{time:.1f}'

                    # record the result
                    result_dict[net][radius][data_number]['drawdown'] = drawdown
                    result_dict[net][radius][data_number]['rsr'] = rs
                    result_dict[net][radius][data_number]['rgr'] = rgr
                    result_dict[net][radius][data_number]['time'] = time

    # save the result to json file
    with open('patchrepair_cifar_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    return result_dict   

def read_patchrepair_label_result():
    DIR_PATH = '/home/chizm/PatchART/results/cifar10/label'
    
    # construct the dictionary to store the result
    result_dict = {}
    for net in ['vgg19', 'resnet18']:        
        result_dict[net] = {}
        for radius in [4, 8]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [50,100,200,500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}
                # for patch_format in ['small', 'big']:
                #     result_dict[net][radius][data_number] = {}
                for item in ['drawdown', 'rsr', 'rgr','time']:
                    result_dict[net][radius][data_number][item] = 0

    
    
    
    # traverse all the files
    for root, dirs, files in os.walk(DIR_PATH):
        for file in files:
            if file.endswith('.log'):
                name =  file.split('-')
                net = name[0]
                radius = name[1]
                data_number = str(name[3])
                # patch_format = name[3]



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
                    if net == 'vgg19':
                        drawdown = VGG19 - drawdown
                    elif net == 'resnet18':
                        drawdown = RESNET18 - drawdown
                    drawdown = f'{drawdown*100:.1f}'
                    # get the rs
                    line = lines[-6]
                    line = line.split(' ')
                    rs = f'{float(line[-1])*100:.1f}'
                    # get the gene
                    line = lines[-5]
                    line = line.split(' ')
                    rgr = f'{float(line[-1])*100:.1f}'
                    # get the time
                    line = lines[-1]
                    line = line.split(' ')

                    time = line[-1]
                    time = time[:-3]
                    time = float(time)
                    time = f'{time:.1f}'

                    # record the result
                    result_dict[net][radius][data_number]['drawdown'] = drawdown
                    result_dict[net][radius][data_number]['rsr'] = rs
                    result_dict[net][radius][data_number]['rgr'] = rgr
                    result_dict[net][radius][data_number]['time'] = time

    # save the result to json file
    with open('patchrepair_label_cifar_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    return result_dict  

def read_care_result():
    DIR_PATH = '/home/chizm/PatchART/tools/cifar/care-cifar'
    
    # construct the dictionary to store the result
    result_dict = {}
    for net in ['vgg19', 'resnet18']:
        result_dict[net] = {}
        for radius in [4, 8]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [50,100,200,500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}
                for item in ['drawdown', 'rsr', 'rgr','time','dsr']:
                    if item != 'dsr':
                        result_dict[net][radius][data_number][item] = '--'
                    else:
                        if net == 'vgg19':
                            # full 0
                            result_dict[net][radius][data_number][item] = '0.0'
                        elif net == 'resnet18':
                            result_dict[net][radius][data_number][item] = '0.0'
    
    
    
    
    # traverse all the files
    for root, dirs, files in os.walk(DIR_PATH):
        for file in files:
            if file.endswith('.log'):
                name =  file.split('-')
                net = name[0]
                # judge the net
                if net == 'vgg':
                    net = 'vgg19'
                elif net == 'resnet':
                    net = 'resnet18'

                data_number = str(name[1])
                radius = name[2]
                # remove the .log
                radius = radius[:-4]



                file_path = os.path.join(root, file)

                with open(file_path, 'r') as f:
                    lines = f.readlines()

                    # resnet pgd result
                    if net == 'resnet18':
                        line = lines[-1]
                        line = line.split(' ')
                        line = line[-2].split(':')
                        pgd = float(line[-1])
                        pgd = f'{pgd*100:.1f}'
                        lines = lines[:-1]
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
                    
                    if net == 'vgg19':
                        drawdown = VGG19 - drawdown
                    elif net == 'resnet18':
                        drawdown = RESNET18 - drawdown
                    drawdown = f'{drawdown*100:.1f}'
                    # get the rs
                    line = lines[1]
                    line = line.split(' ')
                    rs = 1. - float(line[-3][:-1])
                    rs = f'{rs*100:.1f}'
                    # get the gene
                    line = lines[2]
                    line = line.split(' ')
                    gene = 1. - float(line[-3][:-1])
                    gene = f'{gene*100:.1f}'
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
                    result_dict[net][radius][data_number]['rsr'] = rs
                    result_dict[net][radius][data_number]['rgr'] = gene
                    result_dict[net][radius][data_number]['time'] = time
                    result_dict[net][radius][data_number]['dsr'] = pgd
    # save the result to json file
    with open('care_cifar_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    return result_dict     



def read_prdnn_result():
    DIR_PATH = '/home/chizm/PatchART/tools/cifar/prdnn-cifar'
    
    # construct the dictionary to store the result
    result_dict = {}
    for net in ['vgg19', 'resnet18']:
        result_dict[net] = {}
        for radius in [4, 8]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [50,100,200,500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}
                for item in ['drawdown', 'rsr', 'rgr','time']:
                    if item != 'time':
                        result_dict[net][radius][data_number][item] = '--'
                    else:
                        result_dict[net][radius][data_number][item] = '--'
                # full 0
                result_dict[net][radius][data_number]['dsr'] = '0.0'
    
    
    
    
    # traverse all the files
    for root, dirs, files in os.walk(DIR_PATH):
        for file in files:
            if file.endswith('.log'):
                name =  file.split('.')
                net = name[0]
                # net = net[:-4]
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
                            if net == 'vgg19':
                                drawdown = VGG19 - drawdown
                            elif net == 'resnet18':
                                drawdown = RESNET18 - drawdown
                            drawdown = f'{drawdown*100:.1f}'
                            result_dict[net][radius][data_number]['drawdown'] = drawdown
                            continue
                        if line.startswith('rsr after repair:'):
                            line = line.split(' ')
                            line = line[-1].split(':')
                            rs = f'{float(line[-1])*100:.1f}'
                            result_dict[net][radius][data_number]['rsr'] = rs
                            continue
                        if line.startswith('gene after repair:'):
                            line = line.split(' ')
                            line = line[-2].split(':')
                            rgr = f'{float(line[-1])*100:.1f}'
                            result_dict[net][radius][data_number]['rgr'] = rgr
                            continue
                        if line.startswith('time cost:'):
                            line = line.split(' ')
                            line = line[-1].split(':')
                            time = float(line[-1][:-2])
                            time = f'{time:.1f}'
                            result_dict[net][radius][data_number]['time'] = time
                            continue
    # for net in ['vgg19', 'resnet18']:
    #     for radius in [4, 8]:
    #         radius = str(radius)
    #         for data_number in [50,100,200,500,1000]:
    #             data_number = str(data_number)
    #             if result_dict[net][radius][data_number]['time'] == '--':
    #                 result_dict[net][radius][data_number]['dsgr'] = '--'
    # save the result to json file
    with open('prdnn_cifar_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    return result_dict   
                    

# def read_reassure_result():
#     DIR = '/home/chizm/PatchART/tools/cifar/reassure-cifar'
#     result_dict = {}
#     for net in ['vgg19', 'resnet18']:
#         result_dict[net] = {}
#         for radius in [4, 8]:
#             radius = str(radius)
#             result_dict[net][radius] = {}
#             for data_number in [50,100,200,500,1000]:
#                 data_number = str(data_number)
#                 result_dict[net][radius][data_number] = {}
#                 for item in ['drawdown', 'rs', 'gene','time']:
#                     if item != 'time':
#                         result_dict[net][radius][data_number][item] = '--'
#                     else:
#                         result_dict[net][radius][data_number][item] = '--'
    
#     file = os.path.join(DIR, 'result.log')
#     with open(file, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             line = line.split(' ')
#             line = line[2:]
#             # convert the line to string
#             line = ' '.join(line)
#             if line.startswith('Repair MNIST'):
#                 line = line.split(' ')
#                 net = line[2]
#                 if net == 'fnnsmall':
#                     net = 'FNN_small'
#                 elif net == 'fnnbig':
#                     net = 'FNN_big'
#                 elif net == 'cnnsmall':
#                     net = 'CNN_small'
#                 radius = line[4][:-1]
#                 continue
#             if line.startswith('   Gamma :'):
#                 line  = line[3:]
#                 line = line.split(' ')

#                 data_number = line[7]
#                 continue
#             if line.startswith('Acc after repair'):
#                 line = line.split(' ')
#                 drawdown = float(line[-1])
#                 if net == 'vgg19':
#                     drawdown = VGG19 - drawdown
#                 elif net == 'resnet18':
#                     drawdown = RESNET18 - drawdown
#                 result_dict[net][radius][data_number]['drawdown'] = drawdown
#                 continue
#             if line.startswith('Repair success rate'):
#                 line = line.split(' ')
#                 rs = float(line[-1])
#                 result_dict[net][radius][data_number]['rs'] = rs
#                 continue
#             if line.startswith('Generalization'):
#                 line = line.split(' ')
#                 gene = float(line[-1])
#                 result_dict[net][radius][data_number]['gene'] = gene
#                 continue
#             if line.startswith('Time'):
#                 line = line.split(' ')
#                 time = float(line[-1])
#                 result_dict[net][radius][data_number]['time'] = time
#                 continue
#     # save the result to json file
#     with open('reassure_mnist_result.json', 'w') as f:
#         json.dump(result_dict, f, indent=4)
#     return result_dict

def read_aprnn_result():
    DIR_PATH = '/home/chizm/PatchART/tools/cifar/aprnn-cifar'
    
    # construct the dictionary to store the result
    result_dict = {}
    for net in ['vgg19', 'resnet18']:
        result_dict[net] = {}
        for radius in [4, 8]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [50,100,200,500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}
                for item in ['drawdown', 'rsr', 'rgr','time','dsr']:
                    if item != 'time':
                        result_dict[net][radius][data_number][item] = '--'
                    else:
                        result_dict[net][radius][data_number][item] = '--'



    # traverse all the files
    for root, dirs, files in os.walk(DIR_PATH):
        for file in files:
            if file.endswith('.log'):
                name =  file.split('.')
                net = name[0]
                # net = net[:-4]
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
                            if net == 'vgg19':
                                drawdown = VGG19 - drawdown
                            elif net == 'resnet18':
                                drawdown = RESNET18 - drawdown
                            drawdown = f'{drawdown*100:.1f}'
                            result_dict[net][radius][data_number]['drawdown'] = drawdown
                            continue
                        if line.startswith('rsr after repair:'):
                            line = line.split(' ')
                            line = line[-1].split(':')
                            # rs = float(line[-1])
                            rs = f'{float(line[-1])*100:.1f}'
                            result_dict[net][radius][data_number]['rsr'] = rs
                            continue
                        if line.startswith('gene after repair:'):
                            line = line.split(' ')
                            line = line[-2].split(':')
                            rgr = f'{float(line[-1])*100:.1f}'
                            result_dict[net][radius][data_number]['rgr'] = rgr
                            continue


                        if line.startswith('gene of pgd:'):
                            line = line.split(' ')
                            line = line[-2].split(':')
                            pgd = f'{float(line[-1][:5])*100:.1f}'

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
    with open('aprnn_cifar_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    return result_dict    

def convert_dict_to_csv():
    '''
    convert the result to csv file
    
    '''
    import csv
    # read the result
    patchrepair_result = read_patchrepair_result()
    # art_result = read_art_result()
    care_result = read_care_result()
    prdnn_result = read_prdnn_result()
    # reassure_result = read_reassure_result()
    aprnn_result = read_aprnn_result()
    # construct the header
    patchrepair_label_result = read_patchrepair_label_result()
    header = ['net', 'radius', 'data_number', 'tool', 'drawdown', 'rsr', 'rgr', 'time']
    # construct the csv file
    with open('cifar_result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        # write the patchart result
        for net in ['vgg19', 'resnet18']:
            for radius in [4, 8]:
                radius = str(radius)
                for data_number in [50,100,200,500,1000]:
                    data_number = str(data_number)
                    for patch_format in ['PatchRepair']:
                        drawdown = patchrepair_result[net][radius][data_number]['drawdown']
                        rs = patchrepair_result[net][radius][data_number]['rs']
                        gene = patchrepair_result[net][radius][data_number]['gene']
                        time = patchrepair_result[net][radius][data_number]['time']
                        row = [net, radius, data_number, patch_format, drawdown, rs, gene, time]
                        writer.writerow(row)
        # write the art result
        # for net in ['vgg19', 'resnet18']:
        #     for radius in [4, 8]:
        #         radius = str(radius)
        #         for data_number in [50,100,200,500,1000]:
        #             data_number = str(data_number)
        #             drawdown = art_result[net][radius][data_number]['drawdown']
        #             rs = art_result[net][radius][data_number]['rs']
        #             gene = art_result[net][radius][data_number]['gene']
        #             time = art_result[net][radius][data_number]['time']
        #             row = [net, radius, data_number, 'art', drawdown, rs, gene, time]
        #             writer.writerow(row)




        # write the care result
        for net in ['vgg19', 'resnet18']:
            for radius in [4, 8]:
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
        for net in ['vgg19', 'resnet18']:
            for radius in [4, 8]:
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
        # for net in ['vgg19', 'resnet18']:
        #     for radius in [4, 8]:
        #         radius = str(radius)
        #         for data_number in [50,100,200,500,1000]:
        #             data_number = str(data_number)
        #             drawdown = reassure_result[net][radius][data_number]['drawdown']
        #             rs = reassure_result[net][radius][data_number]['rs']
        #             gene = reassure_result[net][radius][data_number]['gene']
        #             time = reassure_result[net][radius][data_number]['time']
        #             row = [net, radius, data_number, 'reassure', drawdown, rs, gene, time]
        #             writer.writerow(row)
        
     # write the aprnn result
        for net in ['vgg19', 'resnet18']:
            for radius in [4, 8]:
                radius = str(radius)
                for data_number in [50,100,200,500,1000]:
                    data_number = str(data_number)
                    drawdown = aprnn_result[net][radius][data_number]['drawdown']
                    rs = aprnn_result[net][radius][data_number]['rs']
                    gene = aprnn_result[net][radius][data_number]['gene']
                    time = aprnn_result[net][radius][data_number]['time']
                    row = [net, radius, data_number, 'aprnn', drawdown, rs, gene, time]
                    writer.writerow(row)
    # write the patchart label result

    # with open('cifar_label_result.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(header)
    #     # write the patchart result
        for net in ['vgg19', 'resnet18']:
            for radius in [4, 8]:
                radius = str(radius)
                for data_number in [50,100,200,500,1000]:
                    data_number = str(data_number)
                    # for patch_format in ['label']:
                    drawdown = patchrepair_label_result[net][radius][data_number]['drawdown']
                    rs = patchrepair_label_result[net][radius][data_number]['rs']
                    gene = patchrepair_label_result[net][radius][data_number]['gene']
                    time = patchrepair_label_result[net][radius][data_number]['time']
                    row = [net, radius, data_number, 'label', drawdown, rs, gene, time]
                    writer.writerow(row)
        # write the art result
        # for net in ['vgg19', 'resnet18']:
        #     for radius in [4, 8]:
        #         radius = str(radius)
        #         for data_number in [50,100,200,500,1000]:
        #             data_number = str(data_number)
        #             drawdown = art_result[net][radius][data_number]['drawdown']
        #             rs = art_result[net][radius][data_number]['rs']
        #             gene = art_result[net][radius][data_number]['gene']
        #             time = art_result[net][radius][data_number]['time']
        #             row = [net, radius, data_number, 'art', drawdown, rs, gene, time]
        #             writer.writerow(row)




        # write the care result
        for net in ['vgg19', 'resnet18']:
            for radius in [4, 8]:
                radius = str(radius)
                for data_number in [50,100,200,500,1000]:
                    data_number = str(data_number)
                    drawdown = care_result[net][radius][data_number]['drawdown']
                    rs = care_result[net][radius][data_number]['rs']
                    gene = care_result[net][radius][data_number]['gene']
                    time = care_result[net][radius][data_number]['time']
                    row = [ net, radius, data_number, 'care', drawdown, rs, gene, time]
                    writer.writerow(row)


    # write another csv file for pgd result
    header = ['net', 'radius', 'data_number', 'tool', 'autoattack_success']
    pgd_r = pgd_result()

    with open('cifar_pgd_result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for net in ['vgg19', 'resnet18']:
            for radius in [4, 8]:
                radius = str(radius)
                for data_number in [50,100,200,500,1000]:
                    data_number = str(data_number)
                    for tool in ['patch', 'ori', 'trades', 'label', 'care', 'prdnn', 'aprnn']:
                        pgd = pgd_r[net][radius][data_number][tool]['dsgr']
                        row = [net, radius, data_number, tool, pgd]
                        writer.writerow(row)
        # # write the aprnn result
        # for net in ['vgg19', 'resnet18']:
        #     for radius in [4, 8]:
        #         radius = str(radius)
        #         for data_number in [50,100,200,500,1000]:
        #             data_number = str(data_number)
        #             pgd = aprnn_result[net][radius][data_number]['dsgr']
        #             row = [net, radius, data_number, 'aprnn', pgd]
        #             writer.writerow(row)

    # write another csv file for generalization result
    header = ['net', 'radius', 'data_number', 'tool', 'dsgr']
    generalization_r = generalizaiton_result()
    with open('cifar_generalization_result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for net in ['vgg19', 'resnet18']:
            for radius in [4, 8]:
                radius = str(radius)
                for data_number in [50,100,200,500,1000]:
                    data_number = str(data_number)
                    for tool in ['PatchRepair', 'original', 'adv_training', 'label', 'care', 'prdnn', 'aprnn']:
                        defense = generalization_r[net][radius][data_number][tool]['dsgr']
                        row = [net, radius, data_number, tool, defense]
                        writer.writerow(row)


# def read_art_result():
#     DIR_PATH = '/home/chizm/PatchART/tools/results/mnist/ART'
    
#     # construct the dictionary to store the result
#     result_dict = {}
#     for net in ['FNN_small', 'FNN_big', 'CNN_small']:
#         result_dict[net] = {}
#         for radius in [4, 8]:
#             radius = str(radius)
#             result_dict[net][radius] = {}
#             for data_number in [50,100,200,500,1000]:
#                 data_number = str(data_number)
#                 result_dict[net][radius][data_number] = {}
#                 # for patch_format in ['small', 'big']:
#                 #     result_dict[net][radius][data_number] = {}
#                 for item in ['drawdown', 'rs', 'gene','time']:
#                     if item != 'time':
#                         result_dict[net][radius][data_number][item] = '--'
#                     else:
#                         result_dict[net][radius][data_number][item] = 'memory overflow'
    
    
    
    
#     # traverse all the files
#     for root, dirs, files in os.walk(DIR_PATH):
#         for file in files:
#             if file.endswith('.log'):
#                 name =  file.split('-')
#                 net = name[0]
#                 radius = name[1]
#                 data_number = str(name[2])



#                 file_path = os.path.join(root, file)

#                 with open(file_path, 'r') as f:
#                     lines = f.readlines()
#                     # remove the white line
#                     lines = lines[:-3]

#                     # for line in lines:
#                     #     # test the begin of the line
#                     #     if line.startswith('  repair_radius:'):
#                             # split the line
#                     # radius
#                     # line = lines[24]
#                     # line = line.split(' ')
#                     # # remove the \n
#                     # # get the radius
#                     # radius = float(line[-1])
#                     # radius = str(radius)
#                     # get the drawdown
#                     line = lines[-9]
#                     line = line.split(' ')
#                     drawdown = float(line[-1][:-2])
#                     if net == 'vgg19':
#                         drawdown = VGG19 - drawdown
#                     elif net == 'resnet18':
#                         drawdown = RESNET18 - drawdown
#                     # get the rs
#                     line = lines[-8]
#                     line = line.split(' ')
#                     rs = float(line[-1][:-2])
#                     # get the gene
#                     line = lines[-6]
#                     line = line.split(' ')
#                     gene = float(line[-1][:-2])
#                     # get the time
#                     line = lines[-1]
#                     line = line.split(' ')

#                     time = line[-1]
#                     time = time[:-3]
#                     time = float(time)

#                     # record the result
#                     result_dict[net][radius][data_number]['drawdown'] = drawdown
#                     result_dict[net][radius][data_number]['rs'] = rs
#                     result_dict[net][radius][data_number]['gene'] = gene
#                     result_dict[net][radius][data_number]['time'] = time

#     # save the result to json file
#     with open('art_mnist_result.json', 'w') as f:
#         json.dump(result_dict, f, indent=4)
#     return result_dict     

def read_trades_result():
    file_path = '/home/chizm/PatchART/tools/cifar/trade-cifar/result.txt'
    result_dict = {}
    for net in ['vgg19', 'resnet18']:
        result_dict[net] = {}
        for radius in [4, 8]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [50,100,200,500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}
                # for patch_format in ['small', 'big']:
                #     result_dict[net][radius][data_number][patch_format] = {}
                for item in ['drawdown', 'rsr', 'rgr', 'dsr', 'dgsr', 'time']:
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
                    if net == 'vgg19':
                        value = VGG19 - value
                    elif net == 'resnet18':
                        value = RESNET18 - value
                    value = f'{value*100:.1f}'
                    item = 'drawdown'
                elif item == 'attack':
                    item = 'rgr'
                    value = f'{value*100:.1f}'
                elif item == 'repair':
                    item = 'rsr'
                    value = f'{value*100:.1f}'
                elif item == 'time':
                    item = 'time'
                    value = f'{value:.1f}'
                else:
                    continue
                result_dict[net][radius][data_number][item] = value

        
    # save the result to json file
    with open('trade_cifar10_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    return result_dict    

def pgd_result():
    
    pgd_result_file = '/home/chizm/PatchART/results/cifar10/repair/autoattack/compare_autoattack_ac.txt'
    
    care_result = read_care_result()
    prdnn_result = read_prdnn_result()
    aprnn_result = read_aprnn_result()
    
    
    result_dict = {}
    for net in ['vgg19', 'resnet18']:
        result_dict[net] = {}
        for radius in [4, 8]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [50,100,200,500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}
                for tool in ['patch', 'ori', 'trades', 'label', 'care', 'prdnn', 'aprnn']:
                    result_dict[net][radius][data_number][tool] = {}
                    for item in ['dsr']:
                        if tool != 'care' and tool != 'prdnn' and tool != 'aprnn':
                            result_dict[net][radius][data_number][tool][item] = '--'
                        elif tool == 'care':
                            result_dict[net][radius][data_number][tool][item] = care_result[net][radius][data_number][item]
                        elif tool == 'prdnn':
                            result_dict[net][radius][data_number][tool][item] = prdnn_result[net][radius][data_number][item]
                        elif tool == 'aprnn':
                            result_dict[net][radius][data_number][tool][item] = aprnn_result[net][radius][data_number][item]
    with open(pgd_result_file, 'r') as f:
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
                tool = a[0]
                if tool == 'adv-train':
                    tool = 'trades'
                pgd = a[1][:-1]
                pgd = 1- float(pgd)/float(data_number)
                pgd = f'{pgd*100:.1f}'
                result_dict[net][radius][data_number][tool]['dsr'] = pgd

    # save the result to json file
    with open('pgd_cifar_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    return result_dict

def generalizaiton_result():
    pgd_result_file = '/home/chizm/PatchART/results/cifar10/repair/generalization/compare/compare_generalization.txt'
    result_dict = {}
    for net in ['vgg19', 'resnet18']:
        result_dict[net] = {}
        for radius in [4, 8]:
            radius = str(radius)
            result_dict[net][radius] = {}
            for data_number in [50,100,200,500,1000]:
                data_number = str(data_number)
                result_dict[net][radius][data_number] = {}
                for tool in ['PatchRepair', 'label', 'original', 'trades', 'care', 'prdnn','aprnn']:
                    result_dict[net][radius][data_number][tool] = {}
                    for item in ['dsgr']:
                        if tool == 'prdnn' and (data_number == '500' or data_number == '1000'):
                            result_dict[net][radius][data_number][tool][item] = '--'
                        else:
                            # because the pgd of repair data is 0
                            result_dict[net][radius][data_number][tool][item] = '0.0'
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
                if tool == 'label_repair':
                    tool = 'label'
                elif tool == 'adv_training':
                    tool = 'trades'
                pgd = a[1][:-1]
                pgd = float(pgd)
                pgd = f'{pgd*100:.1f}'
                result_dict[net][radius][data_number][tool]['dsgr'] = pgd

    # save the result to json file
    with open('pgd_generlization_cifar10_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    return result_dict    

def convert_dict_to_csv_rq():
    patchrepair_result = read_patchrepair_result()
    patchrepair_label_result = read_patchrepair_label_result()
    care_result = read_care_result()
    prdnn_result = read_prdnn_result()
    aprnn_result = read_aprnn_result()
    trades_result = read_trades_result()
    pgd_r = pgd_result()
    generalization_r = generalizaiton_result()

    # with open('cifar_rq1.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',')    
    #     writer.writerow(['Model','radius', 'num', 'RSR/%', 'RGR/%', 'DD/%'])
    #     writer.writerow(['', '', '', 'CARE', 'PRDNN', 'APRNN', 'TRADES', 'Ours', 'CARE', 'PRDNN', 'APRNN', 'TRADES', 'Ours', 'CARE', 'PRDNN', 'APRNN',  'TRADES', 'Ours'])
    #     for net in ['vgg19', 'resnet18']:
    #         for radius in [4, 8]:
    #             radius = str(radius)
    #             for data_number in [50,100,200,500,1000]:
    #                 data_number = str(data_number)

    #                 rs_care = care_result[net][radius][data_number]['rs']+ '[]'
    #                 rs_prdnn = prdnn_result[net][radius][data_number]['rs']+ '[]'
    #                 rs_aprnn = aprnn_result[net][radius][data_number]['rs']+ '[]'
    #                 rs_trades = trades_result[net][radius][data_number]['rs']+ '[]'
    #                 rs_patchrepair = patchrepair_result[net][radius][data_number]['rs']+ '[]'

    #                 rgr_care = care_result[net][radius][data_number]['rgr'] + '[]'
    #                 rgr_prdnn = prdnn_result[net][radius][data_number]['rgr'] + '[]'
    #                 rgr_aprnn = aprnn_result[net][radius][data_number]['rgr'] + '[]'
    #                 rgr_trades = trades_result[net][radius][data_number]['rgr'] + '[]'
    #                 rgr_patchrepair = patchrepair_result[net][radius][data_number]['rgr'] + '[]'

    #                 dd_care = care_result[net][radius][data_number]['drawdown'] + '[]'
    #                 dd_prdnn = prdnn_result[net][radius][data_number]['drawdown'] + '[]'
    #                 dd_aprnn = aprnn_result[net][radius][data_number]['drawdown'] + '[]'
    #                 dd_trades = trades_result[net][radius][data_number]['drawdown'] + '[]'
    #                 dd_patchrepair = patchrepair_result[net][radius][data_number]['drawdown'] + '[]'                    


    #                 writer.writerow([net, radius, data_number, rs_care, rs_prdnn, rs_aprnn, rs_trades, rs_patchrepair, rgr_care, rgr_prdnn, rgr_aprnn, rgr_trades, rgr_patchrepair, dd_care, dd_prdnn, dd_aprnn, dd_trades, dd_patchrepair])


    with open('cifar_rq.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['', '', 'vgg19', '', '', '', '','', '', '','', '', '','resnet18', '', '', '', '', '', '',])
        writer.writerow(['', '', '4','','','','', '8', '','','', '','4', '','','', '','8', '',''])
        writer.writerow(['', '', '50', '100', '200', '500', '1000', '50', '100', '200', '500', '1000', '50', '100', '200', '500', '1000', '50', '100', '200', '500', '1000'])



        for item in ['rsr', 'rgr', 'drawdown']:
            for tool in [ 'care', 'prdnn',  'trades', 'aprnn','ours']:
                if item == 'rsr':
                    row = ['RSR/%']
                elif item == 'rgr':
                    row = ['RGR/%']
                elif item == 'drawdown':
                    row = ['DD/%']
                row.append(tool)
                for net in ['vgg19', 'resnet18']:
                    for radius in [4, 8]:
                        radius = str(radius)
                        for data_number in [50,100,200,500,1000]:
                            data_number = str(data_number)
                            if tool == 'ours':
                                value = patchrepair_result[net][radius][data_number][item] + '[]'
                            # elif tool == 'label':
                            #     value = patchrepair_label_result[net][radius][data_number][item]
                            elif tool == 'care':
                                value = care_result[net][radius][data_number][item] + '[]'
                            elif tool == 'prdnn':
                                value = prdnn_result[net][radius][data_number][item] + '[]'
                            elif tool == 'aprnn':
                                value = aprnn_result[net][radius][data_number][item] + '[]'
                            elif tool == 'trades':
                                value = trades_result[net][radius][data_number][item] + '[]'
                            else:
                                value = generalization_r[net][radius][data_number][tool][item]+ '[]'
                            row.append(value)
                writer.writerow(row)
        for item in [ 'dsr']:
            
            for tool in [ 'care', 'prdnn',  'aprnn',  'trades', 'ours']:
                row = ['DSR/%']
                row.append(tool)
                for net in ['vgg19', 'resnet18']:
                    for radius in [4, 8]:
                        radius = str(radius)
                        for data_number in [50,100,200,500,1000]:
                            data_number = str(data_number)
                            if tool == 'ours':
                                value = pgd_r[net][radius][data_number]['patch'][item]+ '[]'
                            elif tool == 'trades':
                                value = pgd_r[net][radius][data_number][tool][item]+ '[]'
                            else:
                                value = pgd_r[net][radius][data_number][tool][item]+ '[]'
                            # elif tool == 'label':
                            row.append(value)
                writer.writerow(row)
        for item in [ 'dsgr']:
            
            for tool in [ 'care', 'prdnn',  'aprnn',  'trades', 'ours']:
                row = ['DGSR/%']
                row.append(tool)
                for net in ['vgg19', 'resnet18']:
                    for radius in [4, 8]:
                        radius = str(radius)
                        for data_number in [50,100,200,500,1000]:
                            data_number = str(data_number)
                            if tool == 'ours':
                                value = generalization_r[net][radius][data_number]['PatchRepair'][item] + '[]'
                            else:
                                value = generalization_r[net][radius][data_number][tool][item]+ '[]'
                            # elif tool == 'label':
                            row.append(value)
                writer.writerow(row)
        for item in ['time']:
            
            for tool in [ 'care', 'prdnn',  'trades', 'aprnn','ours']:
                row = ['Time/s']
                row.append(tool)
                for net in ['vgg19', 'resnet18']:
                    for radius in [4, 8]:
                        radius = str(radius)
                        for data_number in [50,100,200,500,1000]:
                            data_number = str(data_number)
                            if tool == 'ours':
                                value = patchrepair_result[net][radius][data_number][item] + '[]'
                            # elif tool == 'label':
                            #     value = patchrepair_label_result[net][radius][data_number][item]
                            elif tool == 'care':
                                value = care_result[net][radius][data_number][item]+ '[]'
                            elif tool == 'prdnn':
                                value = prdnn_result[net][radius][data_number][item]+ '[]'
                            elif tool == 'aprnn':
                                value = aprnn_result[net][radius][data_number][item]+ '[]'
                            elif tool == 'trades':
                                value = trades_result[net][radius][data_number][item]+ '[]'
                            else:
                                value = generalization_r[net][radius][data_number][tool][item]+ '[]'
                            row.append(value)
                writer.writerow(row) 









        # writer.writerow(['', 'care'])
        # writer.writerow(['RSR/%', 'prdnn'])
        # writer.writerow(['', 'aprnn'])
        # writer.writerow(['', 'trade'])
        # writer.writerow(['', 'ours'])
        # writer.writerow(['', 'care'])
        # writer.writerow(['RGR/%', 'prdnn'])
        # writer.writerow(['', 'aprnn'])
        # writer.writerow(['', 'trade'])
        # writer.writerow(['', 'ours'])
        # writer.writerow(['', 'care'])
        # writer.writerow(['DD/%', 'prdnn'])
        # writer.writerow(['', 'aprnn'])
        # writer.writerow(['', 'trade'])
        # writer.writerow(['', 'ours'])
        # writer.writerow(['', 'care'])
        # writer.writerow(['DSR/%', 'prdnn'])
        # writer.writerow(['', 'aprnn'])
        # writer.writerow(['', 'trade'])
        # writer.writerow(['', 'ours'])
        # writer.writerow(['', 'care'])
        # writer.writerow(['DGSR/%', 'prdnn'])
        # writer.writerow(['', 'aprnn'])
        # writer.writerow(['', 'trade'])
        # writer.writerow(['', 'ours'])
        # writer.writerow(['', 'care'])
        # writer.writerow(['Time/s', 'prdnn'])
        # writer.writerow(['', 'aprnn'])
        # writer.writerow(['', 'trade'])
        # writer.writerow(['', 'ours'])


if __name__ == '__main__':
    # patchrepair_result = read_patchrepair_result()
    # prdnn_result = read_prdnn_result()
    # care_result = read_care_result()
    # aprnn_result = read_aprnn_result()
    # pgd_result()
    # generalizaiton_result()

    # patch_repair_label_result = read_patchrepair_label_result()
    # convert_dict_to_csv()
    convert_dict_to_csv_rq()
