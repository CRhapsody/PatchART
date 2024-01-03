from pathlib import Path
import os
import json
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
                for patch_format in ['small', 'big']:
                    result_dict[net][radius][data_number] = {}
                    for item in ['drawdown', 'rs', 'gene','time']:
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
                    # get the rs
                    line = lines[-7]
                    line = line.split(' ')
                    rs = float(line[-1])
                    # get the gene
                    line = lines[-5]
                    line = line.split(' ')
                    gene = float(line[-1])
                    # get the time
                    line = lines[-1]
                    line = line.split(' ')

                    time = line[-1]
                    time = time[:-3]
                    time = float(time)

                    # record the result
                    result_dict[net][radius][data_number]['drawdown'] = drawdown
                    result_dict[net][radius][data_number]['rs'] = rs
                    result_dict[net][radius][data_number]['gene'] = gene
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
                for item in ['drawdown', 'rs', 'gene','time']:
                    result_dict[net][radius][data_number][item] = 0

    
    
    
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
                    if net == 'vgg19':
                        drawdown = VGG19 - drawdown
                    elif net == 'resnet18':
                        drawdown = RESNET18 - drawdown
                    # get the rs
                    line = lines[-6]
                    line = line.split(' ')
                    rs = float(line[-1])
                    # get the gene
                    line = lines[-5]
                    line = line.split(' ')
                    gene = float(line[-1])
                    # get the time
                    line = lines[-1]
                    line = line.split(' ')

                    time = line[-1]
                    time = time[:-3]
                    time = float(time)

                    # record the result
                    result_dict[net][radius][data_number]['drawdown'] = drawdown
                    result_dict[net][radius][data_number]['rs'] = rs
                    result_dict[net][radius][data_number]['gene'] = gene
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
                for item in ['drawdown', 'rs', 'gene','time','defense success of apgd & apgdt']:
                    if item != 'defense success of apgd & apgdt':
                        result_dict[net][radius][data_number][item] = 'None'
                    else:
                        if net == 'vgg19':
                            # full 0
                            result_dict[net][radius][data_number][item] = 0.0
                        elif net == 'resnet18':
                            result_dict[net][radius][data_number][item] = 0
    
    
    
    
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
                    # get the rs
                    line = lines[1]
                    line = line.split(' ')
                    rs = 1. - float(line[-3][:-1])
                    # get the gene
                    line = lines[2]
                    line = line.split(' ')
                    gene = 1. - float(line[-3][:-1])
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

                    # record the result
                    result_dict[net][radius][data_number]['drawdown'] = drawdown
                    result_dict[net][radius][data_number]['rs'] = rs
                    result_dict[net][radius][data_number]['gene'] = gene
                    result_dict[net][radius][data_number]['time'] = time

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
                for item in ['drawdown', 'rs', 'gene','time']:
                    if item != 'time':
                        result_dict[net][radius][data_number][item] = 'None'
                    else:
                        result_dict[net][radius][data_number][item] = 'TIMEOUT'
                # full 0
                result_dict[net][radius][data_number]['defense success of apgd & apgdt'] = 0
    
    
    
    
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
                            result_dict[net][radius][data_number]['drawdown'] = drawdown
                            continue
                        if line.startswith('rsr after repair:'):
                            line = line.split(' ')
                            line = line[-1].split(':')
                            rs = float(line[-1])
                            result_dict[net][radius][data_number]['rs'] = rs
                            continue
                        if line.startswith('gene after repair:'):
                            line = line.split(' ')
                            line = line[-2].split(':')
                            gene = float(line[-1])
                            result_dict[net][radius][data_number]['gene'] = gene
                            continue
                        if line.startswith('time cost:'):
                            line = line.split(' ')
                            line = line[-1].split(':')
                            time = float(line[-1][:-2])
                            result_dict[net][radius][data_number]['time'] = time
                            continue
    for net in ['vgg19', 'resnet18']:
        for radius in [4, 8]:
            radius = str(radius)
            for data_number in [50,100,200,500,1000]:
                data_number = str(data_number)
                if result_dict[net][radius][data_number]['time'] == 'TIMEOUT':
                    result_dict[net][radius][data_number]['defense success of apgd & apgdt'] = 'None'
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
#                         result_dict[net][radius][data_number][item] = 'None'
#                     else:
#                         result_dict[net][radius][data_number][item] = 'TIMEOUT'
    
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
                for item in ['drawdown', 'rs', 'gene','time','defense success of apgd & apgdt']:
                    if item != 'time':
                        result_dict[net][radius][data_number][item] = 'None'
                    else:
                        result_dict[net][radius][data_number][item] = 'TIMEOUT'



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
                            result_dict[net][radius][data_number]['drawdown'] = drawdown
                            continue
                        if line.startswith('rsr after repair:'):
                            line = line.split(' ')
                            line = line[-1].split(':')
                            rs = float(line[-1])
                            result_dict[net][radius][data_number]['rs'] = rs
                            continue
                        if line.startswith('gene after repair:'):
                            line = line.split(' ')
                            line = line[-2].split(':')
                            gene = float(line[-1])
                            result_dict[net][radius][data_number]['gene'] = gene
                            continue


                        if line.startswith('gene of pgd:'):
                            line = line.split(' ')
                            line = line[-2].split(':')
                            pgd = float(line[-1][:5])
                            result_dict[net][radius][data_number]['defense success of apgd & apgdt'] = pgd
                            continue
                        if line.startswith('time cost:'):
                            line = line.split(' ')
                            line = line[-1].split(':')
                            time = float(line[-1][:-2])
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
    header = ['net', 'radius', 'data_number', 'tool', 'drawdown', 'rs', 'gene', 'time']
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
                    for tool in ['patch', 'ori', 'adv-train', 'label', 'care', 'prdnn', 'aprnn']:
                        pgd = pgd_r[net][radius][data_number][tool]['defense success of apgd & apgdt']
                        row = [net, radius, data_number, tool, pgd]
                        writer.writerow(row)
        # # write the aprnn result
        # for net in ['vgg19', 'resnet18']:
        #     for radius in [4, 8]:
        #         radius = str(radius)
        #         for data_number in [50,100,200,500,1000]:
        #             data_number = str(data_number)
        #             pgd = aprnn_result[net][radius][data_number]['defense success of apgd & apgdt']
        #             row = [net, radius, data_number, 'aprnn', pgd]
        #             writer.writerow(row)

    # write another csv file for generalization result
    # header = ['net', 'radius', 'data_number', 'tool', 'defense success of apgd & apgdt']
    # generalization_r = generalizaiton_result()
    # with open('cifar_generalization_result.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(header)

    #     for net in ['FNN_small', 'FNN_big', 'CNN_small']:
    #         for radius in [4, 8]:
    #             radius = str(radius)
    #             for data_number in [50,100,200,500,1000]:
    #                 data_number = str(data_number)
    #                 for tool in ['PatchRepair', 'original', 'adv_training', 'art', 'care', 'prdnn', 'reassure', 'aprnn']:
    #                     defense = generalization_r[net][radius][data_number][tool]['defense success of apgd & apgdt']
    #                     row = [net, radius, data_number, tool, defense]
    #                     writer.writerow(row)


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
#                         result_dict[net][radius][data_number][item] = 'None'
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
                for tool in ['patch', 'ori', 'adv-train', 'label', 'care', 'prdnn', 'aprnn']:
                    result_dict[net][radius][data_number][tool] = {}
                    for item in ['defense success of apgd & apgdt']:
                        if tool != 'care' and tool != 'prdnn' and tool != 'aprnn':
                            result_dict[net][radius][data_number][tool][item] = 'None'
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
                pgd = a[1][:-1]
                pgd = float(pgd)/float(data_number)
                result_dict[net][radius][data_number][tool]['defense success of apgd & apgdt'] = pgd

    # save the result to json file
    with open('pgd_cifar_result.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    return result_dict

# def generalizaiton_result():
#     pgd_result_file = '/home/chizm/PatchART/results/mnist/repair/generalization/compare/compare_generalization.txt'
#     result_dict = {}
#     for net in ['FNN_small', 'FNN_big', 'CNN_small']:
#         result_dict[net] = {}
#         for radius in [4, 8]:
#             radius = str(radius)
#             result_dict[net][radius] = {}
#             for data_number in [50,100,200,500,1000]:
#                 data_number = str(data_number)
#                 result_dict[net][radius][data_number] = {}
#                 for tool in ['PatchRepair', 'label', 'original', 'adv_training', 'art', 'care', 'prdnn', 'reassure', 'aprnn']:
#                     result_dict[net][radius][data_number][tool] = {}
#                     for item in ['defense success of apgd & apgdt']:
#                         result_dict[net][radius][data_number][tool][item] = 'None'
#     with open(pgd_result_file, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             if line == '\n':
#                 continue
#             line = line.split(' ')
#             net = line[2][:-1]
#             radius = line[4][:-1]
#             data_number = line[6][:-1]

#             line = line[7:]
#             if line[-1] == '\n':
#                 line = line[:-1]
#             for a in line:
#                 a = a.split(':')
#                 tool = a[0]
#                 pgd = a[1][:-1]
#                 pgd = float(pgd)
#                 result_dict[net][radius][data_number][tool]['defense success of apgd & apgdt'] = pgd

#     # save the result to json file
#     with open('pgd_generlization_mnist_result.json', 'w') as f:
#         json.dump(result_dict, f, indent=4)
#     return result_dict    




if __name__ == '__main__':
    # patchrepair_result = read_patchrepair_result()
    # prdnn_result = read_prdnn_result()
    # care_result = read_care_result()
    # aprnn_result = read_aprnn_result()

    pgd_result()
    # patch_repair_label_result = read_patchrepair_label_result()
    # convert_dict_to_csv()
