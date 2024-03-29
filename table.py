import csv

# use the pandas or csv library to generate a table like this:
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#    Model   |  num_p  |             RSR on repair set/%                                       RGR/%                                            DD/%                                         
#            |         -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#            |         | CARE   PRDNN  APRNN  REASSURE  TRADES  ART Ours | CARE   PRDNN  APRNN  REASSURE  TRADES  ART Ours | CARE   PRDNN  APRNN  REASSURE TRADES ART  Ours  |  
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FNN_small  |   50    | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 |
#            |   100   | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 |
#            |   200   | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 |
#            |   500   | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 |
#            |  1000   | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 |
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FNN_big    |   50    | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 |
#            |   100   | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 |
#            |   200   | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 |
#            |   500   | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 |
#            |  1000   | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 |
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CNN_small  |   50    | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 |
#            |   100   | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 |
#            |   200   | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 |
#            |   500   | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 |
#            |  1000   | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 |
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# avg                  | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 | 0.000  0.000  0.000  0.000  0.000  0.000  0.000 |
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

with open('table.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['Model', 'num_p', 'RSR on repair set/%', 'RGR/%', 'DD/%'])
    writer.writerow(['', '', 'CARE', 'PRDNN', 'APRNN', 'REASSURE', 'TRADES', 'ART', 'Ours', 'CARE', 'PRDNN', 'APRNN', 'REASSURE', 'TRADES', 'ART', 'Ours', 'CARE', 'PRDNN', 'APRNN', 'REASSURE', 'TRADES', 'ART', 'Ours'])
    writer.writerow(['FNN_small', '50', 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 ])
    writer.writerow(['', '100', 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  0.000, 0.000,  0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000,0.000 ])
    writer.writerow(['', '200', 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000,0.000 ])
    writer.writerow(['', '500', 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000,0.000 ])
    writer.writerow(['', '1000', 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  0.000, 0.000,  0.000, 0.000,  0.000, 0.000, 0.000,  0.000, 0.000,  0.000, 0.000,  0.000, 0.000, 0.000, 0.000,0.000 ])
    writer.writerow(['FNN_big', '50', 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 ])
    writer.writerow(['', '100', 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  0.000, 0.000,  0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000,0.000 ])
    writer.writerow(['', '200', 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000,0.000 ])
    writer.writerow(['', '500', 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000,0.000 ])
    writer.writerow(['', '1000', 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  0.000, 0.000,  0.000, 0.000,  0.000, 0.000, 0.000,  0.000, 0.000,  0.000, 0.000,  0.000, 0.000, 0.000, 0.000,0.000 ])
    writer.writerow(['CNN_small', '50', 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 ])
    writer.writerow(['', '100', 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  0.000, 0.000,  0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000,0.000 ])
    writer.writerow(['', '200', 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000,0.000 ])
    writer.writerow(['', '500', 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 0.000,0.000 ])
    writer.writerow(['', '1000', 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  0.000, 0.000,  0.000, 0.000,  0.000, 0.000, 0.000,  0.000, 0.000,  0.000, 0.000,  0.000, 0.000, 0.000, 0.000,0.000 ])  
    writer.writerow(['avg', '', 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  0.000,  0.000, 0.000, 0.000,  0.000,  0.000, 0.000,  0.000, 0.000,  0.000, 0.000,  0.000,  0.000, 0.000, 0.000,0.000 ])    


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#          |           |                        vgg19                      |                      resnet18                                                           
#          |           |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#          |    tool   |              4                      8             |           4                              8 
#          |           |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#          |           | 50, 100, 200, 500, 1000   50, 100, 200, 500, 1000 | 50, 100, 200, 500, 1000      50, 100, 200, 500, 1000
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#          |   care    |
#  RSR/%   |   prdnn   |
#          |   aprnn   |
#          |   trade   |
#          |   ours    |
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#          |   care    |
#  RGR/%   |   prdnn   |
#          |   aprnn   |
#          |   trade   |
#          |   ours    |
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#          |   care    |
#  DD/%    |   prdnn  
#          |   aprnn   |
#          |   trade   |
#          |   ours    |
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#          |   care    |
#  DSR/%   |   prdnn   |
#          |   aprnn   |
#          |   trade   |
#          |   ours    |
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#          |   care    |
#  DGSR/%  |   prdnn   |
#          |   aprnn   |
#          |   trade   |
#          |   ours    |
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#          |   care    |
#  Time/s  |   prdnn   |
#          |   aprnn   |
#          |   trade   |
#          |   ours    |
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

with open('table.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['', '', 'vgg19', '', '', '', 'resnet18', '', '', '', ''])
    writer.writerow(['', '', '4', '8', '4', '8', '4', '8', '4', '8', '4', '8'])
    writer.writerow(['', '', '50', '100', '200', '500', '1000', '50', '100', '200', '500', '1000'])
    writer.writerow(['', 'care'])
    writer.writerow(['RSR/%', 'prdnn'])
    writer.writerow(['', 'aprnn'])
    writer.writerow(['', 'trade'])
    writer.writerow(['', 'ours'])
    writer.writerow(['', 'care'])
    writer.writerow(['RGR/%', 'prdnn'])
    writer.writerow(['', 'aprnn'])
    writer.writerow(['', 'trade'])
    writer.writerow(['', 'ours'])
    writer.writerow(['', 'care'])
    writer.writerow(['DD/%', 'prdnn'])
    writer.writerow(['', 'aprnn'])
    writer.writerow(['', 'trade'])
    writer.writerow(['', 'ours'])
    writer.writerow(['', 'care'])
    writer.writerow(['DSR/%', 'prdnn'])
    writer.writerow(['', 'aprnn'])
    writer.writerow(['', 'trade'])
    writer.writerow(['', 'ours'])
    writer.writerow(['', 'care'])
    writer.writerow(['DGSR/%', 'prdnn'])
    writer.writerow(['', 'aprnn'])
    writer.writerow(['', 'trade'])
    writer.writerow(['', 'ours'])
    writer.writerow(['', 'care'])
    writer.writerow(['Time/s', 'prdnn'])
    writer.writerow(['', 'aprnn'])
    writer.writerow(['', 'trade'])
    writer.writerow(['', 'ours'])


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  Model   |                 RSR/%               |                RGR/%            |               FDD/%            |              time/s            | 
#          |   CARE   PRDNN  REASS  ART  Ours    |  CARE   PRDNN  REASS  ART  Ours | CARE   PRDNN  REASS  ART  Ours | CARE   PRDNN  REASS  ART  Ours |
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# N_{2,1}  |   0.0    0.0    0.0   0.0   0.0     |  0.0    0.0    0.0   0.0   0.0  |  0.0    0.0    0.0   0.0   0.0 |  0.0    0.0    0.0   0.0   0.0 |
# ...
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  Avg     |   0.0    0.0    0.0   0.0   0.0     |  0.0    0.0    0.0   0.0   0.0  |  0.0    0.0    0.0   0.0   0.0 |  0.0    0.0    0.0   0.0   0.0 |
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

with open('table.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['Model', 'RSR/%', 'RGR/%', 'FDD/%', 'time/s'])
    writer.writerow(['', 'CARE', 'PRDNN', 'REASS', 'ART', 'Ours', 'CARE', 'PRDNN', 'REASS', 'ART', 'Ours', 'CARE', 'PRDNN', 'REASS', 'ART', 'Ours', 'CARE', 'PRDNN', 'REASS', 'ART', 'Ours'])
    writer.writerow(['N_{2,1}', 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0])
    writer.writerow(['Avg', 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0])