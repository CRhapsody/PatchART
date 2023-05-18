import argparse

parser = argparse.ArgumentParser()
parser.add_argument('echo', default= 'echo default',
                    help="echo the string you use here")
parser.add_argument('square', default = 3,
                    help="display a square of a given number", type = int)
parser.add_argument('-v','--verbosity',action="count")
args = parser.parse_args()
# print(args.echo)
print(args.square**2)

