from argparse import Namespace
def fmt_args(args: Namespace) -> str : 
    '''
    output args configuation at the beginning of log file
    '''
    title = args.stamp
    s = [f'\n===== {title} =====']
    d = vars(args)
    for k,v in d.items():
        if k == 'stamp':
            continue
        s.append(f'  {k}:{v}')
    s.append(f'===== end of {title} configuration =====')
    return '\n'.join(s)
