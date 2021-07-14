import os
import itertools


def get_available_gpu(phys_machs, req_mem):
    gpu_available = []
    for mach in phys_machs:
        res = os.popen(f"ssh {mach} -T 'nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader'")
        gpu_free_memory = res.readlines()
        for i, gpu in enumerate(gpu_free_memory):
            if int(gpu) > req_mem:
                gpu_available.append((mach,  i, int(gpu)))
    # Sort by free memory in descending order.
    gpu_available.sort(key=lambda x: -x[2])
    return iter(gpu_available)

def main():

    # List your physical machines.
    phys_machs = ['8014', ]
    # List the memory you needed.
    req_mem = 10000
    gpu_available = get_available_gpu(phys_machs, req_mem)
    # E.g., next(gpu_available) = ('8020', 0, 12196)
    # List anaconda_env name on your machine
    mach_info = {
      '8020': {'conda': 'py3.7',     'code': '~/data/Github/xdomain-dep-parser'},
      '8014': {'conda': 'py3.6',     'code': '~/data/Github/xdomain-dep-parser'},
      '8037': {'conda': 'py36-pt18', 'code': '~/data_from_8014/Github/xdomain-dep-parser'},
      '8017': {'conda': 'py36-pt18', 'code': '~/data_from_8014/Github/xdomain-dep-parser'},
      '63':   {'conda': 'py36-pt18', 'code': '~/data/xdomain-dep-parser'},
      '64':   {'conda': 'py36-pt17', 'code': '~/data/xdomain-dep-parser'},
    }
    default_cfg ='../cfgs/default.cfg'
    # List the CMD arguments to explore.
    argu_list = {
        'DOMAIN': ['ZX',],  # 'PC', 'PB', 'ZX', 'FIN', 'LEG'
        # 'D_MODEL': ['400',],
        # 'LR_DECAY': ['0.8', '0.7'],
        # 'LR_ANNEAL': ['15000',],
        # 'LR_DOUBLE': ['75400',],
        # 'XFMR_ATTN_DROP': ['0.4',],
        # 'XFMR_FFN_DROP': ['0.4',],
        # 'XFMR_RES_DROP': ['0.4',],
        # 'MIN_PROB': ['0.5', ],
        'D_CHAR': ['50',],
        'D_TAG': ['50',],
        'N_GNN_LAYER': ['1',],
        # 'LR': ['0.0012',],
        # 'LR_DECAY': ['0.8',],
        # 'LR_WARM': ['800',],
        # 'LR_DOUBLE': ['20400',],
        # 'N_EPOCH': ['1',],
        # 'DEBUG': [''],
    }
    argu_comb = list(itertools.product(*argu_list.values()))

    # Create configuration file
    data_dir = {}
    argu_comb = [dict(zip(argu_list.keys(), values)) for values in argu_comb]
    for argu in argu_comb:
        ckpt_name = '_'.join([k+v for k, v in argu.items()])
        mach, gpu, _ = next(gpu_available)
        cmd_args = ' '.join([f'--{k} {v}'for k, v in argu.items()])
        cmd_args += f' --exp_name {ckpt_name} --CFG ../ckpts/{ckpt_name}/run.cfg'
        # if mach == '8037':
        #     cmd_run = f"rsync -avz -e 'ssh' ../ {'8014'}:{mach_info['8014']['code']}"
        # else:
        cmd_run = f"rsync -avz -e 'ssh' ../ {mach}:{mach_info[mach]['code']}"
        res = os.system(cmd_run)
        print(f"rsync executed {'successfully' if res==0 else 'failed'}")

        cmd_run = (f"ssh {mach} 'bash -s' -T < run.sh "
                   f"{ckpt_name} {mach_info[mach]['conda']} {gpu} {default_cfg} '{cmd_args}'")
        res = os.system(cmd_run)


if __name__ == '__main__':
    main()


