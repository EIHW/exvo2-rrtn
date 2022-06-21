import os
from dataclasses import dataclass, asdict


def get_path_prefix():
    machine = os.uname()[1]
    prefix = ''
    local = False
    if machine == '[YOUR MACHINE NAME]':
        prefix = '[YOUR LOCAL PATH PREFIX]'
        local = True
    return prefix, local


def get_exvo_path(root):
    path = {
        'data_root': './0_ExVo22/baseline',
        'features_path': './features.csv',
    }
    return path[root]


@dataclass
class TrainConfig:
    # basic
    machine, local = get_path_prefix()
    data_root: str = machine + get_exvo_path('data_root')
    features_path: str = machine + get_exvo_path('features_path')
    md_name: str = 'exvo'
    random_seed: int = 10
    # hyper
    lr: float = 0.001
    tr_bs: int = 2 if local else 128
    val_bs: int = 2 if local else 1
    tr_sm: int = 6 if local else 20000
    va_sm: int = 6 if local else 20000
    aug: bool = False
    # control
    save2: str = '[PATH]'
    save_every: int = 10000 if local else 1
    val_every: int = 1
    epoches: int = 150
    log_freq: int = 200
    # ...
    resume: bool = False
    md_path: str = save2
    md_name: str = 'model.pth'

    def asdic(self):
        return asdict(self)
