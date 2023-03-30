import train
import test
import os
from shutil import copy

if __name__ == '__main__':
    file_dir = '.'
    _exp_name = 'boss_new'

    if os.path.split(os.getcwd())[-1] == _exp_name:
        train._exp_name = _exp_name
        test._exp_name = _exp_name
        train.main(**train.parse_args())
        test.main(**test.parse_args())
        exit(0)
    else:
        path = os.path.join(file_dir,_exp_name)
        if not os.path.exists(path):
            os.mkdir(path)

        for f in os.listdir(file_dir):
            if os.path.isfile(os.path.join(file_dir,f)) and f.endswith('.py'):
                copy(os.path.join(file_dir,f),os.path.join(path,f))

        os.chdir(path)
        os.system('python main.py')