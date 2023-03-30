import os

if __name__=='__main__':
    os.chdir(os.path.dirname(__file__))
    os.system(r'python data_preprocess.py')
    os.system(r'python ./zh-en/main.py')
    os.system(r'python generate_dataset.py')
    os.system(r'python ./en-zh/main.py')