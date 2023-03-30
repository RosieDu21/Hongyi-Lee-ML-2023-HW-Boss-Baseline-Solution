import os
import csv

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), 'DATA')
    old_file_prefix = os.path.join(data_dir,'train_dev.clean.')
    pred_file_dir = ''
    new_file_prefix = os.path.join(data_dir,'train_dev.new.')

    src_lang = 'zh'
    tgt_lang = 'en'

    src_old = open(old_file_prefix + src_lang, 'r')
    tgt_old = open(old_file_prefix + tgt_lang, 'r')
    pred_fl = open(pred_file_dir, 'r')
    src_new = open(new_file_prefix + src_lang, 'w')
    tgt_new = open(new_file_prefix + tgt_lang, 'w')

    reader = csv.reader(pred_fl)
    for s, t in reader:
        print(s, file=src_new)
        print(t, file=tgt_new)

    src_new.writelines(src_old.readlines())
    tgt_new.writelines(tgt_old.readlines())

    src_old.close()
    tgt_old.close()
    pred_fl.close()
    src_new.close()
    tgt_new.close()
