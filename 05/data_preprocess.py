import os
import sentencepiece as sp

import re

def strQ2B(ustring):
    """Full width -> half width"""
    # reference:https://ithelp.ithome.com.tw/articles/10233122
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # Full width space: direct conversion
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # Full width chars (except space) conversion
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)


def clean_s(s, lang):
    if lang == 'en':
        s = re.sub(r"\([^()]*\)", "", s)  # remove ([text])
        s = s.replace('-', '')  # remove '-'
        s = re.sub('([.,;!?()\"])', r' \1 ', s)  # keep punctuation
    elif lang == 'zh':
        s = strQ2B(s)  # Q2B
        s = re.sub(r"\([^()]*\)", "", s)  # remove ([text])
        s = s.replace(' ', '')
        s = s.replace('—', '')
        s = s.replace('“', '"')
        s = s.replace('”', '"')
        s = s.replace('_', '')
        s = re.sub('([。,;!?()\"~「」])', r' \1 ', s)  # keep punctuation
    s = ' '.join(s.strip().split())
    return s


def len_s(s, lang):
    if lang == 'zh':
        return len(s)
    return len(s.split())


def clean_corpus_pair(prefix, l1, l2, ratio=9, max_len=1000, min_len=1):
    if os.path.isfile(f'{prefix}.clean.{l1}') and os.path.isfile(f'{prefix}.clean.{l2}'):
        print(f'{prefix}.clean.{l1} & {l2} exists. skipping clean.')
        return

    l1_in_f = open(f'{prefix}.raw.{l1}', 'r')
    l2_in_f = open(f'{prefix}.raw.{l2}', 'r')
    l1_out_f = open(f'{prefix}.clean.{l1}', 'w')
    l2_out_f = open(f'{prefix}.clean.{l2}', 'w')

    for s1 in l1_in_f:
        s1 = s1.strip()
        s2 = l2_in_f.readline().strip()
        s1 = clean_s(s1, l1)
        s2 = clean_s(s2, l2)
        s1_len = len_s(s1, l1)
        s2_len = len_s(s2, l2)
        if min_len > 0: # remove short sentence
            if s1_len < min_len or s2_len < min_len:
                continue
        if max_len > 0: # remove long sentence
            if s1_len > max_len or s2_len > max_len:
                continue
        if ratio > 0: # remove by ratio of length
            if s1_len/s2_len > ratio or s2_len/s1_len > ratio:
                continue
        print(s1, file=l1_out_f)
        print(s2, file=l2_out_f)

    l1_in_f.close()
    l2_in_f.close()
    l1_out_f.close()
    l2_out_f.close()


def clean_corpus_mono(prefix, l, max_len=1000, min_len=1):
    if os.path.isfile(f'{prefix}.clean.{l}'):
        print(f'{prefix}.clean.{l} exists. skipping clean.')
        return

    in_f = open(f'{prefix}.raw.{l}', 'r')
    out_f = open(f'{prefix}.clean.{l}', 'w')

    for s in in_f:
        s = s.strip()
        s = clean_s(s, l)
        s_len = len_s(s, l)
        if min_len > 0 and s_len < min_len:
            continue
        if max_len > 0 and s_len > max_len:
            continue
        print(s, file=out_f)

    in_f.close()
    out_f.close()


if __name__ == '__main__':
    vocab = 10000
    data_dir = './DATA'
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    url = {
        'train.tgz': "https://github.com/figisiwirf/ml2023-hw5-dataset/releases/download/v1.0.1/ml2023.hw5.data.tgz",
        'test.tgz': "https://github.com/figisiwirf/ml2023-hw5-dataset/releases/download/v1.0.1/ml2023.hw5.test.tgz",
        'mono.gz': "https://github.com/figisiwirf/ml2023-hw5-dataset/releases/download/v1.0.1/ted_zh_corpus.deduped.gz"
    }

    rename = {
        'mono' : 'mono_test.raw.zh',
        'raw.en' : 'train_dev.raw.en',
        'raw.zh' : 'train_dev.raw.zh',
        'test.en' : 'test.raw.en',
        'test.zh' : 'test.raw.zh'
    }

    for f, u in url.items():
        path = os.path.join(data_dir, f)
        if os.path.isfile(path):
            print(f"Data package {f} exists, downloading skipped.")
        else:
            os.system(f'wget {u} -O {path}')
        prefix, suffix = path.rsplit('.',1)
        if suffix == "tgz":
            os.system(f'tar -xvf {path} --skip-old-files -C {data_dir}')
        elif suffix == "zip":
            os.system(f'unzip -n {path} -d {data_dir}')
        elif suffix == "gz":
            os.system(f"gzip -kd {path}")

    for s, t in rename.items():
        if os.path.isfile(os.path.join(data_dir,s)):
            os.rename(os.path.join(data_dir,s),os.path.join(data_dir,t))


    print('Dataclean start...')
    clean_corpus_pair(os.path.join(data_dir,'train_dev'),'en','zh')
    clean_corpus_pair(os.path.join(data_dir,'test'),'en','zh')
    clean_corpus_mono(os.path.join(data_dir,'mono_test'),'zh')
    print('Dataclean finished.')

    if os.path.isfile(os.path.join(data_dir, f'sp{vocab}.model')):
        print(f"SentencePiece model exist. Learning skipped.")
    else:
        print('SentencePiece model training begin...')
        lines = []
        for f in [os.path.join(data_dir, f.replace('raw', 'clean')) for f in rename.values()]:
            with open(f, 'r') as fl:
                lines += fl.readlines()
        with open(os.path.join(data_dir,'corpus.txt'), 'w') as fl:
            fl.writelines(lines)
        sp.SentencePieceTrainer.Train(
            f'--input={os.path.join(data_dir,"corpus.txt")} '
            f'--vocab_size={vocab} '
            f'--character_coverage=1.0 '
            f'--model_prefix={os.path.join(data_dir,f"sp{vocab}")} '
            f'--model_type=unigram '
            f'--shuffle_input_sentence=true '
            f'--normalization_rule_name=nmt_nfkc_cf '
            f'--pad_id=0 '
            f'--bos_id=1 '
            f'--eos_id=2 '
            f'--unk_id=3 '
        )
        os.remove(os.path.join(data_dir,'corpus.txt'))