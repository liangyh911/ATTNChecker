import math
import numpy as np
import os

def get_outer_time(file):
    fp = open(file, 'r')
    Lines = fp.readlines()
    
    cnt = 0
    res = 0
    for idx, line in enumerate(Lines):
        tmp = float(line)
        res += tmp
        cnt += 1
    return res / cnt

def prepare_time_attn(file):
    fp = open(file, 'r')
    Lines = fp.readlines()

    if len(Lines) == 0:
        return 0
    else:
        mod = 18 if len(Lines) / 18 == 12 else 12
        # print(mod)
        cnt = 0
        time = 0
        # res = []
        res = 0
        for idx, line in enumerate(Lines):
            tmp = float(line)
            time += tmp
            if(idx % mod == 11):
                # res.append(time)
                res += time
                time = 0
                cnt += 1
        
        return res / cnt

def prepare_time_attn_12(file):
    fp = open(file, 'r')
    Lines = fp.readlines()

    cnt = 0
    time = 0
    # res = []
    res = 0
    for idx, line in enumerate(Lines):
        tmp = float(line)
        time += tmp
        if(idx % 12 == 11):
            # res.append(time)
            res += time
            time = 0
            cnt += 1
    
    return res / cnt

def prepare_time_attn_18(file):
    fp = open(file, 'r')
    Lines = fp.readlines()

    cnt = 0
    time = 0
    # res = []
    res = 0
    for idx, line in enumerate(Lines):
        tmp = float(line)
        time += tmp
        if(idx % 18 == 17):
            # res.append(time)
            res += time
            time = 0
            cnt += 1
    
    return res / cnt


def main():
    Attn = "./records/time/attn.txt"
    prepartion = "./records/time/preparation.txt"

    AttnTime = get_outer_time(Attn)*1000 - (prepare_time_attn(prepartion))

    print("Attention Mechanism Running Time: ", AttnTime)


if __name__=="__main__":
    # os.getcwd()
    main() 