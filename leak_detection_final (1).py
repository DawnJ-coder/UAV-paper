
# -*- coding: utf-8 -*-
"""
leak_detection_final.py

超声气体泄漏检测程序
功能:
1. 读取波束形成wav
2. 20-70kHz频谱分析
3. 40cm背景扣除
4. 距离衰减拟合 E=A*r^-n
5. 多特征融合判断泄漏
"""

import os
import glob
import re
import csv
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# ================= 参数 =================

time_folders = [
    "HM20260626_142938.ld",
    "HM20260626_143034.ld",
    "HM20260626_144226.ld",
    "HM20260626_144325.ld"
]

center_root_dir = r"D:\gas\beamform_results"
offset_root_dir = r"D:\gas\beamform_results_offset_multiple"

distances = np.array([5,10,15,20,25,30,35,40], dtype=float)

directions = [
    "up","down","left","right",
    "up_left","down_left",
    "up_right","down_right"
]

FREQ_LOW = 20000
FREQ_HIGH = 70000
NFFT = 4096

CONF_LEAK = 0.60


# ================= 音频 =================

def read_wav(path):
    if not os.path.exists(path):
        return None,None,None
    try:
        fs,x = wav.read(path)
        if len(x.shape)>1:
            x=x[:,0]
        x=x.astype(float)
        rms=np.sqrt(np.mean(x*x))+1e-12
        return fs,x,rms
    except:
        return None,None,None


def compute_psd(path, norm=None):
    fs,x,rms=read_wav(path)
    if x is None:
        return None,None

    if norm is None:
        norm=rms

    x=x/(norm+1e-12)

    f,p=signal.welch(
        x,
        fs=fs,
        nperseg=NFFT,
        scaling="density"
    )

    mask=(f>=FREQ_LOW)&(f<=FREQ_HIGH)

    return f[mask],p[mask]


def energy(f,p):
    if f is None:
        return 0
    return np.trapz(p,f)


def hf_ratio(f,p):
    if f is None:
        return 0
    a=(f>=20000)&(f<=70000)
    b=(f>=40000)&(f<=70000)
    return np.trapz(p[b],f[b])/(np.trapz(p[a],f[a])+1e-12)


def remove_background(p,bg):
    n=min(len(p),len(bg))
    return np.maximum(p[:n]-bg[:n],0)


# ================= 物理模型 =================

def decay_model(r,A,n):
    return A*r**(-n)


def decay_score(e):
    r=distances[:len(e)]
    e=np.array(e)

    mask=e>0
    r=r[mask]
    e=e[mask]

    if len(r)<3:
        return 0,0,0

    try:
        popt,_=curve_fit(
            decay_model,
            r,
            e,
            p0=[e[0]*25,2],
            bounds=([0,0],[np.inf,5])
        )

        A,n=popt
        pred=decay_model(r,A,n)

        ss1=np.sum((e-pred)**2)
        ss2=np.sum((e-np.mean(e))**2)

        r2=max(0,1-ss1/(ss2+1e-12))

        score=r2 if 0.5<n<3.5 else r2*0.5

        return score,n,r2

    except:
        return 0,0,0



# ================= 文件 =================

def center_ids(folder):
    files=glob.glob(
        os.path.join(folder,"*_beamform_result.wav")
    )

    ids=[]

    for f in files:
        m=re.search(r"_(\d+)_beamform_result",os.path.basename(f))
        if m:
            ids.append(m.group(1))

    return sorted(set(ids))


def find_center(folder,cid):
    a=glob.glob(
        os.path.join(folder,f"*_{cid}_beamform_result.wav")
    )
    return a[0] if a else None


def find_offset(folder,cid,d,dir):
    a=glob.glob(
        os.path.join(folder,f"*_{cid}d{int(d)}_{dir}*.wav")
    )
    return a[0] if a else None



# ================= 主检测 =================

def process(folder):

    cdir=os.path.join(center_root_dir,folder)
    odir=os.path.join(offset_root_dir,folder)

    if not os.path.exists(cdir):
        return []

    results=[]

    for cid in center_ids(cdir):

        cf=find_center(cdir,cid)

        if cf is None:
            continue

        _,_,rms=read_wav(cf)

        matrix={}
        best_spec=None
        best_bg=None

        for d in directions:

            bgfile=find_offset(odir,cid,40,d)

            if bgfile is None:
                continue

            _,bg=compute_psd(bgfile,rms)

            es=[]
            last=None

            for dis in distances:

                f=find_offset(odir,cid,dis,d)

                if f is None:
                    es.append(0)
                    continue

                ff,p=compute_psd(f,rms)

                net=remove_background(p,bg)

                es.append(energy(ff[:len(net)],net))

                if dis==5:
                    last=net

            matrix[d]=es

            if best_spec is None or energy(ff[:len(last)],last)>energy(ff[:len(best_spec)],best_spec):
                best_spec=last
                best_bg=bg

        if not matrix:
            continue

        best_dir=max(
            matrix,
            key=lambda x:max(matrix[x])
        )

        ds,n,r2=decay_score(matrix[best_dir])

        conf=(
            0.45*ds+
            0.35*min(hf_ratio(ff,best_spec)/0.6,1)+
            0.20*r2
        )

        result="TRUE LEAK" if conf>=CONF_LEAK else "FALSE LEAK"

        results.append([
            folder,
            cid,
            result,
            round(float(conf),3),
            best_dir,
            round(float(n),2),
            round(float(r2),3)
        ])

    return results



def main():

    all_results=[]

    for f in time_folders:
        print("processing:",f)
        all_results.extend(process(f))

    with open("leak_result.csv","w",newline="",encoding="utf-8-sig") as fp:
        writer=csv.writer(fp)
        writer.writerow([
            "time","center","result",
            "confidence","direction",
            "n","R2"
        ])
        writer.writerows(all_results)

    print("完成，结果保存 leak_result.csv")


if __name__=="__main__":
    main()
