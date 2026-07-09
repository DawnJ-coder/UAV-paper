# -*- coding: utf-8 -*-
"""
leak_v8_directed_wideband_classifier.py

v8: v7 + 方向性宽频特征。
目标：把人工观察到的“真泄漏宽频集中在少数方向，假泄漏各方向更弥散”量化出来，重点改善 HM20260626_144226.ld。

运行：python leak_v8_directed_wideband_classifier.py
输出：C:\\Users\\jiangxinru6\\Desktop\\wurenji\\leak_v8_directed_wideband_results
"""
import os, re, json, math, warnings
from datetime import datetime
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\jiangxinru6\Desktop\wurenji"
OUT_DIR = os.path.join(BASE_DIR, "leak_v8_directed_wideband_results")
TARGET_TIME = "HM20260626_144226.ld"

BASE_FEATURE_CANDIDATES = [
    os.path.join(BASE_DIR, "leak_v8_1_heatmap_shape_ablation_results", "v8_1_features_A_v7_only.csv"),
    os.path.join(BASE_DIR, "leak_v7_robust_feature_results", "v7_robust_feature_dataset.csv"),
    os.path.join(BASE_DIR, "leak_v4_compare_results", "merged_feature_dataset.csv"),
]
TRUE_DETAIL_CSV = os.path.join(BASE_DIR, "leak_feature_v3_results", "energy_matrix_detail.csv")
FALSE_DETAIL_CSV = os.path.join(BASE_DIR, "leak_feature_cs_v3_results", "energy_matrix_detail.csv")

SUBBANDS = [(20,30),(30,40),(40,50),(50,60),(60,70)]
DIRS = ["up","up_right","right","down_right","down","down_left","left","up_left"]
NEAR_DISTANCE_MAX_CM = 20
BAND_ACTIVE_REL_TO_DIR_MAX = 0.18
BAND_ACTIVE_FRAC_OF_DIR_TOTAL = 0.06
DIR_ACTIVE_REL_TO_MAX = 0.25
RANDOM_STATE = 42
TH_GRID = np.linspace(0.01, 0.99, 99)
RANK_TRUE_FRACTION = 0.50
META = {"dataset","label","true_label","time","test_group","center","center_norm","experiment","row_index",
        "best_direction","energy_direction","decay_direction","representative_file","heatmap_path"}

def ensure(p): os.makedirs(p, exist_ok=True)
def find_existing(paths):
    for p in paths:
        if os.path.exists(p): return p
    return None

def norm_center(x):
    try:
        if pd.isna(x): return "00"
    except Exception: pass
    s = str(x).strip()
    if s.endswith('.0'): s = s[:-2]
    d = ''.join(ch for ch in s if ch.isdigit())
    return d.zfill(2) if d else s

def num(s): return pd.to_numeric(s, errors='coerce').replace([np.inf,-np.inf], np.nan)
def y_from_label(labels): return np.array([1 if str(x)=='TRUE_LEAK' else 0 for x in labels], dtype=int)
def lab(y): return 'TRUE_LEAK' if int(y)==1 else 'FALSE_LEAK'

def entropy_norm(x):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]; x = x[x>0]
    if len(x)<=1: return 0.0
    p = x/(x.sum()+1e-20)
    return float(-np.sum(p*np.log(p+1e-20))/np.log(len(p)))

def flatness(x):
    x = np.maximum(np.asarray(x, dtype=float), 1e-20)
    return float(np.exp(np.mean(np.log(x)))/(np.mean(x)+1e-20)) if len(x) else 0.0

def gini(x):
    x = np.maximum(np.asarray(x, dtype=float), 0); x = x[np.isfinite(x)]
    if len(x)==0 or x.sum()<=1e-20: return 0.0
    x = np.sort(x); n=len(x); c=np.cumsum(x)
    return float((n+1-2*np.sum(c)/c[-1])/n)

def cohen_d(a,b):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    a=a[np.isfinite(a)]; b=b[np.isfinite(b)]
    if len(a)<2 or len(b)<2: return np.nan
    return float((a.mean()-b.mean())/(math.sqrt((a.std(ddof=1)**2+b.std(ddof=1)**2)/2)+1e-12))

def auc_score(y, score):
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y))<2: return np.nan
        return float(roc_auc_score(y, score))
    except Exception: return np.nan

def metrics(y,p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=int)
    tp=int(((y==1)&(p==1)).sum()); tn=int(((y==0)&(p==0)).sum())
    fp=int(((y==0)&(p==1)).sum()); fn=int(((y==1)&(p==0)).sum())
    n=tp+tn+fp+fn
    rt=tp/(tp+fn+1e-12); rf=tn/(tn+fp+1e-12)
    return {"accuracy":(tp+tn)/n if n else 0, "balanced_accuracy":0.5*(rt+rf),
            "recall_TRUE":rt, "recall_FALSE":rf, "tp":tp,"tn":tn,"fp":fp,"fn":fn}

def load_base():
    p=find_existing(BASE_FEATURE_CANDIDATES)
    if not p: raise FileNotFoundError('找不到 v7/v4 特征表，请先运行 v7 或 v8.1')
    df=pd.read_csv(p)
    if 'true_label' in df.columns and 'label' not in df.columns: df=df.rename(columns={'true_label':'label'})
    if 'test_group' in df.columns and 'time' not in df.columns: df=df.rename(columns={'test_group':'time'})
    for c in ['label','time','center']:
        if c not in df.columns: raise ValueError(f'基础特征表缺少列 {c}: {p}')
    df['label']=df['label'].astype(str)
    df=df[df['label'].isin(['TRUE_LEAK','FALSE_LEAK'])].copy()
    df['center_norm']=df['center'].apply(norm_center)
    return df.reset_index(drop=True), p

def load_detail():
    if not os.path.exists(TRUE_DETAIL_CSV): raise FileNotFoundError(f'找不到 TRUE detail: {TRUE_DETAIL_CSV}')
    if not os.path.exists(FALSE_DETAIL_CSV): raise FileNotFoundError(f'找不到 FALSE detail: {FALSE_DETAIL_CSV}')
    a=pd.read_csv(TRUE_DETAIL_CSV); b=pd.read_csv(FALSE_DETAIL_CSV)
    a['label']='TRUE_LEAK'; b['label']='FALSE_LEAK'
    df=pd.concat([a,b], ignore_index=True)
    ren={}
    for c in df.columns:
        lc=c.lower()
        if lc in ['time_folder','time_name','folder','group','test_group']: ren[c]='time'
        if lc in ['center_id','center_idx','center_no','center_num']: ren[c]='center'
        if lc in ['dir','direction_name','direction_label']: ren[c]='direction'
        if lc in ['distance','dist','dist_cm','distance_cm']: ren[c]='distance_cm'
    if ren: df=df.rename(columns=ren)
    for c in ['time','center','direction']:
        if c not in df.columns:
            raise ValueError('detail表缺少列 %s。当前列：\n%s' % (c, list(df.columns)))
    if 'distance_cm' not in df.columns: df['distance_cm']=5.0
    df['center_norm']=df['center'].apply(norm_center)
    df['direction']=df['direction'].astype(str).str.strip()
    df['distance_cm']=num(df['distance_cm'])
    return df

def find_band_cols(df):
    cols=list(df.columns); band_cols={}
    for lo,hi in SUBBANDS:
        cand=[]
        for c in cols:
            lc=c.lower()
            if any(k in lc for k in ['ratio','rank','robust_z']): continue
            pats=[rf'(^|_)energy_{lo}_{hi}k?($|_)', rf'(^|_)band_{lo}_{hi}k?($|_)',
                  rf'(^|_)subband_{lo}_{hi}k?($|_)', rf'(^|_)psd_{lo}_{hi}k?($|_)',
                  rf'{lo}\s*[_-]\s*{hi}\s*k?', rf'{lo}k\s*[_-]\s*{hi}k']
            ok=any(re.search(p,lc) for p in pats)
            if not ok and str(lo) in lc and str(hi) in lc and any(k in lc for k in ['energy','band','subband','psd']): ok=True
            if ok: cand.append(c)
        if cand:
            cand=sorted(cand, key=lambda x:(0 if 'energy' in x.lower() else 1, len(x)))
            band_cols[(lo,hi)]=cand[0]
    miss=[f'{lo}-{hi}k' for lo,hi in SUBBANDS if (lo,hi) not in band_cols]
    if miss:
        raise ValueError('无法识别这些频段能量列：%s\n当前detail列名：\n%s\n请确认v3 detail保存了每个方向/距离的分频段能量。' % (', '.join(miss), list(df.columns)))
    return band_cols

def calc_dw_features(detail, band_cols):
    rows=[]
    for (label,time,center),g in detail.groupby(['label','time','center_norm']):
        near=g[g['distance_cm']<=NEAR_DISTANCE_MAX_CM].copy()
        if len(near)==0: near=g.copy()
        M=np.zeros((len(DIRS), len(SUBBANDS)))
        for i,d in enumerate(DIRS):
            gd=near[near['direction'].str.lower()==d.lower()]
            for j,band in enumerate(SUBBANDS):
                M[i,j]=num(gd[band_cols[band]]).fillna(0).sum() if len(gd) else 0
        if M.sum()<=1e-20:
            actual=sorted(near['direction'].astype(str).unique())
            M=np.zeros((len(actual), len(SUBBANDS)))
            for i,d in enumerate(actual):
                gd=near[near['direction'].astype(str)==d]
                for j,band in enumerate(SUBBANDS): M[i,j]=num(gd[band_cols[band]]).fillna(0).sum()
            dirs_used=actual
        else: dirs_used=DIRS
        total=float(M.sum())
        row={'label':label,'time':time,'center_norm':center,'dw_total_near_band_energy':total,'dw_n_directions_used':len(dirs_used)}
        if total<=1e-20:
            for c in ['dw_top1_direction_ratio','dw_top2_direction_ratio','dw_top3_direction_ratio','dw_direction_entropy_norm','dw_direction_cv','dw_direction_gini','dw_direction_active_count','dw_top1_wideband_quality','dw_top2_wideband_quality','dw_top1_wideband_coverage','dw_top2_wideband_coverage','dw_top1_band_entropy_norm','dw_top2_band_entropy_norm','dw_top1_band_flatness','dw_top2_band_flatness','dw_rest_wideband_quality','dw_directed_wideband_score','dw_diffuse_wideband_score','dw_directional_wideband_contrast','dw_top2_wideband_minus_rest','dw_matrix_entropy_norm','dw_matrix_top10_ratio','dw_matrix_top20_ratio','dw_low_band_20_30_ratio','dw_mid_band_30_50_ratio','dw_high_band_50_70_ratio','dw_top2_low_band_ratio','dw_top2_mid_band_ratio','dw_top2_high_band_ratio']:
                row[c]=0.0
            rows.append(row); continue
        dir_tot=M.sum(axis=1); idx=np.argsort(dir_tot)[::-1]; top1=idx[0]; top2=idx[:2]; top3=idx[:3]
        dir_p=dir_tot/(total+1e-20)
        dir_ent=entropy_norm(dir_p); dir_cv=float(dir_tot.std()/(dir_tot.mean()+1e-20)); dir_gini=gini(dir_tot)
        active_dir=int((dir_tot>=dir_tot.max()*DIR_ACTIVE_REL_TO_MAX).sum())
        cov=[]; bent=[]; bflat=[]
        for i in range(M.shape[0]):
            b=M[i]; s=b.sum()
            if s<=1e-20: cov.append(0); bent.append(0); bflat.append(0); continue
            active=(b>=b.max()*BAND_ACTIVE_REL_TO_DIR_MAX) & (b>=s*BAND_ACTIVE_FRAC_OF_DIR_TOTAL)
            cov.append(float(active.sum()/len(SUBBANDS))); bent.append(entropy_norm(b)); bflat.append(flatness(b))
        cov=np.array(cov); bent=np.array(bent); bflat=np.array(bflat)
        wide=0.50*cov+0.35*bent+0.15*bflat
        top2_wide=float(np.average(wide[top2], weights=dir_tot[top2]+1e-20))
        top2_cov=float(np.average(cov[top2], weights=dir_tot[top2]+1e-20))
        top2_ent=float(np.average(bent[top2], weights=dir_tot[top2]+1e-20))
        top2_flat=float(np.average(bflat[top2], weights=dir_tot[top2]+1e-20))
        rest=[i for i in range(M.shape[0]) if i not in set(top2.tolist())]
        rest_wide=float(np.average(wide[rest], weights=dir_tot[rest]+1e-20)) if rest and dir_tot[rest].sum()>0 else 0.0
        top2_ratio=float(dir_tot[top2].sum()/(total+1e-20))
        top2_wide_energy=float(np.sum(dir_tot[top2]*wide[top2]))
        rest_wide_energy=float(np.sum(dir_tot[rest]*wide[rest])) if rest else 0.0
        contrast=float((top2_wide_energy/(len(top2)+1e-20))/(rest_wide_energy/(len(rest)+1e-20)+1e-20)) if rest else 999.0
        concentration=max(0,1-dir_ent)
        directed_score=float(top2_ratio*top2_wide*(0.55+0.45*concentration)*(0.5+0.5*min(dir_cv,3)/3))
        diffuse_score=float(dir_ent*(0.5*rest_wide+0.5*(1-top2_ratio)))
        mat=np.ravel(M); sm=np.sort(mat)[::-1]
        band_tot=M.sum(axis=0); band_p=band_tot/(total+1e-20)
        top2_band=M[top2,:].sum(axis=0); top2_band_p=top2_band/(top2_band.sum()+1e-20)
        row.update({
            'dw_top1_direction': dirs_used[top1] if top1<len(dirs_used) else str(top1),
            'dw_top2_directions': '|'.join([dirs_used[i] if i<len(dirs_used) else str(i) for i in top2]),
            'dw_top1_direction_ratio':float(dir_tot[top1]/(total+1e-20)), 'dw_top2_direction_ratio':top2_ratio,
            'dw_top3_direction_ratio':float(dir_tot[top3].sum()/(total+1e-20)), 'dw_direction_entropy_norm':dir_ent,
            'dw_direction_cv':dir_cv, 'dw_direction_gini':dir_gini, 'dw_direction_active_count':active_dir,
            'dw_top1_wideband_quality':float(wide[top1]), 'dw_top2_wideband_quality':top2_wide,
            'dw_top1_wideband_coverage':float(cov[top1]), 'dw_top2_wideband_coverage':top2_cov,
            'dw_top1_band_entropy_norm':float(bent[top1]), 'dw_top2_band_entropy_norm':top2_ent,
            'dw_top1_band_flatness':float(bflat[top1]), 'dw_top2_band_flatness':top2_flat,
            'dw_rest_wideband_quality':rest_wide, 'dw_directed_wideband_score':directed_score,
            'dw_diffuse_wideband_score':diffuse_score, 'dw_directional_wideband_contrast':contrast,
            'dw_top2_wideband_minus_rest':float(top2_wide-rest_wide), 'dw_matrix_entropy_norm':entropy_norm(mat),
            'dw_matrix_top10_ratio':float(sm[:max(1,int(len(sm)*0.10))].sum()/(total+1e-20)),
            'dw_matrix_top20_ratio':float(sm[:max(1,int(len(sm)*0.20))].sum()/(total+1e-20)),
            'dw_low_band_20_30_ratio':float(band_p[0]), 'dw_mid_band_30_50_ratio':float(band_p[1]+band_p[2]),
            'dw_high_band_50_70_ratio':float(band_p[3]+band_p[4]), 'dw_top2_low_band_ratio':float(top2_band_p[0]),
            'dw_top2_mid_band_ratio':float(top2_band_p[1]+top2_band_p[2]), 'dw_top2_high_band_ratio':float(top2_band_p[3]+top2_band_p[4])})
        for j,(lo,hi) in enumerate(SUBBANDS):
            row[f'dw_band_{lo}_{hi}k_global_ratio']=float(band_p[j]); row[f'dw_band_{lo}_{hi}k_top2_ratio']=float(top2_band_p[j])
        rows.append(row)
    return pd.DataFrame(rows)

def add_time_rel(df, cols):
    df=df.copy()
    for c in cols:
        v=num(df[c]); df[c+'__time_robust_z']=np.nan; df[c+'__time_rank_pct']=np.nan
        for t,idx in df.groupby('time').groups.items():
            x=v.loc[idx]; med=x.median(); mad=(x-med).abs().median()
            if not np.isfinite(mad) or mad<1e-12: mad=x.std()
            if not np.isfinite(mad) or mad<1e-12: mad=1.0
            df.loc[idx,c+'__time_robust_z']=(x-med)/(1.4826*mad)
            df.loc[idx,c+'__time_rank_pct']=x.rank(method='average',pct=True)
    return df

def build_dataset():
    ensure(OUT_DIR)
    base, base_path=load_base(); detail=load_detail(); bands=find_band_cols(detail)
    print('基础特征表:',base_path); print('识别到频段列:',bands)
    dw=calc_dw_features(detail,bands)
    dw_path=os.path.join(OUT_DIR,'v8_directed_wideband_features.csv'); dw.to_csv(dw_path,index=False,encoding='utf-8-sig')
    df=base.merge(dw,on=['label','time','center_norm'],how='left',validate='one_to_one')
    dw_cols=[c for c in df.columns if c.startswith('dw_') and c not in ['dw_top1_direction','dw_top2_directions']]
    df=add_time_rel(df,dw_cols)
    combined_path=os.path.join(OUT_DIR,'v8_combined_v7_directed_wideband_dataset.csv')
    df.to_csv(combined_path,index=False,encoding='utf-8-sig')
    return df, base_path, dw_path, combined_path

def base_numeric_cols(df):
    cols=[]
    for c in df.columns:
        if c in META or c.startswith('dw_'): continue
        v=num(df[c])
        if v.notna().mean()>=0.8: cols.append(c)
    return cols

def build_X(tr,te,cols):
    Xtr=pd.DataFrame(index=tr.index); Xte=pd.DataFrame(index=te.index)
    for c in cols:
        a=num(tr[c]); b=num(te[c]); med=a.median()
        if not np.isfinite(med): med=0.0
        Xtr[c]=a.fillna(med); Xte[c]=b.fillna(med)
    return Xtr,Xte

def rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=700,random_state=RANDOM_STATE,class_weight='balanced',n_jobs=-1,min_samples_leaf=1)

def oof_prob(X,y,groups):
    y=np.asarray(y); groups=np.asarray(groups).astype(str); p=np.zeros(len(y)); filled=np.zeros(len(y),dtype=bool)
    for g in sorted(pd.unique(groups)):
        val=groups==g; tr=~val
        if len(np.unique(y[tr]))<2: continue
        m=rf(); m.fit(X.loc[tr], y[tr]); p[val]=m.predict_proba(X.loc[val])[:,1]; filled[val]=True
    if not filled.all():
        from sklearn.model_selection import StratifiedKFold
        cv=StratifiedKFold(n_splits=max(2,min(5,int(min((y==0).sum(),(y==1).sum())))),shuffle=True,random_state=RANDOM_STATE)
        for tr_i,val_i in cv.split(X,y):
            m=rf(); m.fit(X.iloc[tr_i],y[tr_i]); p[val_i]=m.predict_proba(X.iloc[val_i])[:,1]
    return p

def choose_thr(y,p):
    best_t=0.5; best=-1; rows=[]
    for t in TH_GRID:
        pred=(p>=t).astype(int); m=metrics(y,pred); score=m['balanced_accuracy']; rows.append({'threshold':t,'score':score,**m})
        if score>best: best=score; best_t=float(t)
    return best_t,best,pd.DataFrame(rows)

def rank_pred_by_group(pred_df, score_col, prefix):
    pred_df=pred_df.copy(); rc=prefix+'_rank_pct'; pc=prefix+'_rank_pred'; pred_df[rc]=0.5
    for g,idx in pred_df.groupby('test_group').groups.items(): pred_df.loc[idx,rc]=num(pred_df.loc[idx,score_col]).rank(method='average',pct=True)
    pred_df[pc]=np.where(pred_df[rc]>(1-RANK_TRUE_FRACTION),'TRUE_LEAK','FALSE_LEAK')
    return pred_df

def validate(df, feat_cols, name, use_dw=False):
    exp_dir=os.path.join(OUT_DIR,name); ensure(exp_dir)
    summaries=[]; preds=[]
    print('\n实验:',name,'特征数:',len(feat_cols))
    for tg in sorted(df['time'].astype(str).unique()):
        mask=df['time'].astype(str).values==tg; tr=df.loc[~mask].reset_index(drop=True); te=df.loc[mask].reset_index(drop=True)
        ytr=y_from_label(tr['label']); yte=y_from_label(te['label'])
        Xtr,Xte=build_X(tr,te,feat_cols)
        oof=oof_prob(Xtr,ytr,tr['time'].astype(str).values); th,score,curve=choose_thr(ytr,oof)
        curve.to_csv(os.path.join(exp_dir,f'{name}_threshold_without_{safe_name(tg)}.csv'),index=False,encoding='utf-8-sig')
        m=rf(); m.fit(Xtr,ytr); prob=m.predict_proba(Xte)[:,1]
        model=(prob>=th).astype(int); default=(prob>=0.5).astype(int)
        pdf=pd.DataFrame({'experiment':name,'test_group':tg,'time':te['time'].astype(str).values,
                          'center':te['center'].values if 'center' in te.columns else te['center_norm'].values,
                          'center_norm':te['center_norm'].values,'true_label':te['label'].astype(str).values,
                          'prob_TRUE_LEAK':prob,'best_threshold':th,
                          'default_pred':[lab(x) for x in default],'model_pred':[lab(x) for x in model]})
        pdf=rank_pred_by_group(pdf,'prob_TRUE_LEAK','prob')
        if 'dw_directed_wideband_score' in te.columns:
            pdf['dw_directed_wideband_score']=num(te['dw_directed_wideband_score']).values
            pdf['dw_diffuse_wideband_score']=num(te.get('dw_diffuse_wideband_score',pd.Series(np.nan,index=te.index))).values
            pdf['dw_directional_wideband_contrast']=num(te.get('dw_directional_wideband_contrast',pd.Series(np.nan,index=te.index))).values
            pdf=rank_pred_by_group(pdf,'dw_directed_wideband_score','dw')
        else:
            pdf['dw_rank_pred']=pdf['model_pred']; pdf['dw_rank_pct']=np.nan
        pdf['hybrid_pred']=pdf['dw_rank_pred'] if use_dw else pdf['model_pred']
        pdf['model_correct']=(pdf['model_pred']==pdf['true_label']).astype(int)
        pdf['prob_rank_correct']=(pdf['prob_rank_pred']==pdf['true_label']).astype(int)
        pdf['dw_rank_correct']=(pdf['dw_rank_pred']==pdf['true_label']).astype(int)
        pdf['hybrid_correct']=(pdf['hybrid_pred']==pdf['true_label']).astype(int)
        preds.extend(pdf.to_dict('records'))
        met_model=metrics(yte,model); met_def=metrics(yte,default); met_prob=metrics(yte,y_from_label(pdf['prob_rank_pred'])); met_dw=metrics(yte,y_from_label(pdf['dw_rank_pred'])); met_h=metrics(yte,y_from_label(pdf['hybrid_pred']))
        auc=auc_score(yte,prob)
        summaries.append({'experiment':name,'test_group':tg,'n_test':len(te),'n_features':len(feat_cols),'auc':auc,'best_threshold':th,
                          'default_acc':met_def['accuracy'],'model_acc':met_model['accuracy'],'prob_rank_acc':met_prob['accuracy'],
                          'dw_rank_acc':met_dw['accuracy'],'hybrid_acc':met_h['accuracy'],
                          'model_recall_TRUE':met_model['recall_TRUE'],'model_recall_FALSE':met_model['recall_FALSE'],
                          'dw_recall_TRUE':met_dw['recall_TRUE'],'dw_recall_FALSE':met_dw['recall_FALSE']})
        print(f"  {tg}: AUC={auc:.3f}, model={met_model['accuracy']:.3f}, prob_rank={met_prob['accuracy']:.3f}, dw_rank={met_dw['accuracy']:.3f}, hybrid={met_h['accuracy']:.3f}")
    s=pd.DataFrame(summaries); p=pd.DataFrame(preds)
    sp=os.path.join(exp_dir,f'{name}_group_summary.csv'); pp=os.path.join(exp_dir,f'{name}_predictions.csv')
    s.to_csv(sp,index=False,encoding='utf-8-sig'); p.to_csv(pp,index=False,encoding='utf-8-sig')
    return {'experiment':name,'summary':s,'preds':p,'summary_path':sp,'pred_path':pp}

def compare_144226(df, cols):
    sub=df[df['time'].astype(str)==TARGET_TIME].copy(); rows=[]
    for c in cols:
        if c not in sub.columns: continue
        tv=num(sub.loc[sub['label']=='TRUE_LEAK',c]).dropna().values; fv=num(sub.loc[sub['label']=='FALSE_LEAK',c]).dropna().values
        if len(tv)<2 or len(fv)<2: continue
        y=np.r_[np.ones(len(tv)),np.zeros(len(fv))]; score=np.r_[tv,fv]; auc=auc_score(y,score)
        rows.append({'feature':c,'true_mean':tv.mean(),'false_mean':fv.mean(),'diff_TRUE_minus_FALSE':tv.mean()-fv.mean(),'cohen_d':cohen_d(tv,fv),'auc_signed_TRUE_larger':auc,'auc_direction_free':max(auc,1-auc) if np.isfinite(auc) else np.nan})
    out=pd.DataFrame(rows).sort_values(['auc_direction_free','cohen_d'],ascending=[False,False]) if rows else pd.DataFrame()
    path=os.path.join(OUT_DIR,'v8_directed_wideband_feature_compare_144226.csv'); out.to_csv(path,index=False,encoding='utf-8-sig')
    return out,path

def pair_144226(preds):
    sub=preds[preds['test_group'].astype(str)==TARGET_TIME].copy(); rows=[]
    for cen,g in sub.groupby('center_norm'):
        t=g[g['true_label']=='TRUE_LEAK']; f=g[g['true_label']=='FALSE_LEAK']
        if len(t)==0 or len(f)==0: continue
        t=t.iloc[0]; f=f.iloc[0]
        rows.append({'center_norm':cen,'true_prob':t['prob_TRUE_LEAK'],'false_prob':f['prob_TRUE_LEAK'],
                     'prob_diff_TRUE_minus_FALSE':t['prob_TRUE_LEAK']-f['prob_TRUE_LEAK'],
                     'prob_order_correct':int(t['prob_TRUE_LEAK']>f['prob_TRUE_LEAK']),
                     'true_dw_score':t.get('dw_directed_wideband_score',np.nan),'false_dw_score':f.get('dw_directed_wideband_score',np.nan),
                     'dw_score_diff_TRUE_minus_FALSE':t.get('dw_directed_wideband_score',np.nan)-f.get('dw_directed_wideband_score',np.nan),
                     'dw_order_correct':int(t.get('dw_directed_wideband_score',np.nan)>f.get('dw_directed_wideband_score',np.nan)),
                     'true_model_pred':t.get('model_pred',''),'false_model_pred':f.get('model_pred',''),
                     'true_dw_pred':t.get('dw_rank_pred',''),'false_dw_pred':f.get('dw_rank_pred',''),
                     'true_hybrid_pred':t.get('hybrid_pred',''),'false_hybrid_pred':f.get('hybrid_pred','')})
    out=pd.DataFrame(rows).sort_values('center_norm') if rows else pd.DataFrame()
    path=os.path.join(OUT_DIR,'v8_144226_pair_check.csv'); out.to_csv(path,index=False,encoding='utf-8-sig')
    return out,path

def plot_results(results):
    fig_dir=os.path.join(OUT_DIR,'figures'); ensure(fig_dir); paths=[]
    allsum=pd.concat([r['summary'] for r in results], ignore_index=True)
    for metric in ['auc','model_acc','dw_rank_acc','hybrid_acc']:
        plt.figure(figsize=(12,5)); exps=allsum['experiment'].unique().tolist(); groups=sorted(allsum['test_group'].unique().tolist()); x=np.arange(len(groups)); w=0.22
        for i,e in enumerate(exps):
            vals=[float(allsum[(allsum['experiment']==e)&(allsum['test_group']==g)][metric].iloc[0]) for g in groups]
            plt.bar(x+(i-(len(exps)-1)/2)*w,vals,w,label=e)
        plt.xticks(x,groups,rotation=45,ha='right'); plt.ylim(0,1.05); plt.ylabel(metric); plt.title('v8 directed-wideband '+metric); plt.grid(True,axis='y',alpha=.3); plt.legend(fontsize=8); plt.tight_layout()
        p=os.path.join(fig_dir,f'v8_{metric}_comparison.png'); plt.savefig(p,dpi=150); plt.close(); paths.append(p)
    return paths

def make_report(df,base_path,dw_path,combined_path,results,cmp_df,cmp_path,pair_df,pair_path,plots):
    lines=[]; lines.append('v8 方向性宽频特征版报告'); lines.append('='*100); lines.append('生成时间: '+str(datetime.now())); lines.append('')
    lines += [f'base/v7: {base_path}', f'directed-wideband features: {dw_path}', f'combined: {combined_path}', f'TRUE detail: {TRUE_DETAIL_CSV}', f'FALSE detail: {FALSE_DETAIL_CSV}', '']
    overall=[]
    for r in results:
        s=r['summary']; lines.append(f"[{r['experiment']}]"); lines.append('  '+r['summary_path']); lines.append('  '+r['pred_path'])
        lines.append(f"  mean AUC={s['auc'].mean():.4f}, model={s['model_acc'].mean():.4f}, dw_rank={s['dw_rank_acc'].mean():.4f}, hybrid={s['hybrid_acc'].mean():.4f}")
        target=s[s['test_group']==TARGET_TIME]
        row={'experiment':r['experiment'],'mean_auc':s['auc'].mean(),'mean_model_acc':s['model_acc'].mean(),'mean_dw_rank_acc':s['dw_rank_acc'].mean(),'mean_hybrid_acc':s['hybrid_acc'].mean()}
        if len(target):
            t=target.iloc[0]; lines.append(f"  144226: AUC={t['auc']:.4f}, model={t['model_acc']:.4f}, dw_rank={t['dw_rank_acc']:.4f}, hybrid={t['hybrid_acc']:.4f}")
            row.update({'auc_144226':t['auc'],'model_acc_144226':t['model_acc'],'dw_rank_acc_144226':t['dw_rank_acc'],'hybrid_acc_144226':t['hybrid_acc']})
        for _,t in s.iterrows(): lines.append(f"    {t['test_group']}: AUC={t['auc']:.3f}, model={t['model_acc']:.3f}, dw_rank={t['dw_rank_acc']:.3f}, hybrid={t['hybrid_acc']:.3f}")
        lines.append(''); overall.append(row)
    overall_df=pd.DataFrame(overall); overall_path=os.path.join(OUT_DIR,'v8_group_validation_summary.csv'); overall_df.to_csv(overall_path,index=False,encoding='utf-8-sig')
    lines.append('144226 directed-wideband feature compare: '+cmp_path)
    if len(cmp_df):
        for _,r in cmp_df.head(12).iterrows(): lines.append(f"  {r['feature']}: AUC={r['auc_direction_free']:.3f}, diff={r['diff_TRUE_minus_FALSE']:.6g}, TRUE={r['true_mean']:.6g}, FALSE={r['false_mean']:.6g}")
    lines.append(''); lines.append('144226 pair check: '+pair_path)
    if len(pair_df):
        lines.append(f"  prob_pair_acc={pair_df['prob_order_correct'].mean():.4f}"); lines.append(f"  dw_pair_acc={pair_df['dw_order_correct'].mean():.4f}")
        fail=pair_df[pair_df['dw_order_correct']==0]['center_norm'].astype(str).tolist(); lines.append('  dw_failed_centers: '+' | '.join(fail))
    lines.append(''); lines.append('plots:'); lines += ['  '+p for p in plots]
    report=os.path.join(OUT_DIR,'v8_report.txt'); open(report,'w',encoding='utf-8').write('\n'.join(lines))
    return report,overall_path

def safe_name(s): return str(s).replace('\\','_').replace('/','_').replace(':','_').replace('.','_')

def main():
    ensure(OUT_DIR); print('='*100); print('v8: v7 + 方向性宽频特征'); print('='*100)
    df,base_path,dw_path,combined_path=build_dataset()
    print('样本数:',len(df)); print(df['label'].value_counts()); print('groups:',sorted(df['time'].astype(str).unique()))
    v7_cols=base_numeric_cols(df)
    dw_cols=[c for c in df.columns if c.startswith('dw_') and c not in ['dw_top1_direction','dw_top2_directions'] and num(df[c]).notna().mean()>=0.8]
    print('v7/base特征数:',len(v7_cols),'dw特征数:',len(dw_cols))
    json.dump({'v7_cols':v7_cols,'dw_cols':dw_cols}, open(os.path.join(OUT_DIR,'v8_feature_columns.json'),'w',encoding='utf-8'), ensure_ascii=False, indent=2)
    results=[]
    results.append(validate(df,v7_cols,'A_v7_baseline',use_dw=False))
    results.append(validate(df,dw_cols,'B_directed_wideband_only',use_dw=True))
    combo=[]
    for c in v7_cols+dw_cols:
        if c not in combo: combo.append(c)
    results.append(validate(df,combo,'C_v7_plus_directed_wideband',use_dw=True))
    all_preds=pd.concat([r['preds'] for r in results],ignore_index=True); pred_path=os.path.join(OUT_DIR,'v8_predictions.csv'); all_preds.to_csv(pred_path,index=False,encoding='utf-8-sig')
    cmp_df,cmp_path=compare_144226(df,dw_cols); pair_df,pair_path=pair_144226(results[-1]['preds']); plots=plot_results(results)
    report,overall=make_report(df,base_path,dw_path,combined_path,results,cmp_df,cmp_path,pair_df,pair_path,plots)
    print('\n完成。输出文件夹:',OUT_DIR); print('报告:',report); print('汇总:',overall); print('预测:',pred_path); print('144226配对:',pair_path); print('144226特征对比:',cmp_path)
    print('\n核心结果摘要:')
    for r in results:
        s=r['summary']; print('\n'+r['experiment']); print(f"  mean AUC={s['auc'].mean():.3f}, model={s['model_acc'].mean():.3f}, dw_rank={s['dw_rank_acc'].mean():.3f}, hybrid={s['hybrid_acc'].mean():.3f}")
        t=s[s['test_group']==TARGET_TIME]
        if len(t):
            x=t.iloc[0]; print(f"  144226: AUC={x['auc']:.3f}, model={x['model_acc']:.3f}, dw_rank={x['dw_rank_acc']:.3f}, hybrid={x['hybrid_acc']:.3f}")
    if len(pair_df):
        print('\n144226 center配对检查:'); print(f"  prob排序正确率: {pair_df['prob_order_correct'].mean():.3f}"); print(f"  directed-wideband排序正确率: {pair_df['dw_order_correct'].mean():.3f}")
        fail=pair_df[pair_df['dw_order_correct']==0]['center_norm'].astype(str).tolist(); print('  directed-wideband失败center:', ' | '.join(fail) if fail else '无')
    print('\n请把核心结果摘要和144226 center配对检查发给我。')

if __name__=='__main__': main()
