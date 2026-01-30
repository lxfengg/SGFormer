#!/usr/bin/env python3
# scripts/parse_pseudo_to_csv.py
import glob, re, csv, json, os
out = 'results/diagnosis_pseudo_summary.csv'
os.makedirs('results', exist_ok=True)
rows=[]
for txt in sorted(glob.glob('results/diagnosis_per_ckpt/*.txt')):
    with open(txt,'r',encoding='utf-8',errors='ignore') as f:
        s=f.read()
    basename=os.path.basename(txt)
    ckpt = re.sub(r'^diag_','',basename).rsplit('.txt',1)[0]
    # try to extract Overall unlabeled acc, thr lines like:
    # Overall unlabeled acc (proxy): 0.2107
    d={}
    m = re.search(r'Overall unlabeled acc .*?:\s*([0-9.]+)', s)
    if m:
        d['teacher_unlabeled_acc']=float(m.group(1))
    for thr in [0.5,0.6,0.7,0.8]:
        # look for lines like "thr 0.5: coverage 0/2568 acc_on_conf=None"
        pat = rf'thr\s*{thr}\s*:\s*coverage\s*([0-9]+)/([0-9]+)\s*acc_on_conf\s*=\s*([0-9.]+|None)'
        m2 = re.search(pat, s)
        if m2:
            cov = int(m2.group(1))
            total = int(m2.group(2))
            acc = None if m2.group(3)=='None' else float(m2.group(3))
            d[f'coverage_{int(thr*100)}']=cov
            d[f'coverage_{int(thr*100)}_frac']=cov/total if total>0 else 0.0
            d[f'acc_on_conf_{int(thr*100)}']=acc
    # fallback: try to extract confidences and counts like "stu_conf_gt05"
    m3 = re.search(r'stu_conf_gt[_ ]?0?5[:=]?\s*([0-9]+)', s)
    if m3:
        d['stu_conf_gt_50'] = int(m3.group(1))
    rows.append({'ckpt':ckpt, **d})
# write CSV
if rows:
    keys = sorted({k for r in rows for k in r.keys()})
    with open(out,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print("Wrote", out)
else:
    print("No diag txt files parsed.")
