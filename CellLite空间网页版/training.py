import numpy as np,pandas as pd,json,joblib,uuid
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_mutual_info_score,adjusted_rand_score
import igraph as ig,leidenalg as la
def truth(df):
    column = 'raw_cells' if 'raw_cells' in df else 'segmented'
    if column not in df:
        raise ValueError('训练/评估需要 raw_cells 或 segmented 真实细胞编号，不能使用 predicted_cells 作为真实标签。')
    cells = df[column].astype(str).to_numpy()
    valid = ~np.isin(cells, ['0', '0.0', '-1', '-1.0', 'background', 'nan', 'None', ''])
    label_column = 'raw_labels' if 'raw_labels' in df else 'labels'
    if label_column in df:
        valid &= df[label_column].astype(str).to_numpy() != 'background'
    cells[~valid] = '0'
    return cells, valid


def spatial_split(df):
    """Spatial holdouts; remove cells straddling partitions and a boundary buffer."""
    cuts=np.quantile(df.x,[.6,.8])
    fold=np.digitize(df.x,cuts)
    cells,fg=truth(df)
    crossing=pd.DataFrame({'cell':cells[fg],'fold':fold[fg]}).groupby('cell')['fold'].nunique()
    crossing=set(crossing[crossing>1].index)
    margin=max(float(df.x.max()-df.x.min())*.01,1e-6)
    keep=~np.isin(cells,list(crossing))
    for cut in cuts: keep &= np.abs(df.x.to_numpy()-cut)>margin
    parts=[df.loc[keep&(fold==i)].reset_index(drop=True) for i in range(3)]
    for part in parts:
        if len(part)<50 or len(np.unique(truth(part)[0][truth(part)[1]]))<2:
            raise ValueError('空间划分后有效细胞不足；请扩大训练区域。')
    return parts



def edges(d):
 xyz=d.reindex(columns=['x','y','z'],fill_value=0).to_numpy(float)
 if len(d)<34 or not np.isfinite(xyz).all():raise ValueError('每个分区至少34条有效坐标')
 dist,nn=NearestNeighbors(n_neighbors=33).fit(xyz).kneighbors(xyz)
 pairs=np.unique(np.sort(np.column_stack([np.repeat(np.arange(len(d)),32),nn[:,1:].ravel()]),axis=1),axis=0)
 a,b=pairs.T;scale=np.maximum(dist[:,-1],1e-6);r=np.linalg.norm(xyz[a]-xyz[b],axis=1)/np.sqrt(scale[a]*scale[b])
 gene=d.features.astype(str).to_numpy()
 x=np.column_stack([r,np.abs(np.log(scale[a]/scale[b])),gene[a]==gene[b],np.log1p(dist[a,8]/scale[a]),np.log1p(dist[b,8]/scale[b])])
 return pairs,x,np.exp(-r*r)
def partition(d,pairs,weights,res):
 g=ig.Graph(n=len(d),edges=pairs.tolist(),directed=False)
 return np.array(la.find_partition(g,la.CPMVertexPartition,weights=weights.tolist(),resolution_parameter=res,seed=42).membership)+1
def metric(d,p):
 c,f=truth(d);return dict(ami=float(adjusted_mutual_info_score(c[f],p[f])),ari=float(adjusted_rand_score(c[f],p[f])))
def train_model(d,folder):
 if not {'x','y','features'}.issubset(d):raise ValueError('训练需要x,y,features及segmented或raw_cells参考编号')
 tr,va,te=spatial_split(d);pairs,x,w=edges(tr);c,fg=truth(tr);a,b=pairs.T;y=(c[a]==c[b])&fg[a]&fg[b]
 if len(np.unique(y))<2:raise ValueError('训练需含同细胞与跨细胞的分子对')
 rng=np.random.default_rng(42);ix=rng.choice(len(y),min(60000,len(y)),replace=False)
 model=ExtraTreesClassifier(n_estimators=100,max_depth=18,min_samples_leaf=5,n_jobs=4,class_weight='balanced',random_state=42).fit(x[ix],y[ix])
 vp,vx,vw=edges(va);learn=model.predict_proba(vx)[:,1];grid=[]
 for mix in [0.,.5,1.]:
  for res in [.001,.003,.01,.03]:
   p=partition(va,vp,vw*((1-mix)+mix*learn),res);grid.append(dict(mix=mix,resolution=res,validation=metric(va,p)))
 best=max(grid,key=lambda q:q['validation']['ami']);tp,tx,tw=edges(te);p=partition(te,tp,tw*((1-best['mix'])+best['mix']*model.predict_proba(tx)[:,1]),best['resolution']);score=metric(te,p)
 artifact=dict(model=model,config=best,genes=sorted(tr.features.astype(str).unique()),schema=1)
 folder.mkdir(parents=True);joblib.dump(artifact,folder/'model.joblib')
 report=dict(id=folder.name,validation=best['validation'],test=score,target_met=score['ami']>=.7,selected=best,split_rows=[len(tr),len(va),len(te)],grid=grid,method='ExtraTrees learned pair classifier + spatial Leiden; validation may select geometry-only fallback if mix=0',limitations='单样本开发空间留出；非跨样本效果保证；参考必须来自独立标注，不能用本程序预测值充当真值。当前未识别背景。')
 (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');return report
def infer_model(d,folder):
 art=joblib.load(folder/'model.joblib');unknown=float((~d.features.astype(str).isin(art['genes'])).mean())
 if unknown>.2:raise ValueError('超过20%基因不匹配训练词表，请选同类型数据或重新训练')
 if art.get('schema')=='ngc_context_v1':
  from ngc_engine import infer_ngc
  return infer_ngc(d,art)
 p,x,w=edges(d);cfg=art['config'];return partition(d,p,w*((1-cfg['mix'])+cfg['mix']*art['model'].predict_proba(x)[:,1]),cfg['resolution'])
