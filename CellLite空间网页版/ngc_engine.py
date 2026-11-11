import numpy as np
from sklearn.neighbors import NearestNeighbors
from training import edges,partition

def infer_ngc(d,artifact):
    if 'z' in d and (d.z!=0).any():raise ValueError('此NGC模型在二维投影训练，需要明确的二维数据（z=0），不能直接预测三维数据。')
    d=d.copy();d['z']=0.
    model=artifact['model'];model.n_jobs=1;genes=artifact['genes'];vocab={g:i+1 for i,g in enumerate(genes)};config=artifact['config']
    def prepare(d):
     p,x,w=edges(d);ids=d.features.astype(str).map(vocab).fillna(0).to_numpy(int);xy=d.reindex(columns=['x','y','z'],fill_value=0).to_numpy(float);nn=NearestNeighbors(n_neighbors=25).fit(xy).kneighbors(xy,return_distance=False)[:,1:];hist=np.zeros((len(d),len(genes)+1),np.float32);np.add.at(hist,(np.repeat(np.arange(len(d)),24),ids[nn].ravel()),1/24);return p,x.astype(np.float32),w,hist
    
    def pair_features(p,x,h,ix):
     a,b=p[ix].T;return np.column_stack([x[ix],np.abs(h[a]-h[b]),(h[a]+h[b])*.5]).astype(np.float32)
    

    pairs,x,w,h=prepare(d);prob=np.empty(len(pairs))
    for start in range(0,len(pairs),16384):
        ix=np.arange(start,min(start+16384,len(pairs)));prob[ix]=model.predict_proba(pair_features(pairs,x,h,ix))[:,1]
    prob=np.round(prob,12)
    return partition(d,pairs,np.round(w*((1-config['mix'])+config['mix']*prob),12),config['resolution'])
