"""Coordinate-only spatial graph baseline. Does not reject background molecules."""
import argparse,json
import numpy as np
import pandas as pd
import igraph as ig
import leidenalg as la
from sklearn.neighbors import NearestNeighbors
def graph(d,k):
 xyz=d.reindex(columns=['x','y','z'],fill_value=0).to_numpy(float)
 dist,nn=NearestNeighbors(n_neighbors=k+1).fit(xyz).kneighbors(xyz)
 a=np.repeat(np.arange(len(d)),k);b=nn[:,1:].ravel();ds=dist[:,1:].ravel();scale=np.maximum(dist[:,-1],1e-6)
 edges=np.unique(np.sort(np.column_stack([a,b]),axis=1),axis=0);a,b=edges.T
 ds=np.linalg.norm(xyz[a]-xyz[b],axis=1)
 weights=np.exp(-ds**2/(scale[a]*scale[b]))
 return ig.Graph(n=len(d),edges=edges.tolist(),directed=False),weights.tolist()
def predict(d,k,resolution):
 g,w=graph(d,k);return np.array(la.find_partition(g,la.CPMVertexPartition,weights=w,resolution_parameter=resolution,seed=42).membership)+1

if __name__=='__main__':
 parser=argparse.ArgumentParser()
 parser.add_argument('--input',required=True)
 parser.add_argument('--output',required=True)
 parser.add_argument('--neighbors',type=int,default=32)
 parser.add_argument('--resolution',type=float,default=.01)
 args=parser.parse_args()
 df=pd.read_csv(args.input)
 if not {'x','y'}.issubset(df.columns):raise ValueError('Input requires x,y; optional z')
 if len(df)<=args.neighbors:raise ValueError('Number of molecules must exceed neighbors')
 if not np.isfinite(df.reindex(columns=['x','y','z'],fill_value=0).to_numpy(float)).all():raise ValueError('Coordinates must be finite')
 labels=predict(df,args.neighbors,args.resolution)
 df['predicted_cells']=labels
 df.to_csv(args.output,index=False)
 print(json.dumps(dict(molecules=len(df),clusters=len(np.unique(labels)),background_rejection=False)))
