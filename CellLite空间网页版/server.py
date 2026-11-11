import sys,os,io,json,threading,webbrowser,argparse,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent
for path in [BASE/'deps',Path(r'C:/Users/李雅雯/Documents/Codex/2026-11-06/bang-w/work/spatial/deps'),Path(r'C:/Users/李雅雯/Documents/Codex/2026-11-06/d-graduation-w/work/ui-deps')]:
 if path.exists():sys.path.insert(0,str(path))
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import numpy as np,pandas as pd
from spatial_graph import predict
from sklearn.metrics import adjusted_mutual_info_score,adjusted_rand_score
LOCK=threading.Lock()
class Handler(BaseHTTPRequestHandler):
 def reply(self,status,data,ctype='application/json; charset=utf-8'):
  raw=data if isinstance(data,bytes) else json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
  self.send_response(status);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(raw)
 def do_GET(self):
  if self.path=='/':return self.reply(200,(BASE/'index.html').read_bytes(),'text/html; charset=utf-8')
  if self.path=='/api/models':return self.reply(200,[json.loads(p.read_text(encoding='utf-8')) for p in (BASE/'models').glob('*/report.json')])
  if self.path=='/api/archived-models':return self.reply(200,[json.loads(p.read_text(encoding='utf-8')) for p in (BASE/'archived_models').glob('*/report.json')])
  if self.path=='/example':return self.reply(200,(BASE/'示例数据.csv').read_bytes(),'text/csv; charset=utf-8')
  return self.reply(404,{'error':'Not found'})
 def do_POST(self):
  if self.path not in ['/api/segment','/api/image','/api/train','/api/rename','/api/delete-model','/api/restore-model']:return self.reply(404,{'error':'Not found'})
  if self.headers.get('Origin') not in [None,f'http://{self.headers.get("Host")}']:return self.reply(403,{'error':'Origin rejected'})
  if not LOCK.acquire(blocking=False):return self.reply(409,{'error':'正在处理另一项分析，请稍后重试。'})
  try:
   size=int(self.headers.get('Content-Length','0'))
   if not 0<size<=50000000:raise ValueError('文件请求超过50MB限制。')
   obj=json.loads(self.rfile.read(size))
   if self.path in ['/api/delete-model','/api/restore-model']:
    model_id=str(obj.get('model_id',''))
    if len(model_id)!=32 or any(c not in '0123456789abcdef' for c in model_id):
     raise ValueError('模型ID无效')
    active=BASE/'models'/model_id
    archived=BASE/'archived_models'/model_id
    for path in [active,archived]:
     if path.resolve()!=path:
      raise ValueError('模型目录不能使用重定向路径')
    source,target=(active,archived) if self.path=='/api/delete-model' else (archived,active)
    if not (source/'report.json').is_file():
     return self.reply(404,{'error':'模型已移动或不存在，请刷新列表。'})
    if target.exists():
     return self.reply(409,{'error':'目标目录已有同一模型，请检查模型列表。'})
    target.parent.mkdir(exist_ok=True)
    source.rename(target)
    return self.reply(200,{'model_id':model_id})
   if self.path=='/api/rename':
    mid=str(obj.get('model_id',''));name=str(obj.get('name','')).strip()
    if len(mid)!=32 or any(c not in '0123456789abcdef' for c in mid):raise ValueError('模型ID无效')
    if not 1<=len(name)<=80:raise ValueError('名称需为1至80字')
    path=BASE/'models'/mid/'report.json'
    report=json.loads(path.read_text(encoding='utf-8'));report['name']=name
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)
    self.reply(200,report);return
   if self.path=='/api/image':
    import base64
    from microscopy import load_image,segment,measurements,overlay,image_bytes,export_image_result
    raw=base64.b64decode(obj['image'],validate=True)
    display,gray=load_image(raw)
    config=dict(factor=float(obj.get('factor',.85)),min_distance=int(obj.get('distance',8)),min_area=int(obj.get('area',30)),exclude_border=False,polarity=obj.get('polarity','亮目标／暗背景'))
    if not .1<=config['factor']<=2 or not 1<=config['min_distance']<=100 or not 1<=config['min_area']<=10000:raise ValueError('参数范围无效')
    labels,stats=segment(gray,**config);table=measurements(labels,gray)
    png=base64.b64encode(image_bytes(overlay(display,labels,True))).decode()
    bundle=base64.b64encode(export_image_result(display,gray,labels,table,dict(config=config,stats=stats))).decode()
    self.reply(200,dict(count=len(table),overlay=png,bundle=bundle,csv=table.to_csv(index=False)))
    return
   k=int(obj.get('neighbors',32));res=float(obj.get('resolution',.01))
   if not 2<=k<=64 or not .00001<=res<=.2:raise ValueError('参数超出范围。')
   df=pd.read_csv(io.StringIO(obj['csv']),sep=None,engine='python')
   df.columns=df.columns.str.strip().str.lstrip('\ufeff')
   if not {'x','y'}.issubset(df.columns):raise ValueError('需要x、y列，可选z；请检查CSV列名。')
   if not k<len(df)<=100000:raise ValueError(f'数据需包含{k+1}至100000条分子。')
   xyz=df.reindex(columns=['x','y','z'],fill_value=0).to_numpy(float)
   if not np.isfinite(xyz).all():raise ValueError('坐标含空值或非数值。')
   if self.path=='/api/train':
    from training import train_model
    import uuid
    report=train_model(df,BASE/'models'/uuid.uuid4().hex)
    report['name']=str(obj.get('name','')).strip()[:80] or '自训练模型 '+report['id'][:8]
    (BASE/'models'/report['id']/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    self.reply(200,report);return
   if obj.get('mode')=='trained':
    from training import infer_model
    model_id=str(obj.get('model_id',''))
    if len(model_id)!=32 or any(c not in '0123456789abcdef' for c in model_id):raise ValueError('请选择已训练模型')
    if 'features' not in df:raise ValueError('模型预测需要features列')
    pred=infer_model(df,BASE/'models'/model_id)
    saved_report=json.loads((BASE/'models'/model_id/'report.json').read_text(encoding='utf-8'));k=32;res=saved_report['selected']['resolution']
   elif obj.get('mode')=='existing':
    if 'predicted_cells' not in df:raise ValueError('查看已有结果需要predicted_cells列')
    pred=pd.to_numeric(df.predicted_cells,errors='raise').to_numpy(dtype=int)
   else:pred=predict(df,k,res)
   df['predicted_cells']=pred
   report=dict(molecules=len(df),clusters=int(len(np.unique(pred))),neighbors=k,resolution=res,background_rejection=False)
   if obj.get('mode')=='trained':report.update(model_id=model_id,model_name=saved_report.get('name',model_id),engine=saved_report.get('engine','legacy'),evaluation_scope=saved_report.get('evaluation_scope','历史开发评测'))
   ref=next((c for c in ['raw_cells','segmented','manual_partial_reference'] if c in df),None)
   if ref:
    cells=df[ref].astype(str);fg=~cells.isin(['0','0.0','-1','-1.0','background','nan','None',''])
    lc=next((c for c in ['raw_labels','labels'] if c in df),None)
    if lc:fg&=df[lc].astype(str)!='background'
    if fg.sum()>1:
     report.update(ami=float(adjusted_mutual_info_score(cells[fg],pred[fg])),ari=float(adjusted_rand_score(cells[fg],pred[fg])),reference_cells=int(cells[fg].nunique()),evaluated_foreground=int(fg.sum()))
   summary=df.groupby('predicted_cells').agg(molecules=('x','size'),center_x=('x','mean'),center_y=('y','mean'),min_x=('x','min'),max_x=('x','max'),min_y=('y','min'),max_y=('y','max')).reset_index()
   report.update(median_molecules_per_group=float(summary.molecules.median()),small_groups_under_10=int((summary.molecules<10).sum()),x_range=[float(xyz[:,0].min()),float(xyz[:,0].max())],y_range=[float(xyz[:,1].min()),float(xyz[:,1].max())])
   genes=df.features.astype(str).value_counts().head(20).to_dict() if 'features' in df else {}
   ix=np.arange(len(df))
   points=np.column_stack([xyz[ix,:2],pred[ix]]).tolist()
   self.reply(200,dict(report=report,points=points,groups=summary.to_dict(orient='records'),genes=genes,csv=df.to_csv(index=False)))
  except Exception as e:self.reply(400,{'error':str(e)})
  finally:LOCK.release()
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8877);parser.add_argument('--no-browser',action='store_true');args=parser.parse_args()
 try:server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
 except OSError:
  server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
 url=f'http://127.0.0.1:{server.server_port}';print(url,flush=True)
 if not args.no_browser:threading.Timer(1,lambda:webbrowser.open(url)).start()
 server.serve_forever()
