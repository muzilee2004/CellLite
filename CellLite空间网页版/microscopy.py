"""Deterministic microscopy baseline: threshold, distance watershed and measurements."""
import hashlib
import html
import io
import json
import zipfile
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from PIL import Image,ImageOps,ImageDraw
from scipy import ndimage as ndi
from skimage.color import rgb2gray,label2rgb
from skimage.filters import threshold_otsu,threshold_local,gaussian
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed,find_boundaries

VERSION='3.0.0-microscopy'
MAX_PIXELS=12000000


def inspect_image(raw):
    with Image.open(io.BytesIO(raw)) as image:
        return {'frames':getattr(image,'n_frames',1),'width':image.width,'height':image.height,'mode':image.mode}


def load_image(raw,frame=0,channel='灰度'):
    with Image.open(io.BytesIO(raw)) as image:
        image.seek(frame)
        if image.width*image.height>MAX_PIXELS:raise ValueError('单张最多1200万像素，请先裁剪或降采样，并相应更新像素尺寸。')
        image=ImageOps.exif_transpose(image)
        arr=np.asarray(image).copy()
        if image.mode=='P':arr=np.asarray(image.convert('RGB'))
    if arr.ndim==3:
        arr=arr[:,:,:3].astype(float)
        if channel in ['红通道','绿通道','蓝通道']:gray=arr[:,: ,['红通道','绿通道','蓝通道'].index(channel)]
        else:gray=rgb2gray(arr)
    elif arr.ndim==2:gray=arr.astype(float)
    else:raise ValueError('只支持二维灰度、RGB图像或TIFF中的单个二维帧。')
    if not np.isfinite(gray).all():raise ValueError('图像包含无效像素。')
    low,high=np.percentile(gray,[.5,99.5])
    if high<=low:low,high=float(gray.min()),float(gray.max())
    norm=np.clip((gray-low)/max(high-low,1e-12),0,1).astype(np.float32)
    display=np.repeat((norm*255).astype(np.uint8)[:,:,None],3,axis=2)
    return display,norm


def segment(gray,polarity='亮目标／暗背景',method='自动全局阈值',factor=.85,min_area=30,max_area=10000,
            min_distance=8,exclude_border=True,background_correction=False):
    if min_area<1 or max_area<min_area:raise ValueError('面积上限不能小于面积下限。')
    work=1-gray if polarity=='暗目标／亮背景' else gray.copy()
    work=gaussian(work,sigma=1,preserve_range=True)
    if background_correction:
        work=np.maximum(work-gaussian(work,sigma=20,preserve_range=True),0)
        if work.max()>0:work/=work.max()
    if np.ptp(work)<1e-8:return np.zeros(gray.shape,np.int32),{'threshold':None,'candidates':0,'excluded_edge':0,'excluded_area':0}
    threshold=float(threshold_otsu(work))*factor
    if method=='背景不均匀局部阈值':
        block=min(51,min(gray.shape)//2*2-1)
        if block<3:raise ValueError('图像尺寸太小，不能使用局部阈值。')
        mask=work>threshold_local(work,block,offset=-.01)*factor
    else:mask=work>threshold
    mask=ndi.binary_fill_holes(mask)
    components,n=ndi.label(mask)
    sizes=np.bincount(components.ravel())
    mask&=sizes[components]>=min_area
    distance=ndi.distance_transform_edt(mask)
    smooth=gaussian(distance,sigma=.7,preserve_range=True)
    peaks=peak_local_max(smooth,min_distance=int(min_distance),labels=mask.astype(np.uint8),exclude_border=False)
    markers=np.zeros(gray.shape,np.int32)
    for i,(y,x) in enumerate(peaks,1):markers[y,x]=i
    # Ensure every foreground component has at least one seed.
    components,n=ndi.label(mask);next_id=int(markers.max())+1
    for region_id,part in enumerate(ndi.find_objects(components),1):
        if part is None:continue
        area=components[part]==region_id
        if not markers[part][area].any():
            y,x=np.unravel_index(np.argmax(np.where(area,distance[part],-1)),area.shape)
            markers[part][y,x]=next_id;next_id+=1
    labels=watershed(-smooth,markers,mask=mask).astype(np.int32)
    stats={'threshold':threshold,'candidates':int(labels.max()),'excluded_edge':0,'excluded_area':0}
    edge_ids=set(np.unique(np.r_[labels[0],labels[-1],labels[:,0],labels[:,-1]]))-{0}
    remove=[]
    for region in regionprops(labels):
        if exclude_border and region.label in edge_ids:
            stats['excluded_edge']+=1;remove.append(region.label)
        elif region.area<min_area or region.area>max_area:
            stats['excluded_area']+=1;remove.append(region.label)
    labels[np.isin(labels,remove)]=0
    return labels,stats


def measurements(labels,gray,pixel_um=None,offset=(0,0)):
    columns=['编号','面积_px²','等效直径_px','周长_px','中心X_px','中心Y_px','长轴_px','短轴_px','平均归一化强度','触及当前图像边界']
    rows=[];edge_ids=set(np.unique(np.r_[labels[0],labels[-1],labels[:,0],labels[:,-1]]))-{0}
    for r in regionprops(labels,intensity_image=gray):
        row={'编号':r.label,'面积_px²':float(r.area),'等效直径_px':float(r.equivalent_diameter_area),
             '周长_px':float(r.perimeter_crofton),'中心X_px':float(r.centroid[1]+offset[0]),'中心Y_px':float(r.centroid[0]+offset[1]),
             '长轴_px':float(r.axis_major_length),'短轴_px':float(r.axis_minor_length),'平均归一化强度':float(r.mean_intensity),
             '触及当前图像边界':r.label in edge_ids}
        if pixel_um is not None:
            if pixel_um<=0:raise ValueError('像素尺寸必须大于0。')
            row.update({'面积_µm²':float(r.area*pixel_um**2),'等效直径_µm':float(r.equivalent_diameter_area*pixel_um),
                        '周长_µm':float(r.perimeter_crofton*pixel_um)})
        rows.append(row)
    if pixel_um is not None:columns+=['面积_µm²','等效直径_µm','周长_µm']
    return pd.DataFrame(rows,columns=columns)


def overlay(display,labels,numbered=False):
    colored=label2rgb(labels,image=display/255.,bg_label=0,alpha=.28,saturation=1)
    colored[find_boundaries(labels,mode='inner')]=[0,1,.5]
    image=Image.fromarray((np.clip(colored,0,1)*255).astype(np.uint8))
    if numbered:
        draw=ImageDraw.Draw(image)
        for r in regionprops(labels):draw.text((r.centroid[1],r.centroid[0]),str(r.label),fill='yellow',stroke_width=1,stroke_fill='black')
    return image


def image_bytes(image,format='PNG'):
    out=io.BytesIO();image.save(out,format=format);return out.getvalue()


def export_image_result(display,gray,labels,table,manifest):
    import base64
    meta=dict(manifest,version=VERSION,count=len(table),timestamp_utc=datetime.now(timezone.utc).isoformat(),
              roi_shape=list(labels.shape),foreground_fraction=float(np.mean(labels>0)),
              interpretation='检测对象是图中可见的前景区域。核染图像测量细胞核，不等于完整细胞。未经独立人工标注验证准确率。')
    png=image_bytes(overlay(display,labels,True))
    report=f'''<!doctype html><meta charset="utf-8"><title>显微图像计数测量报告</title><style>body{{font:16px Arial,'Microsoft YaHei';max-width:1050px;margin:40px auto;padding:0 20px}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd}}pre{{white-space:pre-wrap}}</style><h1>显微图像计数与测量</h1><p>检测对象数：{len(table)}。此结果需要人工检查；荧光核染图像显示的是细胞核。</p><img src="data:image/png;base64,{base64.b64encode(png).decode()}"><h2>逐对象测量</h2>{table.to_html(index=False)}<h2>运行配置</h2><pre>{html.escape(json.dumps(meta,ensure_ascii=False,indent=2))}</pre>'''
    files={'overlay_numbered.png':png,'binary_mask.png':image_bytes(Image.fromarray((labels>0).astype(np.uint8)*255)),
           'instance_labels.tif':image_bytes(Image.fromarray(labels.astype(np.int32)),format='TIFF'),
           'measurements.csv':table.to_csv(index=False).encode('utf-8-sig'),
           'manifest.json':json.dumps(meta,ensure_ascii=False,indent=2).encode(),'report.html':report.encode()}
    files['checksums.json']=json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in files.items()},indent=2).encode()
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for n,v in files.items():z.writestr(n,v)
    return buf.getvalue()
