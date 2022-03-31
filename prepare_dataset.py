import rasterio as rio
from osgeo import gdal
from osgeo import ogr
from rasterio.enums import Resampling
import fiona
import shapely
import geopandas as gpd
from rasterio.mask import mask
from shapely.geometry import box
import os
import tifffile
import numpy as np
from PIL import Image
from patchify import patchify
import matplotlib.pyplot as plt
import cv2

def get_scale_factor(img,target_pixel):
    gt = img.get_transform()
    pixelSizeX = gt[1]
    pixelSizeY = -gt[5]
    return pixelSizeX/target_pixel,pixelSizeY/target_pixel


def merged_img(orto,dsm,target_pixel,output):
    x_s,y_s = get_scale_factor(orto,target_pixel)

    data_orto = orto.read(            
        out_shape=(
            orto.count,
            int(orto.height*y_s),
            int(orto.width*x_s)
        ),
        resampling=Resampling.bilinear
    )

    data_dsm = dsm.read(            
        out_shape=(
            dsm.count,
            int(dsm.height*y_s),
            int(dsm.width*x_s)
        ),
        resampling=Resampling.bilinear
    )

    transform = orto.transform * orto.transform.scale(
        (orto.width / data_orto.shape[2]),
        (orto.height / data_orto.shape[1])
    )
    

    merge = rio.open(output,'w',driver='Gtiff', 
                        width=data_orto.shape[2],height=data_orto.shape[1],
                        count=4,
                        crs=orto.crs, 
                        transform =transform, 
                        dtype = 'float32'
                        )
    
    merge.write(data_orto[0],1)
    merge.write(data_orto[1],2)
    merge.write(data_orto[2],3)
    merge.write(data_dsm[0],4)

    merge.close()



def polyline_to_polygon(vec_path,output):
    poly_ver = []

    vec_ds = ogr.Open(vec_path)
    lyr = vec_ds.GetLayer() 

    for i in lyr:
        geom = i.GetGeometryRef()
        points = geom.GetPointCount()
        one = []
        for j in range(points):
            x, y, z = geom.GetPoint(j)
            one.append((x,y,z))
        poly_ver.append([one])

    schema_props = [('Name','str')]

    feature = {
        "geometry": {"type": "MultiPolygon", "coordinates": poly_ver},
        "properties": {'Name':'ML'},
    }

    with fiona.open(
        output,
        "w",
        driver="ESRI Shapefile",
        schema={"geometry": "3D MultiPolygon", "properties": schema_props},
    ) as collection:
        collection.write(feature)


def poly_to_raster(vec_path,pixelSizeX,pixelSizeY,output=''): # tworzy raster z poligonów lub polilini i wypełnia background zerami

    vec_ds = ogr.Open(vec_path) 
    lyr = vec_ds.GetLayer() 
    source_srs = lyr.GetSpatialRef()
    x_min, x_max, y_min, y_max = lyr.GetExtent()
    x_s = pixelSizeX
    y_s = pixelSizeY
    NoData_value = 0

    x_res = int((x_max - x_min) / x_s)
    y_res = int((y_max - y_min) / y_s)

    target_ds = gdal.GetDriverByName('GTiff').Create(output, x_res, y_res, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((x_min, x_s, 0, y_max, 0, -y_s))
    band = target_ds.GetRasterBand(1)
    band.Fill(NoData_value)

    gdal.RasterizeLayer(target_ds, [1], lyr, burn_values=[1])

    return x_res,y_res


def get_dxf_extent(dxf_file): 
    vec_ds = ogr.Open(dxf_file) 
    lyr = vec_ds.GetLayer() 
    source_srs = lyr.GetSpatialRef()
    x_min, x_max, y_min, y_max = lyr.GetExtent()
    return x_min, x_max, y_min, y_max

def getFeatures(gdf): 
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]


def clip_raster_by_dxf(img,dxf,x_res,y_res,output):

    crs = img.crs
    
    x_min, x_max, y_min, y_max = get_dxf_extent(dxf)
    bbox = box(x_min, y_min, x_max, y_max)
    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=crs)
    coords = getFeatures(geo)

    out_img, out_transform = mask(dataset=img, shapes=coords, crop=True)

    clip = rio.open(output,'w',driver='Gtiff', 
                        width=x_res,height=y_res,
                        count=4,
                        crs=img.crs, 
                        transform =out_transform, 
                        dtype = 'float32'
                        )
    
    clip.write(out_img)

def path_to_files(nazwa_data):
    f = nazwa_data
    for p in os.listdir(f):
        if '.dxf' in p:
            vec_dxf = f+'/'+p
    d = f+'/'+'3_dsm_ortho/1_dsm'
    o = f+'/'+'3_dsm_ortho/2_mosaic'
    for p in os.listdir(d):
        if '.tif' in p:
            dsm_path = d+'/'+p
    for p in os.listdir(o):
        if '.tif' in p:
            ortho_path = o+'/'+p  

    return vec_dxf,dsm_path,ortho_path


def patch(img_dir,mask_dir,img_channels_count,patch_size,img_save_dir,mask_save_dir): # merge into one with check_useful
    
    for path, subdirs, files in os.walk(img_dir): 
    dirname = path.split(os.path.sep)[-1]
    images = os.listdir(path)  
    for i, image_name in enumerate(images):  
        if image_name.endswith(".tif"):
            image = tifffile.imread(path+"/"+image_name)
            SIZE_X = ((image.shape[1])//patch_size)*patch_size
            SIZE_Y = ((image.shape[0])//patch_size)*patch_size
            image = image[:SIZE_Y,:SIZE_X,:]

            patches_img = patchify(image, (patch_size, patch_size, img_channels_count), step=patch_size)

            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    
                    single_patch_img = patches_img[i,j,:,:]
                    single_patch_img = single_patch_img[0]                           
                    
                    tifffile.imwrite(img_save_dir+image_name+"patch_"+str(i)+str(j)+".tif", single_patch_img)

    
    for path, subdirs, files in os.walk(masks_dir): 
    dirname = path.split(os.path.sep)[-1]
    masks = os.listdir(path)  
    for i, mask_name in enumerate(masks):  
        if mask_name.endswith(".tif"):
            mask = tifffile.imread(path+"/"+mask_name)
            mask = np.reshape(mask,[mask.shape[0],mask.shape[1],1])
            SIZE_X = ((mask.shape[1])//patch_size)*patch_size
            SIZE_Y = ((mask.shape[0])//patch_size)*patch_size
            mask = mask[:SIZE_Y,:SIZE_X,:]

            patches_msk = patchify(mask, (patch_size, patch_size, 1), step=patch_size)

            for i in range(patches_msk.shape[0]):
                for j in range(patches_msk.shape[1]):
                    
                    single_patch_msk = patches_msk[i,j,:,:]
                    single_patch_msk = single_patch_msk[0]                           
                    
                    tifffile.imwrite(mask_save_dir+mask_name+"patch_"+str(i)+str(j)+".tif", single_patch_msk)

def check_useful(img_dir,mask_dir,img_save_dir,mask_save_dir,min_cover=0.05):
    img_list = os.listdir(img_dir)
    msk_list = os.listdir(mask_dir)

    for img in range(len(img_list)):
        img_name=img_list[img]
        mask_name = msk_list[img]
    
    temp_image=tifffile.imread(train_img_dir+img_list[img])
    temp_mask=tifffile.imread(train_mask_dir+msk_list[img])

    val, counts = np.unique(temp_mask, return_counts=True)

    if (1 - (counts[0]/counts.sum())) > 0.05:
        print("Save Me")
        tifffile.imwrite(img_save_dir+img_name, temp_image)
        tifffile.imwrite(mask_save_dir+mask_name, temp_mask)


