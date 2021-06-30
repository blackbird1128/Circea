import bz2
import torch
import clip
import os
from PIL import Image  
import pickle
import requests
from io import BytesIO


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def load_cache(filename ,compressed=True):
    """retrieve data from the file """
    if compressed:
        data = bz2.BZ2File(filename, 'rb')
        data_r = pickle.load(data)
        data.close()
        return data_r
    else:
        file_d = open(filename, "rb")
        data =  pickle.load(file_d)
        file_d.close()
        return data


def save_cache(filename  , data , compressed =True):
    """ cache data to the specified filename"""
    if compressed:
        sfile = bz2.BZ2File(filename, 'w')
        pickle.dump(data, sfile)
        sfile.close()
    else:
        pickle.dump(data,open(filename, "wb"))

def add_to_cache(filename , data , compressed=True):
    """ add data to a created cache """
    existing_data = load_cache(filename,compressed)
    if type(existing_data) == torch.Tensor:
        data = torch.cat(( existing_data , data ),-1 )
    elif type(existing_data) == list:
        for elem in data:   
            existing_data.append(elem)
        data = existing_data
    if compressed:
        sfile = bz2.BZ2File(filename, 'wb')
        pickle.dump(data, sfile)
        sfile.close()
    else:
        pickle.dump(data,open(filename, "wb"))


def get_files_in_directory(dir ,extensions):
    """ list all files ending with one of the extension in extensions recursively """
    file_list =[]
    for currentpath, folders, files in os.walk(dir):
        for file in files:
            if file[-4:].lower() in extensions:
                file_list.append(os.path.abspath(os.path.join(currentpath, file)))

def get_images_data(list_url , timeout=None ):
    """get image data for every url in the list """
    data_list = []
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    for url in list_url:
        print("adding " , url,  " to list ")
        try:
            r = requests.get(url, headers=headers,timeout=timeout)
        except Exception as e:
            continue
        img_data = BytesIO(r.content)
        data_list.append(img_data)
    return data_list

def index_frames_batch(list_files , cache_name , batch_size ):
    """ index all images listed in list_data : (data in list_data need to be openable by Image.open)"""
    features_array = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device , jit=True)
    i = 0
    batches = list(chunks(list_files , batch_size))
    with torch.no_grad():
        for batch in batches:
            images = []
            for frame in batch:
                try:
                    image = preprocess(Image.open(frame)).unsqueeze(0).to(device)
                    images += image
                except:
                    pass
            images = torch.stack(images)
            image_features = model.encode_image(images)
            for feature , img_name  in zip(image_features, batch):
                features_array.append((feature,  img_name))
            i += len(batch)
    if not os.path.exists(cache_name):
        save_cache(cache_name , features_array )
    else:
        print("adding to cache")
        add_to_cache(cache_name , features_array) 

def index_urls(list_data , list_urls , cache_name , batch_size):
    """ index all url listed in list_urls corresponding to list_data : (data in list_data need to be openable by Image.open)"""
    features_array = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device , jit=True)
    i = 0
    batches = list(chunks(list_data , batch_size))
    url_batches = list(chunks(list_urls , batch_size))
    with torch.no_grad():
        for batch , url  in zip(batches,url_batches):
            images = []
            for frame in batch:
                try:
                    image = preprocess(Image.open(frame)).unsqueeze(0).to(device)
                    images += image
                except:
                    pass
            images = torch.stack(images)
            image_features = model.encode_image(images)
            for feature , img_name  in zip(image_features, url ):
                features_array.append((feature,  img_name))
            i += len(batch)
    if not os.path.exists(cache_name):
        save_cache(cache_name , features_array )
    else:
        print("adding to cache")
        add_to_cache(cache_name , features_array) 



