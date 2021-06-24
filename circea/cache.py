import bz2
import torch
import clip
import os
from PIL import Image  
import pickle

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



def index_frames_batch(dir_frames  , cache_name , batch_size ):
    features_array = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device , jit=True)
    i = 0
    file_list =[]
    for currentpath, folders, files in os.walk(dir_frames):
        for file in files:
            if file[-4:].lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
                file_list.append(os.path.abspath(os.path.join(currentpath, file)))
    batches = list(chunks(file_list , batch_size))
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




