import datetime
import heapq
from PIL import Image  
import clip
import torch

def frame_to_timecode(frame_rate , frame_index ):
    """convert a frame index and an associated frame rate to an human readable timecode"""
    conversion =  datetime.timedelta(seconds=frame_rate*frame_index)
    return str(conversion)

def heapSearch( bigArray, k ):
    """classic heap search adapted to search a list of tuple """
    heap = []
    for item in bigArray:
        # If we have not yet found k items, or the current item is larger than
        # the smallest item on the heap,
        if len(heap) < k or item[0] > heap[0][0]:
            # If the heap is full, remove the smallest element on the heap.
            if len(heap) == k: heapq.heappop( heap )
            # add the current element as the new smallest.
            heapq.heappush( heap, item )
    return heap

def encode_search_query(search_query , model , device):
  """encode a search query into a CLIP vector"""
  search_query = search_query.strip()
  with torch.no_grad():
    text_encoded = model.encode_text(clip.tokenize([search_query]).to(device))
    text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
  return text_encoded

def compare_vectors(vec , cache , k):
    """compare vectors in cache with vec and return the top K """
    frame_similarities = []
    vec  /= vec.norm(dim=-1 , keepdim=True)
    with torch.no_grad():
        for feature , frame_data in cache:
            feature /= feature.norm(dim=-1, keepdim=True)
            first_image_feature = feature
            similarity = first_image_feature @ vec.T
            frame_similarities.append((similarity[0].item(),frame_data))
    return heapSearch(frame_similarities , k)

def search_by_image(model  , preprocess, image_path , cache , device, k):
    """search in a cache list (list of encoded image ) using an image as the query """
    image =  preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    second_image_feature = model.encode_image(image)
    return compare_vectors(second_image_feature , cache , k )

def search_in_cache(model , text_input,cache , device , k):
    """search in a cache list (list of encoded image ) using text as the query
       and comparing the raw cosine product of each image  """  
    text = clip.tokenize([  text_input]).to(device)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    frame_similarities = []
    with torch.no_grad():
        for feature , frame_data in cache:
            feature /= feature.norm(dim=-1, keepdim=True)
            image_features = feature
            similarity = image_features.T @ text_features.T
            frame_similarities.append((similarity[0].item(),frame_data))
    return heapSearch(frame_similarities , k)

def search_in_cache_v2(model , text_input,cache , device , k):         
        text = clip.tokenize([  text_input, "a picture of something ", "a picture of" , "this feeling" ]).to(device)
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        frame_similarities = []
        with torch.no_grad():
            for feature , frame_index in cache:
                feature /= feature.norm(dim=-1, keepdim=True)
                image_features = feature
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                frame_similarities.append((similarity[0].item(),frame_index))
        return heapSearch(frame_similarities , k)

def combined_search(model  , preprocess, image_path  , text_input , cache , device, k, image_weight = 0.5):
    """search in a cache list (list of encoded image ) using an image and a text as the query """
    image =  preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    second_image_feature = model.encode_image(image)
    second_image_feature /= second_image_feature.norm(dim=-1 , keepdim=True)
    text_vec = encode_search_query(text_input ,model , device )

    combined_feature = text_vec + second_image_feature * image_weight
    combined_feature /= combined_feature.norm(dim=-1 , keepdim=True)
    frame_similarities = []
    with torch.no_grad():
        for feature , frame_data in cache:
            feature /= feature.norm(dim=-1, keepdim=True)
            first_image_feature = feature
            similarity = first_image_feature @ combined_feature.T
            frame_similarities.append((similarity[0].item(),frame_data))
    return heapSearch(frame_similarities , k)


