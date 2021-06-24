import datetime
import heapq
from PIL import Image  
import clip
import torch

def frame_to_timecode(frame_rate , frame_index ):
    conversion =  datetime.timedelta(seconds=frame_rate*frame_index)
    return str(conversion)

def heapSearch( bigArray, k ):
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

def search_by_image(model  , preprocess, image_path , cache , device, k):
    image =  preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    second_image_feature = model.encode_image(image)
    frame_similarities = []
    with torch.no_grad():
        for feature , frame_data in cache:
            feature /= feature.norm(dim=-1, keepdim=True)
            first_image_feature = feature
            second_image_feature /= second_image_feature.norm(dim=-1 , keepdim=True)
            similarity = first_image_feature @ second_image_feature.T
            frame_similarities.append((similarity[0].item(),frame_data))
    return heapSearch(frame_similarities , k)



def search_in_cache(model , text_input,cache , device , k):  

    text = clip.tokenize([  text_input]).to(device)
    text_features = model.encode_text(text)
    frame_similarities = []
    with torch.no_grad():
        for feature , frame_data in cache:
            feature /= feature.norm(dim=-1, keepdim=True)
            image_features = feature
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = image_features.T @ text_features.T
            frame_similarities.append((similarity[0].item(),frame_data))
    return heapSearch(frame_similarities , k)

def search_in_cache_v2(model , text_input,cache , device , k):        
        text = clip.tokenize([  text_input, "a picture of something ", "a picture"]).to(device)
        text_features = model.encode_text(text)
        frame_similarities = []
        with torch.no_grad():
            for feature , frame_index in cache:
                feature /= feature.norm(dim=-1, keepdim=True)
                image_features = feature
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                frame_similarities.append((similarity[0].item(),frame_index))
        return heapSearch(frame_similarities , k)

