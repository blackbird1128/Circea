import cache
import search
import clip
import torch
from operator import itemgetter
import os
import argparse
import pyfiglet



lines = """\
54126=TEST-BJR-01 Voici une phrase
2100=YOU-SLT-OK Lioni Di Col
12457=SRT-TUT-ALO_FO Code For
"""

for line in lines.splitlines():
    value = line.split("=")[1]
    value = value.split()[0]
    print(value)




def display_cache_information(cache_data):
    number_image = len(cache_data)
    plural = "s" if number_image > 0 else ""
    print(f"currently {number_image} image{plural} are stored")


print("\t\n",end="")
pyfiglet.print_figlet("Circea")
print("   A local image search engine \n\n")
parser = argparse.ArgumentParser(prog='circea')
parser.add_argument("-cache" , help="cache a specific folder" )

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
if(device == "cuda"):
    print("Working on GPU")
else:
    print("Working on CPU : Might be slower")
print("loading model ...")
print("""
Help:
list of commands:
cache: [path to a folder ]  recursively index every image in this folder to make them available for search.
k: [number]  change the number of top results.
any text: look for images corresponding to the text
""")
model, preprocess = clip.load("ViT-B/32", device=device)
if not os.path.exists("cache"):
    os.mkdir("cache")

if not os.path.exists("cache/cache_image.cache"):
    cache.save_cache("cache/cache_image.cache", [])
    print("You don't have any image in cache\ntype cache: [path to folder ] to cache every image in this folder to be able to search them \n ")

cache_data = cache.load_cache(r"cache/cache_image.cache")
display_cache_information(cache_data)

num_top_result = 5

while True:
    try:
        text_search = input(">")
        results = []
        if text_search.startswith("cache:"):
            args = text_search.split()
            if len(args) > 1:
                path = text_search.split()[1]
            else:
                print("invalid command\ncache: [path]")
            if not os.path.exists(path):
                print("The path you entered isn't correct")
                continue
            abs_path = os.path.abspath(path)
            print("adding " , abs_path , " images to cache ")
            cache.index_frames_batch(abs_path , "cache/cache_image.cache", 32)
            cache_data = cache.load_cache(r"cache/cache_image.cache")
            display_cache_information(cache_data)
        elif text_search.startswith("k:"):
            temp = num_top_result
            try:
                args = text_search.split()
                if len(args) > 1: 
                    temp = int(args[1])
                    temp = num_top_result
                else:
                    print("invalid command\nk: [number]")
            except ValueError as e:
                temp = num_top_result
        elif(text_search.endswith(".jpg") or text_search.endswith(".png")):
            results = search.search_by_image(model ,preprocess, text_search,cache_data  , device , num_top_result )
        else:
            if len(text_search) > 0:
                results = search.search_in_cache(model , text_search,cache_data,device,num_top_result)
        results = sorted(results , reverse=True , key=itemgetter(0))
        for i, result in enumerate(results):
            print(f"{i+1}) image: {result[1]} | confidence : {result[0]}")
    except KeyboardInterrupt as e:
        print("bye")
        exit()