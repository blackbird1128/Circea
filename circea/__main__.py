import cache
import search
import cli
import clip
import torch
from operator import itemgetter
import os
import pyfiglet


print("\t\n",end="")
pyfiglet.print_figlet("Circea")
print("   A local image search engine \n\n")

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
checkpoint:  display the current checkpoint information
any text: look for images corresponding to the text
""")
model, preprocess = clip.load("ViT-B/32", device=device)
if not os.path.exists("cache"):
    os.mkdir("cache")

if not os.path.exists("cache/cache_image.cache"):
    cache.save_cache("cache/cache_image.cache", [])
    print("You don't have any image in cache\ntype cache: [path to folder ] to cache every image in this folder to be able to search them \n ")

if os.path.exists("cache/checkpoint.cache"):
    print("It look like you have an unsaved caching operation in progress : ")
    check_data = cache.load_cache("cache/checkpoint.cache")
    operation_path = os.path.dirname(check_data[0][0])
    list_files = check_data[0]
    index = check_data[1]
    print("the caching of the files in  :" , operation_path , "was interupted (progression :", round((index / len(list_files)), 2) * 100   ,"%)"  )
    if cli.ask_for_confirmation("Do you want to restart from checkpoint ? " ):
        cache.start_from_checkpoint("cache/checkpoint.cache" , "cache/cache_image.cache" , 32 )
    else:
        print("You can restart from this checkpoint later by starting a caching operation\nat the same location")


cache_data = cache.load_cache(r"cache/cache_image.cache")
cli.display_cache_information(cache_data)

num_top_result = 5
command_dict = {}




while True:
    try:
        text_search = input(">")
        results = []
        if text_search.startswith("webcache:"):
            args = text_search.split()
            if len(args) > 1:
                urls = text_search.split()[1]
                urls_list = urls.split(",")
                list_data = cache.get_images_data(urls_list )
                data = cache.index_urls(list_data , urls_list , r"cache/cache_image.cache" ,8)
            else:
                print("invalid command\n cache: [urls]")
                continue
        elif text_search.startswith("cache:"):
            args = text_search.split()
            if len(args) > 1:
                path = text_search.split()[1]
            else:
                print("invalid command\ncache: [path]")
                continue
            if not os.path.exists(path):
                print("The path you entered isn't correct")
                continue
            abs_path = os.path.abspath(path)
            print("adding " , abs_path , " images to cache ")
            list_files = cache.get_files_in_directory(abs_path , [".png" , ".jpg", ".gif"] )
            cache.index_images_batch(list_files , "cache/cache_image.cache", 32 , 64 , "cache/checkpoint.cache")
            cache_data = cache.load_cache(r"cache/cache_image.cache")
            cli.display_cache_information(cache_data)
        elif text_search.startswith("k:"):
            temp = num_top_result
            try:
                args = text_search.split()
                if len(args) > 1: 
                    temp = int(args[1])
                    print(f"temp: {temp}")
                    num_top_result = temp
                else:
                    print("invalid command\nk: [number]")
            except ValueError as e:
                pass
        elif text_search.endswith(".jpg") or text_search.endswith(".png"):
            text_input_tuple = text_search.partition(" ")
            text_input = text_input_tuple[2] # empty if the separator isn't found / second part of text if the separator is found
            results = []
            if text_input == "":
                print("using search by image ")
                results = search.search_by_image(model ,preprocess, text_search ,cache_data  , device , num_top_result )
                cli.display_result(results)
            else:
                print("using combined search ")
                results = search.combined_search(model ,preprocess, text_search , text_input ,cache_data  , device , num_top_result )
                cli.display_result(results)
        elif text_search.startswith("exit"):
            cli.close_application()
        else:
            if len(text_search) > 0:
                results = search.search_in_cache(model , text_search,cache_data,device,num_top_result)     
                cli.display_result(results)
                results2 = search.search_in_cache_v2(model , text_search,cache_data,device,num_top_result)     
                print("v2")
                cli.display_result(results2)
    except KeyboardInterrupt as e:
        cli.close_application()