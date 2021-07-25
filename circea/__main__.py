import cache
import search
import cli
import filter
import config
import clip
import torch
import os
import pyfiglet
import json


print("\t\n",end="")
pyfiglet.print_figlet("Circea")
print("   A local image search engine \n\n")
cli.print_help()
device = "cuda" if torch.cuda.is_available() else "cpu"
if(device == "cuda"):
    print("Working on GPU")
else:
    print("Working on CPU : Might be slower")
print("loading model ...")

model, preprocess = clip.load(config.env.model, device=device)
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
        cache.start_from_checkpoint("cache/checkpoint.cache" , "cache/cache_image.cache" , config.env.batch_size , config.env.checkpoint_interval)
    else:
        print("You can restart from this checkpoint later by starting a caching operation\nat the same location")

cache_data = cache.load_cache(r"cache/cache_image.cache")
cache_data = list(set(cache_data))
cache_copy = cache_data
cli.display_cache_information(cache_data)
num_top_result = config.env.top_k

while True:
    try:
        text_search = input(">")
        results = []
        if text_search.startswith("webcache:"):
            cli.cli_cache_urls(text_search)
        elif text_search.startswith("cache:"):
            cache_vector = []
            cache_files = []
            if len(cache_data ) > 0:
                cache_vector , cache_files  = zip(*cache_data)
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
            list_files = filter.filter_already_existing(cache_files , list_files)
            cache.index_images_batch(list_files , "cache/cache_image.cache",config.env.batch_size , config.env.checkpoint_interval , "cache/checkpoint.cache")
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
                print("using search by image :")
                results = search.search_by_image(model ,preprocess, text_search ,cache_data  , device , num_top_result )
                cli.display_result(results)
            else:
                print("using combined search :")
                results = search.combined_search(model ,preprocess, text_search , text_input ,cache_data  , device , num_top_result )
                cli.display_result(results)
        elif text_search.startswith("exit"):
            cli.close_application()
        elif text_search.startswith("clearcache:"):
            if cli.ask_for_confirmation("Are you sure you want to delete the cache ?"):
                os.remove("cache/cache_image.cache")
        elif text_search.startswith("filter:"):
            args = cli.get_command_args(text_search , ",")
            filtered_files = []
            for arg in args:
                filtered_files.extend(filter.filter_file_data(cache_data , arg))
            cache_data = filtered_files
        elif text_search.startswith("unfilter:"):
            cache_data = cache_copy
        elif text_search.startswith("sort:"):
            args = text_search.split(":")
            if len(args) > 1:
                categorie_list = cli.get_command_args(text_search, ",")
                categories_images = search.sort_into_categories(model ,  categorie_list, cache_data , device)
                search_file_name = filter.generate_easy_id() + ".json"
                with open("cache/" + search_file_name , "w") as default_file:
                    default_file.write(json.dumps(categories_images , indent=4))
                    print("results saved in " +  "cache/"  + search_file_name)
            else:
                print("invalid command\n sort: [categories]")
                continue
        else:
            if len(text_search) > 0:
                results = search.search_in_cache(model , text_search,cache_data,device,num_top_result)     
                cli.display_result(results)
                results2 = search.search_in_cache_v2(model , text_search,cache_data,device,num_top_result)     
                print("v2")
                cli.display_result(results2)
    except KeyboardInterrupt as e:
        cli.close_application()