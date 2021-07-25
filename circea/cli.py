import os
from operator import itemgetter
import cache
import config

def print_help():
    print("""
Help:
list of commands:
cache: [path to a folder ]  recursively index every image in this folder to make them available for search.
k: [number]  change the number of top results.
checkpoint:  display the current checkpoint information.
exit: exit the tool.
any text: look for images corresponding to the text
    """)

def display_cache_information(cache_data):
    """display informations about cache_data"""
    number_image = len(cache_data)
    plural = "s" if number_image > 0 else ""
    print(f"currently {number_image} image{plural} are stored")
 
def display_checkpoint_information(checkpoint_data):
    """display informations about the checkpoint data"""
    operation_path = os.path.dirname(checkpoint_data[0][0])
    list_files = checkpoint_data[0]
    index = checkpoint_data[1]
    print("checkpoint for files in the directory: " , operation_path)
    print("number of file(s) saved: " , index + 1)
    print("number of file(s) remaining uncached: " , len(list_files) - (index + 1) )

def ask_for_confirmation(message , accepted_yes = ["yes", "y" ] , accepted_no = ["N","no" , "No"] , check_case = False ):
    """an helper function to ask for confirmation 
       return True if the message is confirmed, False otherwise"""
    answer = None
    answer_text = "(" + "/".join(accepted_yes) + "|" + "/".join(accepted_no) + ")"
    while answer == None:
        temp_answer = input(message + " " + answer_text + " :")
        if temp_answer in accepted_yes:
            answer = True
        if temp_answer in accepted_no:
            answer = False
    return answer

def display_result(results):
    """display the results of a search , sorted by model confidence"""
    results = sorted(results , reverse=True , key=itemgetter(0))
    for i, result in enumerate(results):
        print(f"{i+1}) image: {result[1]} | confidence : {result[0]}")

def get_command_args(text_query , separator=","):
    args = text_query.partition(":")[2]
    if args == "":
        return []
    args_list = args.split(separator)
    for i  , arg in enumerate(args_list):
        args_list[i] = arg.strip()
    return args_list
    
def cli_cache_urls(text_query ):
    args = text_query.split()
    if len(args) > 1:
        urls = text_query.split()[1]
        urls_list = urls.split(",")
        list_data = cache.get_images_data(urls_list )
        data = cache.index_urls(list_data , urls_list , r"cache/cache_image.cache" ,config.env.batch_size)
    else:
        print("invalid command\n webcache: [urls]")

def close_application(): 
    print("closing Circea ...")
    print("have a nice day!\nbye")
    exit()