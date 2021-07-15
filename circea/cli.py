import os
from operator import itemgetter

def display_cache_information(cache_data):
    number_image = len(cache_data)
    plural = "s" if number_image > 0 else ""
    print(f"currently {number_image} image{plural} are stored")

def display_checkpoint_information(checkpoint_data):
    operation_path = os.path.dirname(checkpoint_data[0][0])
    list_files = checkpoint_data[0]
    index = checkpoint_data[1]
    print("checkpoint for files in the directory: " , operation_path)
    print("number of file(s) saved: " , index + 1)
    print("number of file(s) remaining uncached: " , len(list_files) - (index + 1) )

def ask_for_confirmation(message , accepted_yes = ["yes", "y" ] , accepted_no = ["no" , "no"] , check_case = False ):
    answer = None
    while answer == None:
        temp_answer = input(message)
        if temp_answer in accepted_yes:
            answer = True
        if temp_answer in accepted_no:
            answer = False
    return answer

def display_result(results):
    results = sorted(results , reverse=True , key=itemgetter(0))
    for i, result in enumerate(results):
        print(f"{i+1}) image: {result[1]} | confidence : {result[0]}")


def close_application():
    print("closing Circea ...")
    print("have a nice day!\nbye")
    exit()