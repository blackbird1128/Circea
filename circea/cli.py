

def display_cache_information(cache_data):
    number_image = len(cache_data)
    plural = "s" if number_image > 0 else ""
    print(f"currently {number_image} image{plural} are stored")

def ask_for_confirmation(message , accepted_yes = ["yes", "y" ] , accepted_no = ["no" , "no"] , check_case = False ):
    answer = None
    while answer == None:
        temp_answer = input(message)
        if temp_answer in accepted_yes:
            answer = True
        if temp_answer in accepted_no:
            answer = False
    return answer