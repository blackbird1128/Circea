import imagehash
from PIL import Image  
import re
import random
import string
import os

def filter_repetitive_image(list_img_path , hash_treshold ):
    """take a list of img_path and return a set of image similar to other images """
    files = sorted(list_img_path, key=lambda x:float(re.findall(r"(\d+)",x)[0]))
    files_hash = []
    for file_d in files:
        files_hash.append(imagehash.average_hash(Image.open(file_d)))
    len_files = len(files)
    repetitive_images = []
    for i in range(len_files-1):
        hash_img = files_hash[i]
        next_hash = files_hash[i+1]
        diff_hash = abs(hash_img - next_hash)
        if diff_hash < hash_treshold:
            repetitive_images.append(files[i])       
    return set(repetitive_images)

def filter_repetion_gif(img_path , hash_treshold):
    """take a path to a gif file and return a list of index of frames similar to other frame in the gif"""
    im = Image.open(img_path)
    list_frame_ignore = []
    list_hashes = []
    n_frames = im.n_frames
    for i in range(n_frames):
        im.seek(i)
        list_hashes.append(imagehash.average_hash(im))
    
    for i in range(len(list_hashes)-1):
        diff_hash = abs(list_hashes[i]  - list_hashes[i+1])
        if diff_hash < hash_treshold:
            list_frame_ignore.append(i)
    return list_frame_ignore

def remove_repetitive_image(list_file ):
    """remove the duplicate in a list of files """
    return list(set(list_file))

def filter_already_existing(list_file , new_file_list):
    """return a list of file minus the one already in list_file"""
    return list(set(new_file_list) - set(list_file)  )

def filter_folder(list_file, target_folder):
    """filter files with a path containing the target folder
       this function take a list of absolute path for list_file
    """
    target_folder = os.path.abspath(target_folder)
    filtered_files = []
    for  file in list_file:
        if file.startswith(target_folder):
            filtered_files.append( file)
    return filtered_files


def filter_file_data(list_file_data , target_folder):
    """filter (data , file ) list , return only the tuples with the files contained in target_folder or
       a target_folder subdirectory  """
    target_folder = os.path.abspath(target_folder)
    filtered_files = []
    for data , file in list_file_data:
        if file.startswith(target_folder):
            filtered_files.append((data , file))
    return filtered_files

def generate_easy_id():
    """generate an easy to remember id"""
    id_part = ['apple','orange','banana','mango','strawberry', 'watermelon', 'coconut' , 'pear' , 'kiwi' , 'fig' , 'lemon'  , 'plum' , 'cherry' , 'lime' , 'raspberry','avocado']
    rdm_post_id = [random.choice(string.ascii_letters + string.digits) for n in range(5)]
    rdm_post_str = "".join(rdm_post_id)
    return "_".join(random.choices(id_part, k=3)) + "_" + rdm_post_str
