import os
import sys
import torch
from PIL import Image
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'circea')))
import circea.cache
import circea.filter
import circea.cli

def create_test_images(num_images):
    """this function create {num_images} test images and return the number
       of image created """
    if not os.path.exists("tests"):
        os.mkdir("tests")
    if not os.path.exists("tests/images"):
        os.mkdir("tests/images")
    for i in range(num_images):
        im = Image.new(mode = "RGB", size = (200,200), color= (random.randint(0, 255) ,random.randint(0, 255) , random.randint(0, 255)  ))
        im.save(f"tests/images/img{i}.png")
    return num_images

def test_caching_date():
    filename = "test.pickle"
    data = [1,"2",3,4,5,6]
    circea.cache.save_cache(filename , data )
    data_retrieved = circea.cache.load_cache(filename)
    assert data_retrieved == [1,"2",3,4,5,6]
    os.remove(filename)

def test_adding_to_cache():
    filename = "test_add.pickle"
    data =torch.tensor([1,2,3,4])
    circea.cache.save_cache(filename , data ,compressed=False)
    new_data = torch.tensor([5,6,7,8])
    circea.cache.add_to_cache(filename , new_data,compressed=False)
    data_retrieved = circea.cache.load_cache(filename,compressed=False)
    assert torch.equal(data_retrieved , torch.tensor([1,2,3,4,5,6,7,8]))
    os.remove(filename)

def test_chunks():
    list_data = [1,2,3,4,5,6,7,8,9]
    list_chunk = list(circea.cache.chunks(list_data,3))
    assert len(list_chunk) == 3
    assert len(list_chunk[0]) == 3


def test_existing_filter():
    list_existing_files = [f"img{i}" for i in range(10)]
    list_new_files = ["img1" , "img4" , "img42"]
    list_new_file_filtered = ["img42"]
    assert circea.filter.filter_already_existing(list_existing_files , list_new_files ) == list_new_file_filtered

def test_parsing_args():
    text_input = "command: arg1 , arg2 , arg3"
    assert circea.cli.get_command_args(text_input , ",") == ["arg1" , "arg2" , "arg3"]
    text_input = "command: "
    arg_list = [f"arg{i}" for i in range(100)]
    text_inputs_args_str = " , ".join(arg_list)
    text_input += text_inputs_args_str
    assert circea.cli.get_command_args(text_input , ",") == arg_list


