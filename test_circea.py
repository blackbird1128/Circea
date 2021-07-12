import os
import sys
import torch
from PIL import Image
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import circea.cache
import circea.filter

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

def test_checkpoint():
    num_images = 5
    create_test_images(num_images)
    list_test_images =circea.cache.get_files_in_directory("tests/images" , [".png"])
    assert len(list_test_images) == num_images
    print("starting to index")
    circea.cache.index_images_batch(list_test_images , "tests/checkpoint_test.cache",  2 , 1 , "tests/checkpoint.check")
    checkpoint_data = circea.cache.load_cache("tests/checkpoint.check")
    print(checkpoint_data)
    list_path = checkpoint_data[0]
    current_index = checkpoint_data[1]
    list_data = checkpoint_data[1]
    assert list_path == list_test_images
    assert current_index == 5







