import pytest
import os
import sys
import torch
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import circea.cache

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







