
import os
from google import google
import unicodedata
import io
import Levenshtein
import urllib.request
from urllib.error import HTTPError
import re
import time
import string
import sys
from bs4 import BeautifulSoup
from unidecode import unidecode
import inflect


def get_mergable(mergable_list, index, upper_threshold):
    prev_merge = float("Inf")
    next_merge = float("Inf")

    if index > 0 and mergable_list[index-1][1] >= mergable_list[index][0]:
        prev_merge = mergable_list[index][1] - mergable_list[index-1][0]
    if len(mergable_list) > index+1 and mergable_list[index][1] >= mergable_list[index+1][0]:
        next_merge = mergable_list[index+1][1] - mergable_list[index][0]

    if prev_merge == next_merge == float("Inf"):
        not_mergable_item = mergable_list.pop(index)
        return mergable_list, not_mergable_item

    if prev_merge < upper_threshold or next_merge < upper_threshold:
        if prev_merge <= next_merge:
            mergable_list[index-1] = [mergable_list[index-1][0], mergable_list[index][1], mergable_list[index-1][2] + mergable_list[index][2]]
        else:
            mergable_list[index+1] = [mergable_list[index][0], mergable_list[index+1][1], mergable_list[index][2] + mergable_list[index+1][2]]
        del mergable_list[index]
        return mergable_list, None
    else:
        not_mergable_item = mergable_list.pop(index)
        return mergable_list, not_mergable_item

def assert_mergable_sequence(mergable_list):
    start_sequence = [m[0] for m in mergable_list]
    end_sequence = [m[1] for m in mergable_list]
    index_sequence = [m[2][0] for m in mergable_list]

    assert(start_sequence==sorted(start_sequence))
    assert(end_sequence==sorted(end_sequence))
    assert(index_sequence==sorted(index_sequence))

def merge_time_tuple_list(mergable_list, upper_threshold):
    assert_mergable_sequence(mergable_list)

    last_min = None
    current_min = False
    not_mergable_list = []

    while True:
        if last_min == current_min or not mergable_list:
            break
        last_min = current_min
        current_min = min(mergable_list, key=lambda x: x[1] - x[0])
        current_min_index = mergable_list.index(current_min)

        mergable_list, not_mergable_item = get_mergable(mergable_list, current_min_index, upper_threshold)
        assert_mergable_sequence(mergable_list)
        if not_mergable_item:
            not_mergable_list.append(not_mergable_item)

    print(mergable_list)
    print(sorted(not_mergable_list))


merge_time_tuple_list([[0,1,[1]], [0.5, 2.4, [1.5]],[2,3,[2]], [5,6,[3]], [5.5,35.6,[4]], [35.5, 37.1, [5]]], 30)
merge_time_tuple_list([[0,1,[1]], [0.5, 2.4, [1.5]],[2,3,[2]], [5,6,[3]], [5.5,30.6,[4]], [30.5, 37.1, [5]]], 30)
#merge_time_tuple_list([[5,6,[3]], [5.5,6.9,[4]], [6.8, 7.1, [5]]], 30)