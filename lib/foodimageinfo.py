"""
Module for getting image file from ayahoo_test_images folder.

Use images_iter function to get all files infomation in given directory.
Use filter_by_attrs function to categorize image by their attribute.
"""

import os
import re
from itertools import imap

def images_iter(directory):
    """
    Get a iterator that contains instances of jpg file info dict.

    Parameters
    ----------
    directory: full path of a directory

    Returns
    -------
    A iterator of a list of file info dict.
    """
    def _imagefiles():
        "Get all of image files from given directory."
        for subdir in os.listdir(directory):
            subdir_path = os.path.join(os.path.abspath(directory), subdir)
            for f in os.listdir(subdir_path):
                yield os.path.join(os.path.abspath(subdir_path), f)

    def _parse_attribute(filepath):
        "Get image attribute from the path of image file."
        filepath = filepath
        description = re.match(r'(.*)\.jpg',
                               os.path.basename(filepath)).group(1)
        businessid = re.match(r'.*/(.*)$',
                              os.path.dirname(filepath)).group(1)
        return [filepath, description, businessid]

    def _jpg_file_info(attrlist):
        """
        Function for storing jpeg file infos.
        """
        attrs = ["location", "description", "businessid"]
        return dict(zip(attrs, attrlist))

    return (_jpg_file_info(_parse_attribute(f)) for f in _imagefiles())

for i in list(images_iter('/Users/hanyan/Leetcode/yelp_images'))[:20]:
    print i
