#!/usr/bin/env python

from __future__ import print_function

import logging

import pathomics
from pathomics import featureextractor
import pandas as pd

import multiprocessing
from multiprocessing import Pool, Manager
from itertools import repeat

# Get the Pypathomics logger (default log-level = INFO)
logger = pathomics.logger
logger.setLevel(
    logging.DEBUG
)  # set level to DEBUG to include debug log messages in log file

# Set up the handler to write out all log entries to a file
handler = logging.FileHandler(filename='testLog.txt', mode='w')
formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Define settings for signature calculation
# These are currently set equal to the respective default values
settings = {}


def run(imageName, maskName):
    # Initialize feature extractor
    extractor = featureextractor.PathomicsFeatureExtractor(**settings)

    # Disable all classes except histoqc
    extractor.disableAllFeatures()

    # Only enable morph in nuclei
    extractor.enableFeaturesByName(
        firstorder=[],
        glcm=[],
        glrlm=[],
        graph=[],
        morph=[],
        cgt=[],
        clustergraph=[],
        flock=[],
        haralick=[],
    )

    featureVector = extractor.execute(imageName, maskName)
    return featureVector


if __name__ == '__main__':
    img_file = './data/image1.png'
    mask_file = './data/image1_mask.png'
    featureVectors = run(img_file, mask_file)
    for feature_name, feature_value in featureVectors.items():
        print(feature_name, feature_value)
