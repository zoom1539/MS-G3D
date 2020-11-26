import xml.etree.ElementTree as ET
import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse

z = np.array([0.5773,0.5773,-0.5773])
x = np.array([0.707,-0.707,0])

print(np.cross(z, x))