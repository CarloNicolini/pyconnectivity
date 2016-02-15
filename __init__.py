#    This file is part of pyConnectivity
#
#    pyConnectivity is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 2 of the License, or
#    (at your option) any later version.
#
#    pyConnectivity is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with pyConnectivity. If not, see <http://www.gnu.org/licenses/>.
#
#    Copyright 2014 Carlo Nicolini <carlo.nicolini@iit.it>

"""
A package to analyze brain connectivity datasets
"""

#"information","utils","visualization"]

# Standard Python imports
from __future__ import division
import sys
import os
import subprocess
import copy
import types
import array
import logging

from itertools import count
from collections import defaultdict
from collections import Counter

# External imports, needed
import networkx as nx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import utils
import statistic
import information
import community
import visualization
import bct
import surprise
import fagso

__author__ = 'Carlo Nicolini'
__version__ = '0.2-alpha'
__all__ = ["utils", "statistic", "community",
           "information", "visualization", "bct", "surprise"]

logging.getLogger().setLevel(logging.INFO)
