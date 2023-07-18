# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import multiprocessing
from multiprocessing import Process
import os
import time
from datetime import timedelta
import random
import numpy

import hidet
import hidet.distributed

def test_priority():
    """
    a -----> a*2 (+5) ----+-----> (a*2)*2 ----> (a*2)*2*2
                          |
                          v
    b -----> b*2 (+20) ---+----> (a*2) * (b*2) (+3) -->
    """
    a = hidet.symbol([2, 2])
    b = hidet.symbol([2, 2])
    a2 = a * 2
    a2.op.attrs['priority'] = 5
    b2 = b * 2
    b2.op.attrs['priority'] = 20
    a22 = a2 * 2
    a222 = a22 * 2
    a2b2 = a2 * b2
    a2b2.op.attrs['priority'] = 3

    g = hidet.trace_from([a222], [a, b], [a2b2.op])
    nodes = g.nodes
    assert nodes.index(a2b2.op) > nodes.index(a2.op) > nodes.index(b2.op)
    

def test_priority_conflict():
    """
    a -----> a*2 (+1) ----+-----> (a*2)*2 ----> (a*2)*2*2
                          |
                          v
    b -----> b*2 (+20) ---+----> (a*2) * (b*2) (+3) -->
    """
    a = hidet.symbol([2, 2])
    b = hidet.symbol([2, 2])
    a2 = a * 2
    a2.op.attrs['priority'] = 1
    b2 = b * 2
    b2.op.attrs['priority'] = 20
    a22 = a2 * 2
    a222 = a22 * 2
    a2b2 = a2 * b2
    a2b2.op.attrs['priority'] = 3

    error = False
    try:
        g = hidet.trace_from([a222], [a, b], [a2b2.op])
    except ValueError:
        error = True
    assert error

def test_dependents():
    x = hidet.symbol([2, 2])
    y = hidet.symbol([2, 2])
    z = hidet.symbol([2, 2])
    z2 = z * 2
    x2 = x * 2
    xy = x * y
    g = hidet.trace_from([xy], [x, y, z], [x2.op, z2.op])
    assert z2.op in g.nodes
    assert x2.op in g.nodes

if __name__ == '__main__':
    test_priority()
    test_priority_conflict()
    test_dependents()