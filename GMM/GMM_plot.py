import matplotlib.pyplot as plt
import test_on_train
import test_on_val
import sys

components = sys.argv[1]

test_on_train.begin(components)
test_on_val.begin(components)
