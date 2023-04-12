import numpy as np
import os


epoch = 0
inference_file = os.path.join(os.path.dirname(__file__), 'best-{}.pth'.format(epoch))
print(inference_file)
