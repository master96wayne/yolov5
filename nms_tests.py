import numpy as np
import torch
from utils.general import non_max_suppression
import detect

pred = detect.run()

# prediction = np.random.rand(1,3,80,80,85)
# prediction = torch.from_numpy(prediction).to("cpu")

# print(non_max_suppression(prediction))