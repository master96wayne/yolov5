import torch
# from torch import nn
import os

# ckpt = torch.load("yolov5_revset_283.pt")  # applies to both official and custom models

# torch.save(ckpt, "updated-model.pt")


# Model
# model = torch.hub.load('ultralytics/yolov5', 'custom',
                            # path=os.path.join(os.getcwd(), 'updated-model.pt'))  # local model
model = torch.hub.load(os.getcwd(), 'custom', path='updated-model.pt', source='local')  # local repo

model.conf = 0.294  # NMS confidence threshold
model.visualize = True
# print(model.FILE)
# print(model.ROOT)
# print(model.name)
# print(model.project[0])

# Set feature visualization argument
# model.model[-1].export = True
model.model.model.model = model.model.model.model[:-1]

# results = model(im, size=320)  # custom inference size


# Images
imgs = ['C:\\Users\\leofi\\OneDrive - Universidade de Lisboa\\Documents\\GitHub\\masters\\Data\\fc2015_yolov5_640\\test\\images\\0b21f0579d247c855e05405d3ed805c1-201205251240_frame4_jpg.rf.fd50664014e6077f70db8746268457c7.jpg']  # batch of images

# Inference
results = model(imgs)

# newmodel = model
# newmodel.fc = torch.nn.Sequential()

# print(newmodel)
# features = newmodel(imgs)

# # Print feature map shapes
# for i, fm in enumerate(feature_maps):
    # print(f"Feature Map {i + 1}: {fm.shape}")

# Results
results.print()
print(results.shape)
results.show()  # or .show()
# features.save()

print(results.xyxy[0])  # img1 predictions (tensor)
print(results.pandas().xyxyn[0])  # img1 predictions (pandas)
# print(features)