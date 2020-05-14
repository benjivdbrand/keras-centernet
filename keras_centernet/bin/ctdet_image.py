#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import pandas as pd
import os
import math
from tqdm import tqdm
from glob import glob
import sys
sys.path.append('/content/keras-centernet')

from keras_centernet.models.networks.hourglass import HourglassNetwork, normalize_image
from keras_centernet.models.decode import CtDetDecode
from keras_centernet.utils.utils import COCODrawer
from keras_centernet.utils.letterbox import LetterboxTransformer

coco_names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',]  # noqa
camera_matrix = np.array(
    [[2304.5479, 0,  1686.2379],
     [0, 2305.8757, 1354.9849],
     [0, 0, 1]], dtype=np.float32)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--fn', default='train_images/ID_001a12fb2.jpg', type=str)
  parser.add_argument('--output', default='output', type=str)
  parser.add_argument('--inres', default='512,512', type=str)
  args, _ = parser.parse_known_args()
  args.inres = tuple(int(x) for x in args.inres.split(','))
  os.makedirs(args.output, exist_ok=True)
  kwargs = {
    'num_stacks': 2,
    'cnv_dim': 256,
    'weights': 'ctdet_coco',
    'inres': args.inres,
  }
  heads = {
    'hm': 80,  # 3
    'reg': 2,  # 4
    'wh': 2  # 5
  }
  model = HourglassNetwork(heads=heads, **kwargs)
  model = CtDetDecode(model)
  drawer = COCODrawer()
  fns = sorted(glob(args.fn))
  for fn in tqdm(fns):
    img = cv2.imread(fn)
    letterbox_transformer = LetterboxTransformer(args.inres[0], args.inres[1])
    pimg = letterbox_transformer(img)
    pimg = normalize_image(pimg)
    pimg = np.expand_dims(pimg, 0)
    detections = model.predict(pimg)[0]
    
    train_matrix = train_2_matrix(os.path.splitext(os.path.basename(fn))[0])
    #train_matrix = train_2_matrix('ID_679502721')
    x_img = [0] * len(train_matrix[0])
    y_img = [0] * len(train_matrix[0])
    for car in range(len(train_matrix[0])):
      x_img[car] = get_img_coords(train_matrix[4][car], train_matrix[5][car], train_matrix[6][car])[0]
      y_img[car] = get_img_coords(train_matrix[4][car], train_matrix[5][car], train_matrix[6][car])[1]
  
    train_matrix = np.append(train_matrix, [x_img], axis=0)
    train_matrix = np.append(train_matrix, [y_img], axis=0)

    i = 1
    for d in detections:
      completeName = os.path.join('output/', os.path.splitext(os.path.basename(fn))[0] + '_' + str(i) + '.txt')         

      if os.path.exists(completeName):
        os.remove(completeName)

      x1, y1, x2, y2, score, cl = d
      if score < 0.3:
        break
      
      cl = int(cl)
      name = coco_names[cl].split()[-1]
      if name != 'car':
        continue
       
      x1, y1, x2, y2 = letterbox_transformer.correct_box(x1, y1, x2, y2)
      #img = drawer.draw_box(img, x1, y1, x2, y2, cl)

      x_center = (x2-x1)/2+x1
      y_center = (y2-y1)/2+y1
      car_index = 0
      faulty = True
      dist_x = math.inf
      dist_y = math.inf
      for car in range(len(train_matrix[0])):
        if abs(train_matrix[7][car]-x_center) < 10000 / train_matrix[6][car] and abs(train_matrix[8][car]-y_center) < 10000 / train_matrix[6][car]:
          faulty = False
          if abs(train_matrix[7][car]-x_center) < dist_x and abs(train_matrix[8][car]-y_center) < dist_y:
            dist_x = abs(train_matrix[7][car]-x_center)
            dist_y = abs(train_matrix[8][car]-y_center)
            car_index = car
      if(faulty):
        continue

      f = open(completeName,"w+")

      croppedIm = img[int(y1):int(y2), int(x1):int(x2)]
      #out_fn = os.path.join(args.output,  str(i) + '_' + 'ctdet.' + os.path.basename(fn))
      out_fn = os.path.join(args.output,  os.path.splitext(os.path.basename(fn))[0] + '_' + str(i) + '.jpg')
      cv2.imwrite(out_fn, croppedIm)

      f.write(str(x2-x1) + '\n')
      f.write(str(y2-y1) + '\n')
      f.write(str((x2-x1)/(y2-y1)) + '\n')
      f.write(str((x2-x1)*(y2-y1)) + '\n')
      f.write(str(x_center) + '\n')
      f.write(str(y_center) + '\n')
      f.write(str(math.sqrt(math.pow((y2-y1)/2+y1-len(img[0])/2, 2) + math.pow((x2-x1)/2+x1-len(img), 2))) + '\n')
      f.write(str(math.atan2((y2-y1)/2+y1-len(img[0])/2, (x2-x1)/2+x1-len(img))) + '\n')

      if((x2-x1)/2+x1 < len(img)):
        f.write('1\n0')
      else:
        f.write('0\n1')

      f.write('\n\n')
      f.write(str(train_matrix[0][car_index]) + '\n')
      f.write(str(train_matrix[1][car_index]) + '\n')
      f.write(str(train_matrix[2][car_index]) + '\n')
      f.write(str(train_matrix[3][car_index]) + '\n')
      f.write(str(train_matrix[4][car_index]) + '\n')
      f.write(str(train_matrix[5][car_index]) + '\n')
      f.write(str(train_matrix[6][car_index]) + '\n')
      
      i+= 1

    #out_fn = os.path.join(args.output, 'ctdet.' + os.path.basename(fn))
    out_fn = os.path.join(args.output, os.path.basename(fn))
    cv2.imwrite(out_fn, img)
    print("Image saved to: %s" % out_fn)


def get_img_coords(x, y, z):
    p = np.array([x, y, z]).T
    img_p = np.dot(camera_matrix, p)
    img_p[0] /= img_p[2]
    img_p[1] /= img_p[2]
    return img_p[0], img_p[1]


def train_2_matrix(image_id: str):
  train = pd.read_csv("train.csv")

  new_lists = [float(item) for item in train[train["ImageId"] == 'ID_679502721']["PredictionString"].tolist()[0].split()]

  matrix_list = [0]
  for plaats in range(1):
    matrix_list[plaats] = np.zeros((len(new_lists)//7,7))
    for i in range(len(new_lists)//7):
      matrix_list[plaats][i] = new_lists[7*i:7*i+7]
    matrix_list[plaats] = matrix_list[plaats].transpose()
  
  return matrix_list[0]


if __name__ == '__main__':
  main()
  #train_matrix = train_2_matrix('ID_679502721')
  #x_img = [0] * len(train_matrix[0])
  #y_img = [0] * len(train_matrix[0])
  #for car in range(len(train_matrix[0])):
  #  x_img[car] = get_img_coords(train_matrix[4][car], train_matrix[5][car], train_matrix[6][car])[0]
  #  y_img[car] = get_img_coords(train_matrix[4][car], train_matrix[5][car], train_matrix[6][car])[1]
  #train_matrix = np.append(train_matrix, [x_img], axis=0)
  #train_matrix = np.append(train_matrix, [y_img], axis=0)
  #print(train_matrix)
