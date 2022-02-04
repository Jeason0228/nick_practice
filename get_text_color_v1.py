import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time


class Get_Text_Color:
    """
    Version 1: accuracy: 126/152 = 82.9%
    Version 2: accuracy: 149/152 = 98%
    """
    def __init__(self) -> None:
        self.n_clusters = 2  # number of cluster to classify colors
        self.upate_threshold = 0.6  # decide if update color based on foreground postprocess
        
        # limit image size
        self.min_border = 20.
        self.max_wh_ratio = 5.
        
        self.square_crop_threshold = 1.5
        self.square_crop = 0.1
    
    def get_color(self, image):
        foregound = 0
        img = self.pre_process(image)
        mask, self.canditates = self.get_mask(img)
        self.background = self.get_background_color(img)
        
        # judge foreground based on squared distance
        abs_square = [np.sum((self.background-f)**2) for f in self.canditates]
        if abs_square[0] < abs_square[1]:
            self.canditates = self.canditates[::-1]
            foregound = 1
        self.foregound = foregound  # foreground label
        
        # erode or dilate
        kernel = np.ones((2,2), np.uint8)
        if foregound:
            mask_2 = cv2.erode(mask, kernel, iterations=1)
        else:
            mask_2 = cv2.dilate(mask, kernel, iterations=1)
        if np.unique(mask_2).shape[0] == 1:
            self.mask = mask
        else:
            self.mask = mask_2
        
        centers, update = self.post_process_foregound(img, self.mask, foregound)
        if update:
            self.canditates[0] = centers
        return self.canditates
    
    def pre_process(self, image):
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image
            
        h, w = img.shape[:2]
        # if min(h, w) > self.min_border:
        ratio = self.min_border / min(h, w)
        _h, _w = int(h*ratio), int(w*ratio)
        img = cv2.resize(img, (_w, _h), interpolation=cv2.INTER_NEAREST)
            
        wh_ratio = max(h/w, w/h)
        if wh_ratio >= self.max_wh_ratio:
            if h > w:
                img = img[:int(w*self.max_wh_ratio)]
            else:
                img = img[:, :int(h*self.max_wh_ratio)]
                
        # if the box is square, then crop the border to get more accureate text
        if wh_ratio <= self.square_crop_threshold:
            h, w = img.shape[:2]
            crop = self.square_crop
            img = img[int(h*crop):int(h*(1-crop)), int(w*crop):int(w*(1-crop))]
        self.img = img
        return img
             
    def get_background_color(self, img):
        # use border color as estimated background color
        h, w = img.shape[:2]
        total_pixel = h*w
        b, g, r = cv2.split(img)
        border = 0.05
        x1, y1 = int(w*border), int(h*border)
        x2, y2 = int(w*(1-border)), int(h*(1-border))
        border_pixel = total_pixel - (y2-y1)*(x2-x1)
        res = []
        for c in [b, g, r]:
            c[y1:y2, x1:x2] = 0
            res.append(int(np.sum(c)/border_pixel))
        return res

    def get_mask(self, img):
        # Use kmeans to cluster color to (foreground, background)
        h, w = img.shape[:2]
        rgbs = img.reshape(-1, 3)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(rgbs)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        mask = labels.reshape(h, w)
        mask = mask.astype(np.uint8)
        return mask, centers
        
    def post_process_foregound(self, img, mask, F_label):
        # After we get foregound, we can get more accurate foreground by kmeans,
        #   in case of art font where there can be addifitonal border for the word.
        update = False
        rgbs = img.reshape(-1, 3)
        mask = mask.reshape(-1, 1)
        inds = np.squeeze(mask == F_label)
        fore = rgbs[inds]
        _total = fore.shape[0]
        kmeans = KMeans(n_clusters=2, random_state=0).fit(fore)
        labels = kmeans.labels_
        _second = np.sum(labels)  # 1
        _first = _total - _second  # 0
        _fb_ratio = float(abs(_first - _second))/_total
        if _fb_ratio >= self.upate_threshold:
            update = True
        centers = kmeans.cluster_centers_
        
        # _tmp = kmeans.cluster_centers_
        # if _first < _second:
        #     _tmp = _tmp[::-1]
        return centers[0] if _first >= _second else centers[1], update


if __name__ == '__main__':
    TextColor = Get_Text_Color()
    
    image = sys.argv[1]  # image folder
    images = [i for i in os.listdir(image) if i.endswith('png')]
    start = time.time()
    for im in images:
        fore_back = TextColor.get_color(os.path.join(image, im))
        img = TextColor.img
        _y, _x = img.shape[:2]
        
        blank = np.ones((_y, _y*2, 3), np.uint8)
        
        for i, bgr in enumerate(fore_back):
            b, g, r= list(map(int, bgr))
            blank[:, i*_y:(i+1)*_y, 0] *= b
            blank[:, i*_y:(i+1)*_y, 1] *= g
            blank[:, i*_y:(i+1)*_y, 2] *= r
        
        b = np.concatenate((img, blank), axis=1)
        mask = TextColor.mask
        mask *= 255
        h, w = mask.shape
        mask = mask.reshape(h, w, 1)
        mask = np.concatenate((mask, mask, mask),axis=-1)
        b = np.concatenate((mask, b), axis=1)
        cv2.imwrite(os.path.join(image, 'crop_color', im), b)
        # cv2.imshow('1.png', b)
        # cv2.waitKey()
    gap = time.time() - start
    print(f'Average speed: {len(images)/gap} image/second')
