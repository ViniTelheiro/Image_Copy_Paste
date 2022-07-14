from ast import arg
import cv2
import numpy as np
from random import randint
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1", type=str,
                        help="path of the image that will be cuttet out")
    parser.add_argument("--img2", type=str,
                        help="path of the image that will be pasted")
    parser.add_argument("--output_dir", type=str, default="output/",
                        help="directory that will be saved the tampered image and the mask")
    return parser.parse_args()


def tampering(src, src2):
    img = np.array(cv2.imread(src))
    img2 = np.array(cv2.imread(src2))
    cv2.waitKey(0)
            
    
    (h, w, _) = img.shape
    n = randint(4,6)
    pol = []
    for i in range(0,n):
        x = randint(0,w)
        y = randint(0,h)
        temp = [x,y]
        while temp in pol:
            x = randint(0,w)
            y = randint(0,h)
            temp = [x,y]
        pol.append(temp)
    pol = np.array(pol, np.int32)

    img1_mask = img.copy()
    img1_mask = cv2.cvtColor(img1_mask, cv2.COLOR_BGR2GRAY)
    img1_mask.fill(0)
    _ = cv2.fillPoly(img1_mask, [pol], 255)
    
    roi = img2[np.min(pol[:,1]):np.max(pol[:,1]), np.min(pol[:,0]):np.max(pol[:,0])]
    mask = img1_mask[np.min(pol[:,1]):np.max(pol[:,1]), np.min(pol[:,0]):np.max(pol[:,0])]
    
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
    src1_cut = img[np.min(pol[:,1]):np.max(pol[:,1]), np.min(pol[:,0]):np.max(pol[:,0])]
    
    img2_fg = cv2.bitwise_and(src1_cut, src1_cut,mask= mask)
    
    dst = cv2.add(img1_bg,img2_fg)
    output = img2.copy()
    output[np.min(pol[:,1]):np.max(pol[:,1]), np.min(pol[:,0]):np.max(pol[:,0])] = dst
    
        
    return output, mask
    
if __name__ == "__main__":
    args = get_args()
    tp, gt = tampering(args.img1,args.img2)
    name = os.path.split(args.img2)[-1]
    cv2.imwrite(os.path.join(args.output_dir,name),tp)
    cv2.imwrite(os.path.join(args.output_dir,f"gt_{name}"), gt)