# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 21:20:56 2020

@author: Theodore G C 
"""
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops

def hist_3d(points,bin_n,min_hist=-255,max_hist=255):
    array_edges = np.array([
            [max_hist,max_hist,max_hist],
            [max_hist,min_hist,max_hist],
            [max_hist,max_hist,min_hist],
            [max_hist,min_hist,max_hist],
            [min_hist,max_hist,max_hist],
            [min_hist,max_hist,min_hist],
            [min_hist,min_hist,max_hist],
            [min_hist,min_hist,min_hist],
            ])
    
    points = np.append(points,array_edges,axis = 0)
    
    hist, binedges = np.histogramdd(points, normed=False, bins = bin_n)
    
    
    bin_n_min = bin_n - 1
    hist[0,0,0] = hist[0,0,0] - 1
    hist[0,0,bin_n_min] = hist[0,0,bin_n_min] - 1
    hist[0,bin_n_min,0] = hist[0,bin_n_min,0] - 1
    hist[0,bin_n_min,bin_n_min] = hist[0,bin_n_min,bin_n_min] - 1
    hist[bin_n_min,0,0] = hist[bin_n_min,0,0] - 1
    hist[bin_n_min,0,bin_n_min] = hist[bin_n_min,0,bin_n_min] - 1
    hist[bin_n_min,bin_n_min,0] = hist[bin_n_min,bin_n_min,0] - 1
    hist[bin_n_min,bin_n_min,bin_n_min] = hist[bin_n_min,bin_n_min,bin_n_min] - 1
    
    return hist, binedges

class lesion:
     def __init__(self, image_address,remove_edge_px=5,hair_remover_px=3):
          
          if hair_remover_px<=0:
               raise ValueError("remove_edge_px must be positive or None")
          
          self.img = cv2.imread(image_address)
          self.imy,self.imx,_ = np.shape(self.img)
          
          if remove_edge_px!=None:
               #Removing edge pixels because usually there are borders from the image, just a precaution, 
               self.img = self.img[remove_edge_px:self.imy-remove_edge_px,remove_edge_px:self.imx-remove_edge_px,:]
               
               #Changing the dimension data
               self.imy,self.imx,_ = np.shape(self.img)

          self.gs = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
          
          _,inv_dermascope_border = cv2.threshold(self.gs,25,255,cv2.THRESH_BINARY)
          
          #Smoothing the skin blob
          closing_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
          closing = cv2.morphologyEx(inv_dermascope_border, cv2.MORPH_CLOSE, closing_kernel,iterations = 4)
          
          kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
          borders = cv2.erode(closing,kernel,iterations = 50)
          
          
          _,lesion_dermascope = cv2.threshold(self.gs,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
          
          #REMOVING HAIR MORPH TRANSFORM------------- START
          #Morphologic Transform ------------- START
          kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(hair_remover_px,hair_remover_px))
          erosion = cv2.erode(lesion_dermascope,kernel,iterations = int(min([self.imx,self.imy])/100))
          segment_mask = cv2.dilate(erosion,kernel,iterations = int(min([self.imx,self.imy])/100))
          #Morphologic Transform ------------- End
          #REMOVING HAIR MORPH TRANSFORM------------- END
          
          
          
          self.nmask =  cv2.bitwise_and(borders, segment_mask)
          
          self.contours,hierarchy = cv2.findContours(self.nmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
          
          self.contimg = self.img.copy()
          
          height, width, channels = self.img.shape
          self.all_cnt_bimg = np.zeros((height,width,1), np.uint8)
          cv2.drawContours(self.all_cnt_bimg, self.contours, -1, 255, 1)
          
          if len(self.contours) != 0:
               #find the biggest area
               self.biggest_contour = max(self.contours, key = cv2.contourArea)
                
               # draw in blue the contours that were founded
               cv2.drawContours(self.contimg, self.biggest_contour, -1, 255, 3)
               
                
               
               #Finding center of mass
               M = cv2.moments(self.biggest_contour)
               self.centroid_x = int(M["m10"] / M["m00"])
               self.centroid_y = int(M["m01"] / M["m00"])             

          
               #CREATING BORDER 
               self.square_r = np.maximum(self.imx,self.imy)
               bimg = np.zeros((height,width,1), np.uint8)
               cv2.drawContours(bimg, self.biggest_contour, -1, 255, 1)
               flood_img = bimg.copy()
               maskfill = np.zeros((height+2, width+2), np.uint8)    
               cv2.floodFill(flood_img, maskfill, (self.centroid_x,self.centroid_y), 255)
               self.segment = cv2.bitwise_and(self.img, self.img, mask=flood_img) 

     def assymetry(self):
          rhalf = (self.square_r/2).astype(int)
          
          nc_img = np.zeros((self.square_r,self.square_r,1), np.uint8)
          
          (r_x,r_y),(MA,ma),angle = cv2.minAreaRect(self.biggest_contour)
          
         
          t_cx = self.centroid_x-(self.imx/2)
          t_cy = self.centroid_y-(self.imy/2)
              
          t_cx = r_x-(rhalf)
          t_cy = r_y-(rhalf)
           
          xc,yc,zc = np.shape(self.biggest_contour)        
          nc = np.zeros([xc,yc,zc])
           
          nc[:,0,0] = self.biggest_contour[:,0,0]- t_cx
          nc[:,0,1] = self.biggest_contour[:,0,1]- t_cy
          nc = nc.astype(int)
          
          cv2.drawContours(nc_img, nc, -1, 255, 1)
          maskfill = np.zeros((self.square_r+2, self.square_r+2), np.uint8)    
          cv2.floodFill(nc_img, maskfill, (rhalf,rhalf), 255)  
          rotate_M = cv2.getRotationMatrix2D((rhalf,rhalf),angle,1)            #rotate Matrix
          rotated = cv2.warpAffine(nc_img,rotate_M,(self.square_r,self.square_r))        #rotated image (1 channel)
          
          box = cv2.boxPoints(((rhalf,rhalf),(MA,ma),0))
          box = np.int0(box)
          
          square_x1 = box[1][1]
          square_x2 = box[0][1]
          square_y1 = box[0][0]
          square_y2 = box[2][0]
          
          ROI = rotated[square_x1:square_x2, square_y1:square_y2]
           
          [roiy,roix] = np.shape(ROI)
           
          roix_half = roix/2
          roiy_half = roiy/2
          
          roi_top = ROI[0:(math.floor(roiy_half)),0:roix]    
          roi_bot = ROI[math.ceil(roiy_half):roiy,0:roix]    
          roi_left = ROI[0:roiy,0:math.floor(roix_half)]    
          roi_right = ROI[0:roiy,math.ceil(roix_half):roix]    
          roi_right_flip = cv2.flip(roi_right, 1)
          roi_bot_flip = cv2.flip(roi_bot,0)
          
          overlap_y_1 = roi_top - roi_bot_flip
          overlap_y_2= roi_bot_flip - roi_top
          overlap_y =  cv2.bitwise_and(overlap_y_1, overlap_y_2)
          _,overlap_y = cv2.threshold(overlap_y,0,255,cv2.THRESH_BINARY)
           
          overlap_x = roi_left - roi_right_flip 
           
          overlap_x_1 = roi_left - roi_right_flip
          overlap_x_2= roi_right_flip - roi_left
          overlap_x =  cv2.bitwise_and(overlap_x_1, overlap_x_2)
          _,overlap_x = cv2.threshold(overlap_x,0,255,cv2.THRESH_BINARY)
          
          sum_overlap_x = sum(sum(np.int64(overlap_x)))/255
          sum_overlap_y = sum(sum(np.int64(overlap_y)))/255
          sum_roi  = sum(sum(np.int64(ROI)))/255
          
          AI = sum([sum_overlap_x, sum_overlap_y])/(2*sum_roi)

          return AI
          

#Test Code