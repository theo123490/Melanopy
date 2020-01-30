import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from hist_3d import hist_3d
from scipy.misc import imresize

#CHOOSE IMAGE

#img = cv2.imread('RGB color wheel.png')
#img = cv2.imread('redcrcle.png')
#img = cv2.imread('testcrcle.jpg')
#img = cv2.imread('images.jpg')
#img = cv2.imread('Malignant1.jpg')
img = cv2.imread('benign3.jpg')

#img = cv2.imread('img_buat_flow_chart_1.jpg')

imy,imx,imz = np.shape(img)
shortening = 5
img = img[shortening:imy-shortening,shortening:imx-shortening,:]
imy,imx,imz = np.shape(img)
gs = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)



ret1,th1 = cv2.threshold(gs,25,255,cv2.THRESH_BINARY)

ckernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
closing = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, ckernel,iterations = 4)

kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
erosion = cv2.erode(closing,kernel,iterations = 50)

derm_scope = erosion



#REMOVING HAIR FILTERING------------- START

#gs_filter = cv2.medianBlur(gs,21) #median blur
#gs_filter = cv2.GaussianBlur(gs,(31,31),0) #Gaussian Blur
#gs_filter = cv2.blur(gs,(31,31)) #Averaging
#REMOVING HAIR FILTERING------------- END



ret,th = cv2.threshold(gs,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#REMOVING HAIR MORPH TRANSFORM------------- START
#Morphologic Transform ------------- START
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
erosion = cv2.erode(th,kernel,iterations = int(min([imx,imy])/100))
dilation = cv2.dilate(erosion,kernel,iterations = int(min([imx,imy])/100))
#Morphologic Transform ------------- End
#REMOVING HAIR MORPH TRANSFORM------------- END

segment_mask = dilation




nmask =  cv2.bitwise_and(derm_scope, segment_mask)

im2,contours,hierarchy = cv2.findContours(nmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

contimg = img.copy()

height, width, channels = img.shape
all_cnt_bimg = np.zeros((height,width,1), np.uint8)
cv2.drawContours(all_cnt_bimg, contours, -1, 255, 1)



if len(contours) != 0:
    #find the biggest area
    c = max(contours, key = cv2.contourArea)

    # draw in blue the contours that were founded
    cv2.drawContours(contimg, c, -1, 255, 3)
    

    #Finding center of mass
    M = cv2.moments(c)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    t_cx = cx-(imx/2)
    t_cy = cy-(imy/2)
    

    #CREATING BORDER 
    square_r = np.maximum(imx,imy)
    bimg = np.zeros((height,width,1), np.uint8)
    cv2.drawContours(bimg, c, -1, 255, 1)
    flood_img = bimg.copy()
    maskfill = np.zeros((height+2, width+2), np.uint8)    
    cv2.floodFill(flood_img, maskfill, (cx,cy), 255)
    segment = cv2.bitwise_and(img, img, mask=flood_img)
    
    






    
    
#-------------------------------------------------------------- ASSYMETRY START

    rhalf = (square_r/2).astype(int)

    nc_img = np.zeros((square_r,square_r,1), np.uint8)
    
    (r_x,r_y),(MA,ma),angle = cv2.minAreaRect(c)
    
    t_cx = r_x-(rhalf)
    t_cy = r_y-(rhalf)

    xc,yc,zc = np.shape(c)        
    nc = np.zeros([xc,yc,zc])

    nc[:,0,0] = c[:,0,0]- t_cx
    nc[:,0,1] = c[:,0,1]- t_cy
    nc = nc.astype(int)
    
    cv2.drawContours(nc_img, nc, -1, 255, 1)
    maskfill = np.zeros((square_r+2, square_r+2), np.uint8)    
    cv2.floodFill(nc_img, maskfill, (rhalf,rhalf), 255)  
    rotate_M = cv2.getRotationMatrix2D((rhalf,rhalf),angle,1)            #rotate Matrix
    rotated = cv2.warpAffine(nc_img,rotate_M,(square_r,square_r))        #rotated image (1 channel)
    
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
    ret,overlap_y = cv2.threshold(overlap_y,0,255,cv2.THRESH_BINARY)

    overlap_x = roi_left - roi_right_flip 

    overlap_x_1 = roi_left - roi_right_flip
    overlap_x_2= roi_right_flip - roi_left
    overlap_x =  cv2.bitwise_and(overlap_x_1, overlap_x_2)
    ret,overlap_x = cv2.threshold(overlap_x,0,255,cv2.THRESH_BINARY)
    
    sum_overlap_x = sum(sum(np.int64(overlap_x)))/255
    sum_overlap_y = sum(sum(np.int64(overlap_y)))/255
    sum_roi  = sum(sum(np.int64(ROI)))/255
    
    AI = sum([sum_overlap_x, sum_overlap_y])/(2*sum_roi)
    print('Assymetry Index : ' + str(AI))
    print('overlap_x : ' + str(sum_overlap_x))
    print('overlap y : ' + str(sum_overlap_y))
    print('ROI : ' + str(sum_roi))
    
#-------------------------------------------------------------- ASSYMETRY END






#-------------------------------------------------------------- BORDER FD START

    cv2.drawContours(nc_img, nc, -1, 255, 1)
    fract_dim = np.zeros((square_r,square_r,3), np.uint8)
    fract_dim_cont = fract_dim.copy()
    cv2.drawContours(fract_dim_cont, nc, -1, 255, 1)
    fract_dim =fract_dim_cont.copy()
    
    

    divider = 5
    FD_points = 20
    FD_counter = np.zeros(FD_points)
    FD_r = np.zeros([FD_points])
        
    
    
    for n in range (1,FD_points+1):        
        FD_r[n-1] = int(square_r/(divider+(n*1.5)))
        for k in range (int(divider+(n*1.5))):
            for l in range (0,int(divider+(n*1.5))):
                mat_xa = int(((l-1)*FD_r[n-1]))
                mat_xb = int((l*FD_r[n-1]))
                mat_ya = int(((k-1)*FD_r[n-1]))
                mat_yb = int((k*FD_r[n-1]))
                mat = fract_dim_cont[mat_ya:mat_yb , mat_xa:mat_xb,0] 
                value = np.sum(mat)
                value = int(value/255)
                if value > 0 :
                    fract_dim[mat_ya:mat_yb , mat_xa:mat_xb,1] =(50*n)
                    cv2.rectangle(fract_dim,(mat_xa,mat_ya),(mat_xb,mat_yb),[50*n,0,0],3)
                    FD_counter [n-1] = FD_counter[n-1] + 1
    
    FD_log_n = np.log10(1/FD_counter)
    FD_log_r = np.log10(FD_r)    

    fit = np.polyfit(FD_log_n,FD_log_r,1)
    fit_fn = np.poly1d(fit)
    plt.plot(FD_log_n,FD_log_r, 'yo', FD_log_n, fit_fn(FD_log_n), '--k')
    plt.title('FRACTAL DIMENSION')
    plt.ylabel('log box radius')
    plt.xlabel('log 1/amount of box')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('fractal dimension.png')
    plt.show()


    fd = fit_fn[1]
    print('fractal dimension : ' + str(fd))
    
#-------------------------------------------------------------- BORDER FD END
    


#-------------------------------------------------------------- TEXTURE START
    glcm_img = img.copy()
    glcm_rotate_M = cv2.getRotationMatrix2D((r_x,r_y),angle,1)
    glcm_rotated = cv2.warpAffine(glcm_img,glcm_rotate_M,glcm_img.shape[1::-1])     
    
    glcm_mask_img = flood_img.copy()
    glcm_mask_rotated = cv2.warpAffine(glcm_mask_img,glcm_rotate_M,glcm_mask_img.shape[1::-1])     
    
    

    GLCM_segment = cv2.bitwise_and(glcm_rotated, glcm_rotated, mask=glcm_mask_rotated)
    glcm_gs = cv2.cvtColor(GLCM_segment,cv2.COLOR_BGR2GRAY)

    _,glcm_contours,_ = cv2.findContours(glcm_mask_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    glcm_c =max(glcm_contours, key = cv2.contourArea)

    x,y,w,h = cv2.boundingRect(glcm_c)
    # draw the book contour (in green)
    col_roi1 = glcm_gs[ y:y+h, x:x+w]
    col_roi = cv2.resize(col_roi1, (600,400), interpolation = cv2.INTER_AREA)
    glcm = greycomatrix(col_roi, [1], [0, np.pi/2, np.pi*3/4, np.pi/4],levels = 256)
    glcm_mat = glcm[:,:,0,0]
    glcm_mat = glcm_mat.astype('uint8')
    glcm_mat_norm = glcm_mat/np.sum(glcm_mat)
    
    
    GLCM_resize = imresize(glcm_mat, (85,85))

    
    R_col_roi1 = GLCM_segment[ y:y+h, x:x+w ,0]
    G_col_roi1 = GLCM_segment[ y:y+h, x:x+w ,1]
    B_col_roi1 = GLCM_segment[ y:y+h, x:x+w ,2]

    R_col_roi = cv2.resize(R_col_roi1, (600,400), interpolation = cv2.INTER_AREA)
    G_col_roi = cv2.resize(G_col_roi1, (600,400), interpolation = cv2.INTER_AREA)
    B_col_roi = cv2.resize(B_col_roi1, (600,400), interpolation = cv2.INTER_AREA)

    R_glcm = greycomatrix(R_col_roi, [1], [0, np.pi/2, np.pi*3/4, np.pi/4],levels = 256)
    G_glcm = greycomatrix(G_col_roi, [1], [0, np.pi/2, np.pi*3/4, np.pi/4],levels = 256)
    B_glcm = greycomatrix(B_col_roi, [1], [0, np.pi/2, np.pi*3/4, np.pi/4],levels = 256)


    #GLCM PROPERTIES EXTRACTION
    contrast = greycoprops(glcm, 'contrast')
    dissimilarity = greycoprops(glcm, 'dissimilarity')
    homogeneity = greycoprops(glcm, 'homogeneity')
    ASM = greycoprops(glcm, 'ASM')
    energy = greycoprops(glcm, 'energy')
    correlation = greycoprops(glcm, 'correlation')

    R_contrast = greycoprops(R_glcm, 'contrast')
    R_dissimilarity = greycoprops(R_glcm, 'dissimilarity')
    R_homogeneity = greycoprops(R_glcm, 'homogeneity')
    R_ASM = greycoprops(R_glcm, 'ASM')
    R_energy = greycoprops(R_glcm, 'energy')
    R_correlation = greycoprops(R_glcm, 'correlation')

    G_contrast = greycoprops(G_glcm, 'contrast')
    G_dissimilarity = greycoprops(G_glcm, 'dissimilarity')
    G_homogeneity = greycoprops(G_glcm, 'homogeneity')
    G_ASM = greycoprops(G_glcm, 'ASM')
    G_energy = greycoprops(G_glcm, 'energy')
    G_correlation = greycoprops(G_glcm, 'correlation')
    
    B_contrast = greycoprops(B_glcm, 'contrast')
    B_dissimilarity = greycoprops(B_glcm, 'dissimilarity')
    B_homogeneity = greycoprops(B_glcm, 'homogeneity')
    B_ASM = greycoprops(B_glcm, 'ASM')
    B_energy = greycoprops(B_glcm, 'energy')
    B_correlation = greycoprops(B_glcm, 'correlation')


    print('GLCM Contrast : ' + str(contrast))
    print('GLCM Dissimiliartiy : ' + str(dissimilarity))
    print('GLCM homogeneity : ' + str(homogeneity))
    print('GLCM ASM : ' + str(ASM))
    print('GLCM energy : ' + str(energy))
    print('GLCM correlation : ' + str(correlation))

#-------------------------------------------------------------- TEXTURE END





#-------------------------------------------------------------- COLOR START

    # INVERSE SEGMENTATION OF SKIN
    inv_flood_img = cv2.bitwise_not(flood_img)
    inv_segment = cv2.bitwise_and(img, img, mask=inv_flood_img)

    # SPLITTING RGB CHANNEL
    s_R = inv_segment[:,:,2]
    s_G = inv_segment[:,:,1]
    s_B = inv_segment[:,:,0]

    # FIND SKIN AVERAGE VALUE
    ave_s_R = np.sum(s_R)/(np.sum(inv_flood_img)/255)
    ave_s_G = np.sum(s_G)/(np.sum(inv_flood_img)/255)
    ave_s_B = np.sum(s_B)/(np.sum(inv_flood_img)/255)

    print('red skin average val :' + str(ave_s_R))
    print('Green skin average val :' + str(ave_s_G))
    print('Blue skin average val :' + str(ave_s_B))
    
    # CONVERT RGB DATA TYPE
    R_mat = segment[:,:,2].astype('int32')
    G_mat = segment[:,:,1].astype('int32')
    B_mat = segment[:,:,0].astype('int32')

    R_new_val = R_mat - ave_s_R
    G_new_val = G_mat - ave_s_G
    B_new_val = B_mat - ave_s_B

    col_segment = np.zeros([imy,imx,3])
    col_segment[:,:,2] = R_new_val
    col_segment[:,:,1] = G_new_val
    col_segment[:,:,0] = B_new_val
    
    col_segment_point = col_segment.reshape([(imx*imy),3])
    
    points = col_segment_point
    
    hist,binedges = hist_3d(col_segment_point,8)
    
    bg_px = np.sum(inv_flood_img)/255
    hist[0,0,0]=hist[0,0,0] - bg_px
    norm_hist = hist/np.sum(hist)
#-------------------------------------------------------------- COLOR END






# SHOWING IMAGE ALGORITHMS ---------------------------------------------------------------------- START
#Resize settings
resize = True
rheight = 600
rwidth = 800

#Show Image List
show_imagename = ['img',
                  'th',
                  'nmask',
                  'flood img',
                  'blank image',
                  'segment mask',
                  'gs',
                  'segment',
                  'all_cnt_bimg',
                  'erotion',
                  'dilation',
                  'inv_segment',
                  'col_roi',
                  'ROI',
                  'glcm_rotated',
                  'GLCM_segment'
                  ]
show_image = [img,
              th,
              nmask,
              flood_img,
              bimg,
              segment_mask,
              gs,
              segment,
              all_cnt_bimg,
              erosion,
              dilation,
              inv_segment,
              col_roi,
              ROI,
              glcm_rotated,
              GLCM_segment
              ]





#show_imagename = ['img','nmask',  'Contour Image', 'Blank Image', 'floodfill','segment','ROI','rotated','fractal dimension','colored ROI','inverse flood image']
#show_image = [img,nmask, contimg, bimg, flood_img,segment,ROI,rotated,fract_dim,col_roi,inv_flood_img]
n_showimg = len(show_image)


#Image Showing Sequencing
for k in range (0,n_showimg):
    if resize == True:
        cv2.namedWindow(show_imagename[k],cv2.WINDOW_NORMAL)
        cv2.resizeWindow(show_imagename[k],rwidth,rheight)    
    cv2.imshow(show_imagename[k],show_image[k])

# SHOWING IMAGE ALGORITHMS ---------------------------------------------------------------------- END





cv2.waitKey(0)
cv2.destroyAllWindows()
