import cv2
import numpy as np

def apply_mask(edges):
    mask = np.zeros_like(edges)
    height, width = mask.shape
    vertices = np.array([[(0,height), (width,height), (width/2,height*11/20)]],dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    important_edges = cv2.bitwise_and(mask,edges)
    return important_edges

def hough_lines(edges, img):
    # first line
    lines = cv2.HoughLines(edges,1,np.pi/180,50)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
    x1old = x1
    x2old = x2
    y1old = y1
    y2old = y2

    mask = np.zeros_like(edges)
    height, width = mask.shape
    if (y2 - y1) / (x2 - x1) > 0:
        vertices = np.array([[(0,height), (width/2,height), (width/2,height*11/20)]],dtype=np.int32)
    else:
        vertices = np.array([[(width/2,height), (width,height), (width/2,height*11/20)]],dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    edges = cv2.bitwise_and(mask,edges)
    
    # second line
    lines = cv2.HoughLines(edges,1,np.pi/180,50)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

    # drawing lines
    a_old = (y2old - y1old) / (x2old - x1old)
    b_old = y2old - a_old*x2old
    a = (y2 - y1) / (x2 - x1)
    b = y2 - a*x2
    common_x = ((b - b_old) / (a_old - a))
    common_y = a * common_x + b
    common_x = int(common_x)
    common_y = int(common_y)
    if a < 0:
        cv2.line(img,(common_x,common_y),(x2old,y2old),(0,0,255),2)
        cv2.line(img,(x1,y1),(common_x,common_y),(0,0,255),2)
    else:
        cv2.line(img,(x1old,y1old),(common_x,common_y),(0,0,255),2)
        cv2.line(img,(common_x,common_y),(x2,y2),(0,0,255),2)
    return img

def find_lanes(frames):
    #################################################
    # grey scale
    img = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    ##################################################
    # blur 
    # img = cv2.blur(img,(5,5))
    img = cv2.GaussianBlur(img,(7,7),0)
    # img = cv2.medianBlur(img,5)
    #img = cv2.bilateralFilter(img,9,75,75)
    ###################################################
    # edge detection
    edges = cv2.Canny(img,100,200)
    # masking unnecessary edges
    edges = apply_mask(edges)
    # Hough lines, getting lanes
    final = hough_lines(edges, img)
    ####

    return final