import matplotlib.pyplot as plt
import cv2
import os
from os.path import join, basename
from _collections import deque
from laneDetection import find_lanes


if __name__ == '__main__':

    resize_h, resize_w = 540, 960

    verbose = False
    if verbose:
        plt.ion()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

    # test on images
    test_images_dir = join('LinesDetecting\\data', 'test_images')
    # print(test_images_dir)
    test_images = [join(test_images_dir, name) for name in os.listdir(test_images_dir)]
    # print(test_images)
    for test_img in test_images:
        print('Processing image: {}'.format(test_img))
        
        out_path = join('out', 'images', basename(test_img))
        in_image = cv2.imread(test_img)
        out_image = find_lanes(in_image)
        # cv2.imwrite(out_path, cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR))  
        cv2.imshow('1',in_image)
        cv2.imshow('2',out_image)
        cv2.waitKey()    
    if verbose:
            cv2.imshow('1',in_image)
            cv2.waitKey()
            cv2.imshow('2',out_image)
            cv2.waitKey()
            plt.close('all')
    cv2.destroyAllWindows()
