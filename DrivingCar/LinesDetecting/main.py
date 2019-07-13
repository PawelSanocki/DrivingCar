import cv2
import os
from os.path import join, basename
from collections import deque
from laneDetection import find_lanes


if __name__ == '__main__':

    resize_h, resize_w = 540, 960

    # test on images
    test_images_dir = join('C:\\Users\\sanoc\\Documents\\repos\\SelfDriving\\DrivingCar\\DrivingCar\\LinesDetecting\\data', 'test_images')
    test_images = [join(test_images_dir, name) for name in os.listdir(test_images_dir)]

    for test_img in test_images:

        print('Processing image: {}'.format(test_img))

        out_path = join('C:\\Users\\sanoc\\Documents\\repos\\SelfDriving\\DrivingCar\\DrivingCar\\LinesDetecting\\data\\out', 'images', basename(test_img))
        in_image = cv2.imread(test_img, cv2.IMREAD_COLOR)
        out_image = find_lanes(in_image)
        cv2.imwrite(out_path, out_image)
        
        cv2.imshow('final',out_image)
        cv2.waitKey()
    cv2.destroyAllWindows()

    # # test on videos
    # test_videos_dir = join('data', 'test_videos')
    # test_videos = [join(test_videos_dir, name) for name in os.listdir(test_videos_dir)]

    # for test_video in test_videos:

    #     print('Processing video: {}'.format(test_video))

    #     cap = cv2.VideoCapture(test_video)
    #     out = cv2.VideoWriter(join('out', 'videos', basename(test_video)),
    #                           fourcc=cv2.VideoWriter_fourcc(*'DIVX'),
    #                           fps=20.0, frameSize=(resize_w, resize_h))

    #     frame_buffer = deque(maxlen=10)
    #     while cap.isOpened():
    #         ret, color_frame = cap.read()
    #         if ret:
    #             color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
    #             color_frame = cv2.resize(color_frame, (resize_w, resize_h))
    #             frame_buffer.append(color_frame)
    #             blend_frame = color_frame_pipeline(frames=frame_buffer, solid_lines=True, temporal_smoothing=True)
    #             out.write(cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR))
    #             cv2.imshow('blend', cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR)), cv2.waitKey(1)
    #         else:
    #             break
    #     cap.release()
    #     out.release()
        # cv2.destroyAllWindows()



