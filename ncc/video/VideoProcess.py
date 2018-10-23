# coding: utf-8

import os
import sys
import cv2


def get_property(video_file, print_only=True):
    """ Get video property.
    """
    cap = cv2.VideoCapture(video_file)
    fps = float(cap.get(5))
    frame_count = int(cap.get(7))
    frame_height = int(cap.get(4))
    frame_width = int(cap.get(3))

    if print_only == True:
        print('----- Video Property -----')
        print('Path: ', video_file)
        print('fps: ', fps)
        print('frame_count: ', frame_count)
        print('frame_height: ', frame_height)
        print('frame_width: ', frame_width)
    else:
        return (fps, frame_count, frame_height, frame_width)


def to_frames(video_file, image_dir='./frames/', image_file='%s.jpg', start=0, end=None):
    """ Split video to frames and save as jpg files.
    """
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    cap = cv2.VideoCapture(video_file)
    cap.set(1, start)
    if not end:
        end = int(cap.get(7))
    if start > end:
        raise ValueError('start(args) must be larger than end(args).')
    
    for _ in range(start, end):
        frame_id = int(cap.get(1))
        flag, frame = cap.read()
        if flag == False:
            break
        cv2.imwrite(os.path.join(image_dir, image_file % str(frame_id).zfill(6)), frame)
        sys.stdout.write('\r%s%d%s%d' % ('Saving ... ', frame_id, ' / ', end))
        sys.stdout.flush()
    cap.release()
    print('\nDone')


class VideoProcess:

    # 動画保存の初期設定を行う
    def __init__(self, video_file, target_dir, duration_second):
        self.video_file = video_file
        self.target_dir = target_dir
        self.video = cv2.VideoCapture(video_file)
        self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.frame_width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame_rate = self.video.get(cv2.CAP_PROP_FPS)
        self.duration = duration_second * self.frame_rate

    # 目的のフレーム番号から指定した秒数だけ抜き出して保存する
    def extract(self, target_frame):
        if not 0 <= target_frame < self.video.get(cv2.CAP_PROP_FRAME_COUNT):
            raise ValueError('frame index is invalid')
        self.video.set(1, target_frame)
        video_writer = cv2.VideoWriter(
            self.target_dir + self.video_file.replace('.mp4', '').split('/')[-1] + '_' + str(target_frame) + '.mp4',
            self.fourcc,
            self.frame_rate,
            (self.frame_width, self.frame_height)
        )
        for _ in range(self.duration):
            is_capturing, frame = self.video.read()
            if is_capturing:
                video_writer.write(frame)
            else:
                print('the end of video')
