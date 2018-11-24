# coding: utf-8

import os
import sys
import cv2
from threading import Thread
from timeit import default_timer as timer


class VideoProcess:

    # 動画保存の初期設定を行う
    def __init__(self, src):
        self.src = src
        self.video = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.video.read()
        self.stopped = False
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.frame_width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame_rate = self.video.get(cv2.CAP_PROP_FPS)
        self.frame_count = self.video.get(cv2.CAP_PROP_FRAME_COUNT)

    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.video.read()

    def stop(self):
        self.stopped = True

    # 目的のフレーム番号から指定した秒数だけ抜き出して保存する
    def extract(self, save_dir, target_frame, duration_second):
        if not 0 <= target_frame < self.video.get(cv2.CAP_PROP_FRAME_COUNT):
            raise ValueError('frame index is invalid')
        self.video.set(1, target_frame)
        video_writer = cv2.VideoWriter(
            save_dir + self.src.replace('.avi', '').split('/')[-1] + '_' + str(target_frame) + '.avi',
            self.fourcc,
            self.frame_rate,
            (self.frame_width, self.frame_height)
        )
        for _ in range(duration_second * self.frame_rate):
            is_capturing, frame = self.video.read()
            if is_capturing:
                video_writer.write(frame)
            else:
                print('the end of video')

    def to_frames(self, save_dir='./frames/', image_file='%s.jpg', start=0, end=None):
        """ Split video to frames and save as jpg files.
        """
        os.makedirs(save_dir, exist_ok=True)

        self.video.set(1, start)
        if not end:
            end = int(self.video.get(7))
        if start > end:
            raise ValueError('start(args) must be larger than end(args).')

        for _ in range(start, end):
            frame_id = int(self.video.get(1))
            flag, frame = self.video.read()
            if not flag:
                break
            cv2.imwrite(os.path.join(save_dir, image_file % str(frame_id).zfill(6)), frame)
            sys.stdout.write('\r%s%d%s%d' % ('Saving ... ', frame_id, ' / ', end))
            sys.stdout.flush()

        print('\nDone')

    @property
    def properties(self):
        print('----- Video Properties -----')
        print('Path: ', self.src)
        print('fps: ', self.frame_rate)
        print('frame_count: ', self.frame_count)
        print('frame_height: ', self.frame_height)
        print('frame_width: ', self.frame_width)

        return self.frame_rate, self.frame_count, self.frame_height, self.frame_width

class VideoShow:

    def __init__(self, draw, frame=None):
        self.frame = frame
        self.show_frame = frame
        self.draw = draw
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        self.show_frame = self.draw(self.frame)

    def stop(self):
        self.stopped = True

def threadVideoShow(draw, source=0):

    video_getter = VideoProcess(source).start()
    video_shower = VideoShow(draw, video_getter.frame).start()

    while True:
        if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
            video_getter.stop()
            break

        cv2.imshow("Video", video_shower.show_frame)
        video_shower.frame = video_getter.frame

"""Example

def resize(frame):
    size = (256, 256)
    return cv2.resize(img, size)

threadVideoShow(resize, 'test.avi')

"""