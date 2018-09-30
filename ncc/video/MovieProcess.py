# coding: utf-8

# 動画を特定のフレームから特定の秒数だけ抜き出して、別のフォルダに保存するコード
# TODO : フレームレートやコーデックも引数にする.
# TODO : あり得ないフレーム番号を指定した時に注意文を出るようにする.
# TODO : コマンドライン引数で実行できるようにする.

import cv2


class MovieProcess:

    # 動画保存の初期設定を行う
    def __init__(self, movie_file, target_dir, duration_second):
        self.movie_file = movie_file
        self.target_dir = target_dir
        self.video = cv2.VideoCapture(movie_file)
        self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.frame_width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame_rate = self.video.get(cv2.CAP_PROP_FPS)
        self.duration = duration_second * self.frame_rate

    # 目的のフレーム番号から指定した秒数だけ抜き出して保存する
    def extract(self, target_frame):
        self.video.set(1, target_frame)
        video_writer = cv2.VideoWriter(
            self.target_dir + self.movie_file.replace('.mp4', '').split('/')[-1] + '_' + str(target_frame) + '.mp4',
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
