import xml.etree.ElementTree as ET
import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse


class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.fps = 0
        self.resolution = (0, 0)
        self.labels = ['slap', 'kick', 'strike', 'fall_down',
                       'walk_difficultly', 'squat_down', 'stand_up',
                       'jump', 'run', 'push', 'group_walking']

    def listdir(self, dir, list_name):
        for file in os.listdir(dir):
            file_path = os.path.join(dir, file)
            if os.path.isdir(file_path):
                listdir(file_path, list_name)
            elif os.path.splitext(file_path)[1] == '.avi':
                list_name.append(file_path)

    def run(self):
        #
        video_paths = []
        self.listdir(self.arg.dataset_dir + '/videos', video_paths)

        #
        label_paths = []
        for video_path in video_paths:
            label_path = video_path.replace(
                'videos', 'labels').replace('avi', 'xml')
            label_paths.append(label_path)

        #
        for video_path in video_paths:
            print(video_path)
            #
            # video_path = '/data1/dataset/zjlab_indoor_actions/videos/snippet_cam_2_4.avi'
            
            video = self.read_video(video_path)
            
            print('read done')
            #
            label_path = video_path.replace(
                'videos', 'labels').replace('avi', 'xml')
            label = ET.ElementTree(file=label_path)
            
            #
            for i, track in tqdm(enumerate(label.iter(tag='track'))):
                frames = []
                for box in track:
                    frame_id = int(box.attrib['frame'])
                    img = video[frame_id]
                    frames.append(img)

                save_path = self.parse_save_path(video_path, track)

                self. generate_video(
                    frames, self.fps, self.resolution, save_path)

    def parse_save_path(self, video_path, track):
        # format:label_snippet_cam_person_track
        snippet, cam = video_path.split('/')[-1].split('.')[0].split('_')[2:]
        label_id = self.labels.index(track.attrib['label'])
        save_path = self.arg.work_dir + '/L' + \
            str(label_id) + 'S' + snippet + 'C' + cam + \
            'P' + '0' + 'T' + track.attrib['id'] + '.avi'
        return save_path

    def read_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(3))
        height = int(cap.get(4))
        self.fps = cap.get(5)
        self.resolution = (width, height)

        frame_num = int(cap.get(7))

        video = []
        cnt = 0
        while(cap.isOpened()):
            ret, img = cap.read()
            if not ret:
                break
            video.append(img)
            cnt += 1
            if cnt % 1000 == 0:
                print(cnt)
        cap.release()

        return video

    def generate_video(self, frames, fps, resolution, save_path):
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        videoWriter = cv2.VideoWriter(save_path, fourcc, fps, resolution)
        for frame in frames:
            videoWriter.write(frame)
        videoWriter.release()


def get_parser():
    # parameter priority: command line > config file > default
    parser = argparse.ArgumentParser(description='MS-G3D')

    parser.add_argument(
        '--dataset_dir',
        default='/data1/zhumh/tmp',
        type=str,
        help='dataset directory')
    parser.add_argument(
        '--work_dir',
        default='/data1/zhumh/tmp/work_dir',
        type=str,
        help='work directory')

    return parser


def main():
    parser = get_parser()
    arg = parser.parse_args()

    processor = Processor(arg)
    processor.run()


if __name__ == '__main__':
    main()
