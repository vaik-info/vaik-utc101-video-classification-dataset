import argparse
import os
import shutil
import tqdm

import imageio
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def dump(split, classes_txt_path, image_height, image_width, fps, max_class_num, output_dir_path):
    output_split_dir_path = os.path.join(output_dir_path, split)

    os.makedirs(output_split_dir_path, exist_ok=True)
    with open(classes_txt_path, 'r') as f:
        classes = f.readlines()
    classes = [label.strip() for label in classes][:max_class_num]

    config = tfds.download.DownloadConfig(verify_ssl=False)
    ds = tfds.load(name='ucf101', split=split, download_and_prepare_kwargs={"download_config": config})

    for data_index, data in tqdm.tqdm(enumerate(ds), total=len(ds), desc='dump videos'):
        class_index = data['label']
        if class_index >= max_class_num:
            continue
        class_label = classes[class_index]
        output_sub_dir_path = os.path.join(output_split_dir_path, class_label)
        os.makedirs(output_sub_dir_path, exist_ok=True)

        video = data['video']
        ratio = min(image_height / video.shape[1], image_width / video.shape[2])
        video = tf.image.resize(video, (round(ratio * video.shape[1]), round(ratio * video.shape[2])))
        image_array_list = [np.clip(image.numpy(), 0.0, 255.0).astype(np.uint8) for image in video]
        imageio.mimwrite(os.path.join(output_sub_dir_path, f'{class_label}_{data_index}.avi'), image_array_list, fps=fps, codec='rawvideo')

    with open(os.path.join(output_dir_path, os.path.basename(classes_txt_path)), 'w') as f:
        for class_label in classes:
            f.write(f'{class_label}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dump')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--classes_txt_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'ucf101_labels.txt'))
    parser.add_argument('--image_height', type=int, default=224)
    parser.add_argument('--image_width', type=int, default=224)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--max_class_num', type=int, default=25)
    parser.add_argument('--output_dir_path', type=str, default='~/.vaik-utc101-video-classification-dataset')

    args = parser.parse_args()
    args.classes_txt_path = os.path.expanduser(args.classes_txt_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    dump(args.split, args.classes_txt_path, args.image_height, args.image_width, args.fps,
         args.max_class_num, args.output_dir_path)
