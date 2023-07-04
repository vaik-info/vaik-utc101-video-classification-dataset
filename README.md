# vaik-utc101-video-classification-dataset
Parse and extract utc101 by tensorflow dataset

## Example

![baby_crawling_example](https://github.com/vaik-info/vaik-utc101-video-classification-dataset/assets/116471878/9f636575-8598-405c-aeec-b139bf162fa0)

## Usage

```shell
pip install -r requirements.txt
python main.py --split train \
                --classes_txt_path  ucf101_labels.txt\
                --image_height 224 \
                --image_width 224 \
                --fps 25 \
                --quality 10 \
                --max_class_num 25 \
                --output_dir_path ~/.vaik-utc101-video-classification-dataset
```

## Output

```shell
~/.vaik-utc101-video-classification-dataset$ tree
.
├── train
│   └── ApplyEyeMakeup
│       ├── ApplyEyeMakeup_1007.mp4
│       ├── ApplyEyeMakeup_1014.mp4
│       ├── ・・・
│       └── ApplyEyeMakeup_899.mp4
・・・
└── test
│   └── ApplyEyeMakeup
│       ├── ApplyEyeMakeup_1125.mp4
│       ├── ApplyEyeMakeup_1217.mp4
│       ├── ・・・
・・・
│       └── ApplyEyeMakeup_1218.mp4
```