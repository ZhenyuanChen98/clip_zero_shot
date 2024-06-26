CLS2IDX = {1: 'aeroplane',
           4: 'apple',
           8: 'bed',
           9: 'backpack',
           12: 'banana',
           15: 'baseball',
           19: 'bear',
           22: 'bench',
           23: 'bicycle',
           25: 'bird',
           27: 'tie',
           29: 'bowl',
           32: 'bus',
           36: 'car',
           42: 'chair',
           46: 'keyboard',
           47: 'mouse',
           53: 'cup',
           55: 'clock',
           57: 'dog',
           58: 'cat',
           63: 'elephant',
           79: 'hair drier', 
           91: 'horse', 
           92: 'hot dog', 
           96: 'teddy bear', 
           100: 'laptop', 
           109: 'microwave', 
           113: 'motorbike', 
           118: 'orange', 
           123: 'person', 
           126: 'sports ball',
           128: 'pizza', 
           141: 'racket', 
           144: 'refrigerator', 
           147: 'sports ball', 
           154: 'sheep', 
           155: 'skis', 
           162: 'sports ball', 
           163: 'sofa', 
           176: 'table', 
           178: 'sports ball', 
           180: 'tie', 
           182: 'toaster', 
           183: 'traffic light', 
           184: 'train', 
           188: 'tvmonitor', 
           195: 'bottle', 
           198: 'bottle', 
           199: 'zebra'}

categories = ['aeroplane',
              'apple',
              'bed',
              'backpack',
              'banana',
              'baseball',
              'bear',
              'bench',
              'bicycle',
              'bird',
              'tie',
              'bowl',
              'bus',
              'car',
              'chair',
              'keyboard',
              'mouse',
              'cup',
              'clock',
              'dog',
              'cat',
              'elephant',
              'hair drier',
              'horse',
              'hot dog',
              'teddy bear',
              'laptop',
              'microwave',
              'motorbike',
              'orange',
              'person',
              'sports ball',
              'pizza',
              'racket',
              'refrigerator',
              'sheep',
              'skis',
              'sofa',
              'table',
              'tie',
              'toaster',
              'traffic light',
              'train',
              'tvmonitor',
              'bottle',
              'zebra']

def process_line(line):
    parts = line.strip().split(' ')
    file_name, indices = parts[0], [int(idx) - 1 for idx in parts[1:]]

    matched_indices = [idx for idx in indices if idx in CLS2IDX.keys()]
    
    return (file_name, [CLS2IDX[idx] for idx in matched_indices])

with open('/data/clip_zero_shot/data/data_list/ImageNet/annotations_all.txt', 'r') as f_a, open('/data/clip_zero_shot/data/data_list/ImageNet/annotations_new.txt', 'w') as f_b:
    for line in f_a:
        file_name, matched_categories = process_line(line)
        if matched_categories:
            category_idx = ' '.join([str(categories.index(cat)) for cat in matched_categories])
            f_b.write(f'{file_name} {category_idx}\n')