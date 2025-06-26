import argparse
import os
import numpy as np
from copy import deepcopy
import random
import open3d as o3d
import logging
import matplotlib.pyplot as plt

def translate_boxes_to_o3d_instance(gt_boxes):
    '''
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    '''
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set, box3d

def draw_box_no_track(vis, pts, gt_boxes, color=None, ref_labels=None, score=None, bins=None, all=None):
    vis.add_geometry(pts)
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_o3d_instance(gt_boxes[i])
        if bins:
            box3d_num_points = len(pts.crop(box3d).points)
        
        if all:
            gt_color = (1,0,0)
            if bins:
                bins_range = list(map(int, (bins).split(',')))
                if len(bins_range) == 1:
                    if bins_range[0] > box3d_num_points:
                        gt_color = (0,1,0)
                elif len(bins_range) == 2:
                    if bins_range[0] > box3d_num_points or box3d_num_points > bins_range[1]:
                        gt_color = (0,1,0)
                else:
                    logging.error('Have more than 2 inputs for --bins')
                    exit(1)
        
            if ref_labels is None:
                if(color is None):
                    line_set.paint_uniform_color(gt_color)
                else:
                    line_set.paint_uniform_color(color[i])
            else:
                line_set.paint_uniform_color(score[i])

            vis.add_geometry(line_set)

        else:
            if bins:
                bins_range = list(map(int, (bins).split(',')))
                if len(bins_range) == 1:
                    if bins_range[0] > box3d_num_points:
                        continue
                elif len(bins_range) == 2:
                    if bins_range[0] > box3d_num_points or box3d_num_points > bins_range[1]:
                        continue
                else:
                    logging.error('Have more than 2 inputs for --bins')
                    exit(1)
        
            if ref_labels is None:
                if(color is None):
                    line_set.paint_uniform_color((0, 1, 0))
                else:
                    line_set.paint_uniform_color(color[i])
            else:
                line_set.paint_uniform_color(score[i])

            vis.add_geometry(line_set)

    return vis

class Dataset():
    def __init__(self, data_path, labels_path, trackingInfoPresent=False, ext='.npy'):
        self.label_list = sorted([f for f in os.listdir(labels_path) if f.endswith('.txt')])
        self.data_path = data_path
        self.labels_path = labels_path
        self.trackingInfoPresent = trackingInfoPresent
        self.colour_dict = {}
        self.ext = ext

    def __len__(self):
        return len(self.label_list)
    
    def get_colour(self, track_id):
        if(track_id not in self.colour_dict):
            self.colour_dict[track_id] = np.array([random.random(), random.random(), random.random()], dtype=np.float64)
        
        return self.colour_dict[track_id]
    
    def read_txt(self, path):
        with open(path,'r') as f:
            data = f.readlines()
        data = [da.strip() for da in data]
        return data[1:]


    def txt2json_dict(self, data):
        res_dict = {}
        labels = []
        for da in data:
            temp_dict = {}
            x,y,z,w,l,h,rt,id,score = da.split(',')[0:9]
            if int(id) < 10:
                temp_dict['x'] = float(x)
                temp_dict['y'] = float(y)
                temp_dict['z'] = float(z)
                temp_dict['l'] = float(l)
                temp_dict['w'] = float(w)
                temp_dict['h'] = float(h)
                temp_dict['rt'] = float(rt)
                temp_dict['id'] = int(id)
                temp_dict['score'] = float(score)
                labels.append(temp_dict)
        res_dict['labels'] = labels
        print("Number of labels: ", len(labels))
        return res_dict
        
    def __getitem__(self, index):
        data_dict = {}
        label_filename = self.label_list[index]
        label_filepath = os.path.join(self.labels_path, label_filename)
        labels = self.read_txt(label_filepath)
        labels = self.txt2json_dict(labels)
        gt_boxes = np.asarray([[label['x'], label['y'], label['z'], label['l'], label['w'], label['h'], label['rt']] for label in labels['labels']], dtype=np.float64)

        frame_id = label_filename.split('.')[0]
        if(self.ext == '.npy'):
            data_filename = frame_id + '.npy'
            data_filepath = os.path.join(self.data_path, data_filename)
            data = np.load(data_filepath)
        elif(self.ext == '.pcd'):
            data_filename = frame_id + '.pcd'
            data_filepath = os.path.join(self.data_path, data_filename)
            pcd = o3d.io.read_point_cloud(data_filepath)
            data = np.asarray(pcd.points)
        elif(self.ext == '.bin'):
            data_filename = frame_id + '.bin'
            data_filepath = os.path.join(self.data_path, data_filename)
            data = np.fromfile(data_filepath, dtype=np.float32).reshape(-1, 4)

        data_dict['frame_id'] = frame_id
        data_dict['points'] = data
        data_dict['gt_boxes'] = gt_boxes

        if(self.trackingInfoPresent):
            track_ids = [label[7] for label in labels]
            data_dict['track_ids'] = track_ids
            data_dict['colours'] = [self.get_colour(track_id) for track_id in track_ids]
            # print(data_dict['track_ids'])
        
        return data_dict

def get_points_labels(pts, test_set, idx, use_intensity=False):
    logger = logging.getLogger('visualize_gt')
    data_dict = test_set[idx]
    # data_dict = test_set.collate_batch([data_dict])
    gt_boxes = data_dict['gt_boxes']
    
    if('colours' in data_dict):
        colours = data_dict['colours']
    else:
        colours = None

    points = data_dict['points']
    logger.debug(f'Points shape: {points.shape}')
    pts.points = o3d.utility.Vector3dVector(points[:, :3])

    if(use_intensity and points.shape[1] == 4):
        # Use the fourth column as intensity values
        colors = plt.get_cmap('terrain')(points[:, 3] / np.max(points[:, 3]))[:, :3]  # Normalize and convert to RGB
        pts.colors = o3d.utility.Vector3dVector(colors)
    else:
        pts.colors = o3d.utility.Vector3dVector(np.ones((data_dict['points'].shape[0], 3)))
        
    return pts, data_dict['frame_id'], gt_boxes, colours

def visualize_point_clouds(test_set, use_intensity, bins_range=None, all=None):
    logger = logging.getLogger('visualize_gt')
    if len(test_set) == 0:
        logger.error('No point clouds found in the directory.')
        return
    
    logger.info(f'Total number of files to visualize: {len(test_set)}')

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    pts = o3d.geometry.PointCloud()

    idx = 0
    pts, frame_id, gt_boxes, colours = get_points_labels(pts, test_set, idx, use_intensity)
    draw_box_no_track(vis, pts, gt_boxes, colours, bins=bins_range, all=all)
    logger.info(f'Visualizing: {frame_id}')

    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()

    def load_next_point_cloud(vis):
        nonlocal idx, pts, camera_params
        if idx < len(test_set) - 1:
            idx += 1
            pts, frame_id, gt_boxes, colours = get_points_labels(pts, test_set, idx, use_intensity)
            vis.clear_geometries()
            vis = draw_box_no_track(vis, pts, gt_boxes, colours, bins=bins_range, all=all)
            view_control.convert_from_pinhole_camera_parameters(camera_params)
            logger.info(f'Visualizing: {frame_id}')
        else:
            vis.destroy_window()  # Signal to close the visualizer

    def load_previous_point_cloud(vis):
        nonlocal idx, pts, camera_params
        if idx > 0:
            idx -= 1
            pts, frame_id, gt_boxes, colours = get_points_labels(pts, test_set, idx, use_intensity)
            vis.clear_geometries()
            vis = draw_box_no_track(vis, pts, gt_boxes, colours, bins=bins_range, all=all)
            view_control.convert_from_pinhole_camera_parameters(camera_params)
            logger.info(f'Visualizing: {frame_id}')

    def save_camera_params(vis):
        nonlocal camera_params
        camera_params = view_control.convert_to_pinhole_camera_parameters()

    vis.register_key_callback(262, load_next_point_cloud)  # Right arrow key
    vis.register_key_callback(263, load_previous_point_cloud)  # Left arrow key
    vis.register_animation_callback(save_camera_params)
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    vis.run()

def parse_config():
    parser = argparse.ArgumentParser(description='Visualize ground truth annotations on point cloud from directories')
    parser.add_argument('--path', type=str, default=None,
                        help='basic directory which should include both points and labels folder')
    parser.add_argument('--data_path', type=str, default='../data/curium/motion1/',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--labels_path', type=str,
                        help='specify the labels file or directory')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--use_intensity', action='store_true', help='Use the fourth column as intensity for coloring the point cloud')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--use_tracking', action='store_true', help='Use the eigth column of labels as tracking ID')
    parser.add_argument('--bins', type=str, default=None, help='filter ground truth boxes with certain range')
    parser.add_argument('--all', action='store_true', help='draw all gt boxes in different color according to the range of number of points selected')

    args = parser.parse_args()
    return args

def main():
    args = parse_config()
    if args.debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)

    if args.all and not args.bins:
        logging.error('Have to use --bins when using --all')
        exit(1)

    logger = logging.getLogger('visualize_gt')
    
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    logger.info('-----------------Visualizing Ground Truth Data-------------------------')
    
    path = args.path
    if path:
        test_set = Dataset(os.path.join(path, 'points'), os.path.join(path, 'labels'), trackingInfoPresent=args.use_tracking, ext=args.ext)
    else:
        test_set = Dataset(args.data_path, args.labels_path, trackingInfoPresent=args.use_tracking, ext=args.ext)
    
    num_samples = len(test_set)
    logger.info(f'Total number of samples: {num_samples}')

    visualize_point_clouds(test_set, args.use_intensity, args.bins, args.all)

if __name__ == '__main__':
    main()