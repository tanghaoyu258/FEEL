import numpy as np
import argparse
import shutil
import os


def parse_args():
    description = 'Weakly supervised action localization'
    parser = argparse.ArgumentParser(description=description)

    # dataset parameters
    parser.add_argument('--data_path', type=str, default='DATA_PATH')
    parser.add_argument('--exp_name', type=str, required=True, help="Name of the current experiment")
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--enlarge_rate', type=float, default=0)

    # data parameters
    parser.add_argument('--modal', type=str, default='all', choices=['rgb', 'flow', 'all'])
    parser.add_argument('--num_segments', default=50, type=int)
    parser.add_argument('--num_segments1', default=50, type=int)
    parser.add_argument('--num_segments2', default=1500, type=int)
    parser.add_argument('--scale', default=24, type=int)
    
    # model parameters
    parser.add_argument('--model_name', required=True, type=str, help="Which model to use")

    # training parameters
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rates for steps(list form)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--detection_inf_step', default=50, type=int, help="Run detection inference every n steps")  # 50
    parser.add_argument('--q_val', default=0.7, type=float)

    # inference parameters
    parser.add_argument('--inference_only', action='store_true', default=False)
    parser.add_argument('--class_th', type=float, default=0.15)  # 0.1 27.59% 0.15 27.63%
    parser.add_argument('--model_file', type=str, default=None, help='the path of pre-trained model file')
    parser.add_argument('--gamma', type=float, default=0.2, help='Gamma for oic class confidence')
    parser.add_argument('--soft_nms', default=False, action='store_true')
    parser.add_argument('--nms_alpha', default=0.35, type=float)
    parser.add_argument('--nms_thresh', default=0.6, type=float)
    parser.add_argument('--load_weight', default=False, action='store_true')
    
    # system parameters
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1, help='random seed (-1 for no manual seed)')  # 42
    parser.add_argument('--verbose', default=False, action='store_true')
    
    return init_args(parser.parse_args())


def init_args(args):

    args.model_path = os.path.join(args.output_dir, args.exp_name)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.model_path = os.path.join(args.output_dir, args.exp_name)

    return args


class Config(object):
    def __init__(self, args):
        self.lr = args.lr
        self.num_classes = 100
        self.modal = args.modal
        self.iter = args.iter
        if self.modal == 'all':
            self.len_feature = 2048
        else:
            self.len_feature = 1024
        self.batch_size = args.batch_size
        self.data_path = args.data_path
        self.data_anno_path = './data'
        if args.inference_only:
            self.data_anno_path = os.path.join(self.data_anno_path, str(int(self.iter)))
            self.inference_only = True
            if not os.path.exists(self.data_anno_path):
                os.makedirs(self.data_anno_path)
        else:
            self.data_anno_path = os.path.join(self.data_anno_path, str(int(self.iter)))
            self.inference_only = False
        self.model_path = os.path.join(args.output_dir, args.exp_name)
        self.model_path = os.path.join(self.model_path, str(int(self.iter)))
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.enlarge_rate = args.enlarge_rate
        # if args.inference_only:
        #     self.data_anno_path = os.path.join(self.data_anno_path, str(int(self.iter)))
        #     if not os.path.exists(self.data_anno_path):
        #         os.makedirs(self.data_anno_path)
        # else:
        #     self.data_anno_path = os.path.join(self.data_anno_path, self.iter)
        self.num_workers = args.num_workers
        self.class_thresh = args.class_th
        self.act_thresh = np.arange(0.1, 1.0, 0.1)
        self.act_thresh1 = np.arange(0.1, 1.0, 0.1)     
        self.q_val = args.q_val
        self.scale = args.scale
        self.gt_path = os.path.join(self.data_path, 'gt.json')
        self.model_file = args.model_file
        self.seed = args.seed
        self.feature_fps = 25
        self.num_segments = args.num_segments
        self.num_segments1 = args.num_segments1
        self.num_segments2 = args.num_segments2
        self.num_epochs = args.num_epochs
        self.gamma = args.gamma
        self.inference_only = args.inference_only
        self.model_name = args.model_name
        self.detection_inf_step = args.detection_inf_step
        self.soft_nms = args.soft_nms
        self.nms_alpha = args.nms_alpha
        self.nms_thresh = args.nms_thresh
        self.load_weight = args.load_weight
        self.verbose = args.verbose

class_dict = {0: 'Archery', 1: 'Ballet', 2: 'Bathing dog', 3: 'Belly dance', 4: 'Breakdancing', 5: 'Brushing hair', 6: 'Brushing teeth', 7: 'Bungee jumping', 8: 'Cheerleading', 9: 'Chopping wood', 10: 'Clean and jerk', 11: 'Cleaning shoes', 12: 'Cleaning windows', 13: 'Cricket', 14: 'Cumbia', 15: 'Discus throw', 16: 'Dodgeball', 17: 'Doing karate', 18: 'Doing kickboxing', 19: 'Doing motocross', 20: 'Doing nails', 21: 'Doing step aerobics', 22: 'Drinking beer', 23: 'Drinking coffee', 24: 'Fixing bicycle', 25: 'Getting a haircut', 26: 'Getting a piercing', 27: 'Getting a tattoo', 28: 'Grooming horse', 29: 'Hammer throw', 30: 'Hand washing clothes', 31: 'High jump', 32: 'Hopscotch', 33: 'Horseback riding', 34: 'Ironing clothes', 35: 'Javelin throw', 36: 'Kayaking', 37: 'Layup drill in basketball', 38: 'Long jump', 39: 'Making a sandwich', 40: 'Mixing drinks', 41: 'Mowing the lawn', 42: 'Paintball', 43: 'Painting', 44: 'Ping-pong', 45: 'Plataform diving', 46: 'Playing accordion', 47: 'Playing badminton', 48: 'Playing bagpipes', 49: 'Playing field hockey', 50: 'Playing flauta', 51: 'Playing guitarra', 52: 'Playing harmonica', 53: 'Playing kickball', 54: 'Playing lacrosse', 55: 'Playing piano', 56: 'Playing polo', 57: 'Playing racquetball', 58: 'Playing saxophone', 59: 'Playing squash', 60: 'Playing violin', 61: 'Playing volleyball', 62: 'Playing water polo', 63: 'Pole vault', 64: 'Polishing forniture', 65: 'Polishing shoes', 66: 'Preparing pasta', 67: 'Preparing salad', 68: 'Putting on makeup', 69: 'Removing curlers', 70: 'Rock climbing', 71: 'Sailing', 72: 'Shaving', 73: 'Shaving legs', 74: 'Shot put', 75: 'Shoveling snow', 76: 'Skateboarding', 77: 'Smoking a cigarette', 78: 'Smoking hookah', 79: 'Snatch', 80: 'Spinning', 81: 'Springboard diving', 82: 'Starting a campfire', 83: 'Tai chi', 84: 'Tango', 85: 'Tennis serve with ball bouncing', 86: 'Triple jump', 87: 'Tumbling', 88: 'Using parallel bars', 89: 'Using the balance beam', 90: 'Using the pommel horse', 91: 'Using uneven bars', 92: 'Vacuuming floor', 93: 'Walking the dog', 94: 'Washing dishes', 95: 'Washing face', 96: 'Washing hands', 97: 'Windsurfing', 98: 'Wrapping presents', 99: 'Zumba'}

