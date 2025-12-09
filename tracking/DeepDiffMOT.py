import os

import numpy as np
from tracking.detection import Detection_3D_Fusion, Detection_3D_only, Detection_2D, Detection_2D_Fusion
# from tracking.tracker import Tracker
from tracking.tracker_new import Tracker
from utils.kitti_oxts import load_oxts
from diffusion.D2MP_model import D2MP
from diffusion.condition_embedding import History_motion_embedding

import torch


class DeepDiffMOT():
    def __init__(self, cfg, category):
        '''
        :param max_age:  The maximum frames in which an object disappears.
        :param min_hits: The minimum frames in which an object becomes a trajectory in succession.
        '''
        self.min_frames = cfg[category].min_frames
        self.tracker = Tracker(cfg, category)
        self.reorder = [3, 4, 5, 6, 2, 1, 0]
        self.reorder_back = [6, 5, 4, 0, 1, 2, 3]
        self.frame_count = 0
        self.config = cfg
        self._build(self.config)

    def _build(self, cfg):
        self._build_dir(cfg)
        self._build_encoder()
        self._build_model(cfg)

    def _build_dir(self, cfg):
        model_dir = cfg.model_dir
        epoch = cfg.eval_at
        checkpoint_dir = os.path.join(model_dir, f"{self.config.dataset}_epoch{epoch}.pt")
        self.checkpoint = torch.load(checkpoint_dir, map_location = "cpu")
        # print("> Directory built!")

    def _build_encoder(self):
        self.encoder = History_motion_embedding()
        # print("> Encoder built!")

    def _build_model(self, cfg):
        """ Define Model """
        config = cfg
        model = D2MP(config, encoder=self.encoder)

        self.model = model
        self.model = self.model.cuda()
        self.model = self.model.eval()
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in self.checkpoint['ddpm'].items()})
        # print("> Model built!")

    def update(self, dets_fusion_2d_3d, dets_2d_only, dets_3d_only, cfg, frame, seq_id):
        calib_file = os.path.join(cfg.dataset_path, cfg.spilt, 'calib' + "/" + str(seq_id).zfill(4) + '.txt')
        oxts_file = os.path.join(cfg.dataset_path, cfg.spilt, 'oxts' + "/" + str(seq_id).zfill(4) + '.txt')
        imu_poses = load_oxts(oxts_file)

        dets_fusion_3d_camera = np.array(dets_fusion_2d_3d['dets_3d_fusion'])
        dets_fusion_3d_info = np.array(dets_fusion_2d_3d['dets_3d_fusion_info'])
        dets_fusion_2d = np.array(dets_fusion_2d_3d['dets_2d_fusion'])
        dets_fusion_2d_info = np.array(dets_fusion_2d_3d['dets_2d_fusion_info'])
        dets_3d_only_camera = np.array(dets_3d_only['dets_3d_only'])
        dets_3d_only_info = np.array(dets_3d_only['dets_3d_only_info'])

        # -------------- [h,w,l,x,y,z,rot_y] to [x,y,z,rot_y，l,w,h] ---------------
        if len(dets_fusion_3d_camera) == 0:
            dets_fusion_3d_camera = dets_fusion_3d_camera
        else:
            dets_fusion_3d_camera = dets_fusion_3d_camera[:, self.reorder]
        if len(dets_3d_only_camera) == 0:
            dets_3d_only_camera = dets_3d_only_camera
        else:
            dets_3d_only_camera = dets_3d_only_camera[:, self.reorder]

        dets_fusion_3d_camera = [Detection_3D_Fusion(det_fusion, dets_fusion_3d_info[i]) for i, det_fusion in enumerate(dets_fusion_3d_camera)]
        dets_fusion_2d = [Detection_2D_Fusion(det_fusion, dets_fusion_2d_info[i]) for i, det_fusion in enumerate(dets_fusion_2d)]
        dets_3d_only_camera = [Detection_3D_only(det_only, dets_3d_only_info[i]) for i, det_only in enumerate(dets_3d_only_camera)]
        # dets_2d_only = [Detection_2D(det_fusion) for i, det_fusion in enumerate(dets_2d_only)]
        dets_2d_only = [Detection_2D(det_only[:4], det_only[4]) for i, det_only in enumerate(dets_2d_only)]

        self.tracker.predict_diff_2d(self.tracker.tracks_2d, self.model, img_w=1242, img_h=375)
        # self.tracker.predict_diff_3d(self.tracker.tracks_3d, self.model, img_w=1242, img_h=375)
        # self.tracker.predict_3d_kf_diff()
        
        # self.tracker.predict_2d()
        self.tracker.predict_3d()

        # -------------------- Motion Compensation ---------------------
        if (frame > 0) and (calib_file is not None):
            self.tracker.ego_motion_compensation(frame, calib_file, imu_poses)

        # ------------------------- Track update ------------------------
        # self.tracker.update(dets_fusion_3d_camera, dets_fusion_2d, dets_3d_only_camera, dets_2d_only)
        self.tracker.update(dets_fusion_3d_camera, dets_3d_only_camera, dets_2d_only)

        # --------------------------- Outputs ----------------------------
        self.frame_count += 1
        outputs = []
        for track in self.tracker.tracks_3d:
            # --------------- Only outputs trajectory with confirmed status --------------
            if track.is_confirmed() or self.frame_count <= self.min_frames:
                bbox = np.array(track.pose[self.reorder_back])
                outputs.append(np.concatenate(([track.track_id_3d], bbox, track.additional_info)).reshape(1, -1))
        # for track in self.tracker.tracks_2d:
        #     if track.is_confirmed():
        #         bbox = track.to_x1y1x2y2()
        #         id_2d = track.track_id_2d
        #         add_bbox = np.array([-1, -1, -1, -1, -1 -1000, -1000, -1000, -10, 1])
        #         outputs.append(np.concatenate(([id_2d], add_bbox, bbox, [-1])).reshape(1,-1))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs