import copy
from collections import deque
import numpy as np

from utils.kitti_oxts import egomotion_compensation_ID, get_ego_traj

'''
  3D track management
  Reactivate: When a confirmed trajectory is occluded and in turn cannot be associated with any detections for several frames, it 
  is then regarded as a reappeared trajectory.
'''

class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3
    Reactivate = 4

class TrackState3Dor2D:
    Tracking_3D = 1
    Tracking_2D = 2


class Track_3D:

    def __init__(self, pose, kf_3d, track_id_3d, n_init, max_age,additional_info, feature=None):
        self.pose = pose
        self.kf_3d = kf_3d
        self.track_id_3d = track_id_3d
        self.hits = 1
        self.age = 1
        self.state = TrackState.Tentative
        self.n_init = n_init
        self._max_age = max_age
        self.is3D_or_2D_track = TrackState3Dor2D.Tracking_3D
        self.additional_info = additional_info
        self.time_since_update = 0
        self.fusion_time_update = 0

    # def __init__(self, pose, kf_3d, track_id_3d, n_init, max_age, additional_info, tlwh, feature=None):
    #     self.pose = pose
    #     self.kf_3d = kf_3d
    #     self.track_id_3d = track_id_3d
    #     self.hits = 1
    #     self.age = 1
    #     self.state = TrackState.Tentative
    #     self.n_init = n_init
    #     self._max_age = max_age
    #     self.is3D_or_2D_track = TrackState3Dor2D.Tracking_3D
    #     self.additional_info = additional_info
    #     self.time_since_update = 0
    #     self.fusion_time_update = 0

    #     # new add
    #     self.tlwh = tlwh
    #     self.conds = deque([], maxlen=5)
    #     self.xywh_omemory = deque([], maxlen=30)       #初始化一个双端队列，用于存储目标的预测状态历史
    #     self.xywh_pmemory = deque([], maxlen=30)       #初始化一个双端队列，用于存储目标的更新状态历史
    #     self.xywh_amemory = deque([], maxlen=30)       #初始化一个双端队列，用于存储目标的confirmed状态历史

    #     self.his_center_dist = []
    #     self.track_state = 0
    #     self.track_score = []

        self.scores = []

    # additional_info include (alpha,type,x1,y1,x2,y2,score)
    def to_xywh(self):
        # x1y1x2y2 = self.additional_info[2:6]
        # tlwh = np.asarray([x1y1x2y2[0], x1y1x2y2[1], x1y1x2y2[2] - x1y1x2y2[0], x1y1x2y2[3] - x1y1x2y2[1]], dtype=np.float64)
        ret = self.tlwh.copy()
        ret[:2] = ret[:2] + ret[2:] / 2
        return ret

    # 用于融合扩散模型和卡尔曼模型的预测结果
    def predict_3d_kf_diff(self, trk_3d):
        self.pose = trk_3d.predict()


    def predict_3d(self, trk_3d):
        self.pose = trk_3d.predict()

    # predict for reactivate object
    # def predict_3d_further(self, trk_3d):
    #     self.pose = trk_3d.predict_further()

    def update_3d(self, detection_3d):
        self.kf_3d.update(detection_3d.bbox)
        self.additional_info = detection_3d.additional_info
        self.pose = np.concatenate(self.kf_3d.kf.x[:7], axis=0)
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        if self.hits >= self.n_init:
            self.state = TrackState.Confirmed
        else:
            self.state = TrackState.Tentative
        if self.fusion_time_update >= 3:
            self.state = TrackState.Reactivate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # def update_3d(self, detection_3d):
    #     self.kf_3d.update(detection_3d.bbox)
    #     self.additional_info = detection_3d.additional_info
    #     self.pose = np.concatenate(self.kf_3d.kf.x[:7], axis=0)
    #     self.hits += 1
    #     self.age += 1
    #     self.time_since_update = 0
    #     if self.hits >= self.n_init:
    #         self.state = TrackState.Confirmed
    #     else:
    #         self.state = TrackState.Tentative
    #     if self.fusion_time_update >= 3:
    #         self.state = TrackState.Reactivate

    #     self.xywh_omemory.append(detection_3d.to_xywh())
    #     self.xywh_amemory[-1] = detection_3d.to_xywh().copy()
    #     if self.state == TrackState.Confirmed or self.state == TrackState.Tentative:
    #         tmp_delta_bbox = detection_3d.to_xywh().copy() - self.xywh_amemory[-2].copy()
    #         tmp_conds = np.concatenate((detection_3d.to_xywh().copy(), tmp_delta_bbox))
    #         self.conds[-1] = tmp_conds
    #     else:
    #         tmp_delta_bbox = detection_3d.to_xywh().copy() - self.xywh_omemory[-2].copy()
    #         tmp_conds = np.concatenate((detection_3d.to_xywh().copy(), tmp_delta_bbox))
    #         self.conds[-1] = tmp_conds

    def state_update(self):
        if self.hits >= self.n_init:
            self.state = TrackState.Confirmed
        else:
            self.state = TrackState.Tentative

    def mark_missed(self):
        self.time_since_update += 1
        if self.state == TrackState.Confirmed and self.hits >= self.n_init:
            self.state = TrackState.Reactivate
        elif self.time_since_update >= 1 and self.state != TrackState.Reactivate:
            self.state = TrackState.Deleted
        elif self.state == TrackState.Reactivate and self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def fusion_state(self):
        if  self.fusion_time_update >= 2:
            self.state = TrackState.Deleted

    def is_deleted(self):
        return self.state == TrackState.Deleted

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_track_id_3d(self):
        track_id_3d= self.track_id_3d
        return track_id_3d

    def ego_motion_compensation_3d(self, frame, calib_file, oxts):
        # inverse ego motion compensation, move trks from the last frame of coordinate to the current frame for matching

        # assert len(self.trackers) == len(trks), 'error'
        ego_xyz_imu, ego_rot_imu, left, right = get_ego_traj(oxts, frame, 1, 1, only_fut=True, inverse=True)
        xyz = np.array([self.pose[0], self.pose[1], self.pose[2]]).reshape((1, -1))
        compensated = egomotion_compensation_ID(xyz, calib_file, ego_rot_imu, ego_xyz_imu, left, right)
        self.pose[0], self.pose[1], self.pose[2] = compensated[0]
        try:
            self.kf_3d.kf.x[:3] = copy.copy(compensated).reshape((-1))
        except:
            self.kf_3d.kf.x[:3] = copy.copy(compensated).reshape((-1, 1))