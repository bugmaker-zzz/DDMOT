from collections import deque
import numpy as np
'''
  2D track management
  Reactivate: When a confirmed trajectory is occluded and in turn cannot be associated
  with any detections for several frames, it is then regarded as a reappeared trajectory.
'''

class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3
    Reactivate = 4


class TrackState3Dor2D:
    Tracking_3D = 1
    Tracking_2D = 2


class Track_2D:
    def __init__(self, track_id, n_init, max_age, tlwh, score, buffer_size=30, feature=None):

        # self.mean = mean
        # self.covariance = covariance
        self.track_id_2d = track_id  
        self.hits = 1
        self.age = 1
        self.state = TrackState.Tentative
        self.is3D_or_2D_track = TrackState3Dor2D.Tracking_2D  # 2D tracking
        self.time_since_update = 0
        self.n_init = n_init    # 连续n_init帧被检测到，状态就被设为confirmed
        self._max_age = max_age  # 一个跟踪对象丢失多少帧后会被删去（删去之后将不再进行特征匹配）

        self.score = score
        self.fusion_time_update = 0

        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        # change 5 -> 15
        self.conds = deque([], maxlen=5)
        self.xywh_omemory = deque([], maxlen=buffer_size)       #初始化一个双端队列，用于存储目标的预测状态历史
        self.xywh_pmemory = deque([], maxlen=buffer_size)       #初始化一个双端队列，用于存储目标的更新状态历史
        self.xywh_amemory = deque([], maxlen=buffer_size)       #初始化一个双端队列，用于存储目标的confirmed状态历史

        self.his_center_dist = []
        self.track_state = 0
        self.track_score = []

        self.scores = []

    # def to_tlwh(self):
    #     """
    #     Get current position in bounding box format `(top left x, top left y, width, height)`.
    #     Returns
    #     """
    #     ret = self.mean[:4].copy()
    #     ret[2] *= ret[3]
    #     ret[:2] -= ret[2:] / 2
    #     return ret

    def to_x1y1x2y2(self):
        """
        Get current position in bounding box format `(min x, miny, max x, max y)`.
        """
        ret = self._tlwh.copy()
        ret[2:] = ret[:2] + ret[2:]
        return ret
    
    # new add
    def to_x1y1x2y2_extended(self):
        """
        Get current position in bounding box format `(min x, miny, max x, max y)`.
        """
        ret = self._tlwh.copy()
        ret[2:] = ret[:2] + ret[2:]
        x1 = ret[0]
        y1 = ret[1]
        x2 = ret[2]
        y2 = ret[3]
        return [x1, y1, x2, y2, self.time_since_update]
    
    def to_xywh(self):
        ret = self._tlwh.copy()
        ret[:2] = ret[:2] + ret[2:] / 2
        return ret

    def increment(self):
        self.age += 1
        self.time_since_update += 1

    def predict_2d(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.increment()

    # new add
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def update_2d(self, detection):
        # self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())
        # self.features.append(detection.feature)
        self.hits += 1
        # self.age += 1
        new_tlwh = detection.tlwh
        new_score = detection.score
        self._tlwh = new_tlwh
        self.score = new_score
        self.xywh_omemory.append(detection.to_xywh())
        self.xywh_amemory[-1] = detection.to_xywh().copy()

        # track_score = 1 - (1 - self.track_score[-1]) * self.track_state
        # self.track_score.append(track_score)
        # self._max_age = self._max_age * (0.5 + self.sigmoid(self.track_score[1] / self.track_score[0]))
        # self.track_score.pop(0)

        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self.n_init:
            self.state = TrackState.Confirmed
        if self.state == TrackState.Reactivate:
            self.state =TrackState.Confirmed

        if self.fusion_time_update >= 3:
            self.state = TrackState.Reactivate

        if self.state == TrackState.Confirmed or self.state == TrackState.Tentative:
            tmp_delta_bbox = detection.to_xywh().copy() - self.xywh_amemory[-2].copy()
            tmp_conds = np.concatenate((detection.to_xywh().copy(), tmp_delta_bbox))
            self.conds[-1] = tmp_conds
        else:
            tmp_delta_bbox = detection.to_xywh().copy() - self.xywh_omemory[-2].copy()
            tmp_conds = np.concatenate((detection.to_xywh().copy(), tmp_delta_bbox))
            self.conds[-1] = tmp_conds

    # new add
    def state_update(self):
        if self.hits >= self.n_init:
            self.state = TrackState.Confirmed
        else:
            self.state = TrackState.Tentative

    def mark_missed(self):
        # new add
        # track_score = self.track_score[-1] - 0.1 * self.track_state
        # self.track_score.append(track_score)
        # self._max_age = self._max_age * (0.5 + self.sigmoid(self.track_score[1] / self.track_score[0]))

        if self.state == TrackState.Tentative or self.time_since_update > self._max_age:
            self.state = TrackState.Deleted
        elif self.state == TrackState.Confirmed and self.hits >= self.n_init:
            self.state = TrackState.Reactivate

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted
