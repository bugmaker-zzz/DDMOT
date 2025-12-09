import numpy as np
from tracking import kalman_filter_2d
from tracking.matching import associate_dets_to_trks_fusion, associate_dets_to_trks, associate_2D_to_3D_tracking, associate_dets_to_trks_by_extended_bbox_2d   #new add
from tracking.track_2d import Track_2D
from tracking.kalman_fileter_3d import  KalmanBoxTracker
from tracking.track_3d import Track_3D
import math


class Tracker():
    def __init__(self, cfg, category):
        self.cfg = cfg
        self.cost_3d = self.cfg[category].metric_3d
        self.cost_2d = self.cfg[category].metric_2d
        self.threshold_3d = self.cfg[category]["cost_function"][self.cost_3d]
        self.threshold_2d = self.cfg[category]["cost_function"][self.cost_2d]
        self.max_age = self.cfg[category].max_ages
        self.min_frames = self.cfg[category].min_frames
        self.tracks_3d = []
        self.tracks_2d = []
        self.track_id_3d = 0   # The id of 3D track is represented by an even number.
        self.track_id_2d = 1   # The id of 3D track is represented by an odd number.
        self.unmatch_tracks_3d = []
        self.kf_2d = kalman_filter_2d.KalmanFilter_kf()

        # new add 用于计算动态生命周期的尺度系数α和偏移系数β
        self.alpha = self.cfg[category].alpha
        self.beta = self.cfg[category].beta
        # new add 仿照bytetrack进行级联匹配（高置信度匹配+低置信度匹配）
        self.track_threshold_high_3d = self.cfg.track_threshold_high_3d
        self.track_threshold_low_3d = self.cfg.track_threshold_low_3d
        self.track_threshold_high_2d = self.cfg.track_threshold_high_2d
        self.track_threshold_low_2d = self.cfg.track_threshold_low_2d
        # new add 用于解决队列下标问题
        self.track3d_count = 0
        self.track2d_count = 0

    # 基于扩散模型的运动预测
    def predict_diff_2d(self, tracks, model, img_w, img_h):
        if len(tracks) > 0:
            # dets = np.asarray([st.xywh.copy() for st in tracks]).reshape(-1, 4)
            dets = np.asarray([st.to_xywh().copy() for st in tracks]).reshape(-1, 4)

            dets[:, 0::2] = dets[:, 0::2] / img_w
            dets[:, 1::2] = dets[:, 1::2] / img_h

            conds = [st.conds for st in tracks]

            multi_track_pred = model.generate(conds, sample=1, bestof=True, img_w=img_w, img_h=img_h)
            track_pred = multi_track_pred.mean(0)


            track_pred = track_pred + dets

            track_pred[:, 0::2] = track_pred[:, 0::2] * img_w
            track_pred[:, 1::2] = track_pred[:, 1::2] * img_h
            track_pred[:, 0] = track_pred[:, 0] - track_pred[:, 2] / 2
            track_pred[:, 1] = track_pred[:, 1] - track_pred[:, 3] / 2


            for i, st in enumerate(tracks):
                st._tlwh = track_pred[i]
                st.xywh_pmemory.append(st.to_xywh().copy())
                st.xywh_amemory.append(st.to_xywh().copy())

                tmp_delta_bbox = st.to_xywh().copy() - st.xywh_amemory[-2].copy()
                tmp_conds = np.concatenate((st.to_xywh().copy(), tmp_delta_bbox))
                st.conds.append(tmp_conds)

            for track in tracks:
                track.increment()

    # 3D 扩散模型预测
    def predict_diff_3d(self, tracks, model, img_w, img_h):
        if len(tracks) > 0:
            dets = np.asarray([st.to_xywh().copy() for st in tracks]).reshape(-1, 4)

            dets[:, 0::2] = dets[:, 0::2] / img_w
            dets[:, 1::2] = dets[:, 1::2] / img_h

            conds = [st.conds for st in tracks]

            multi_track_pred = model.generate(conds, sample=1, bestof=True, img_w=img_w, img_h=img_h)
            track_pred = multi_track_pred.mean(0)

            # 输出目标预测xywh坐标（track_pred），经变换后为tlwh坐标
            track_pred = track_pred + dets

            track_pred[:, 0::2] = track_pred[:, 0::2] * img_w
            track_pred[:, 1::2] = track_pred[:, 1::2] * img_h
            track_pred[:, 0] = track_pred[:, 0] - track_pred[:, 2] / 2
            track_pred[:, 1] = track_pred[:, 1] - track_pred[:, 3] / 2

            for i, st in enumerate(tracks):
                st.tlwh = track_pred[i]
                st.xywh_pmemory.append(st.to_xywh().copy())
                st.xywh_amemory.append(st.to_xywh().copy())

                tmp_delta_bbox = st.to_xywh().copy() - st.xywh_amemory[-2].copy()
                tmp_conds = np.concatenate((st.to_xywh().copy(), tmp_delta_bbox))
                st.conds.append(tmp_conds)

    def predict_3d_kf_diff(self):
        for track in self.tracks_3d:
            track.predict_3d_kf_diff(track.kf_3d)

    def predict_3d(self):
        for track in self.tracks_3d:
            track.predict_3d(track.kf_3d)

    # def predict_3d(self):
    #     for track in self.tracks_3d:
    #         if track.state == 4:
    #             track.predict_3d_further(track.kf_3d)
    #         else:
    #             track.predict_3d(track.kf_3d)

    def predict_2d(self):
        for track in self.tracks_2d:
            track.predict_2d(self.kf_2d)

    def ego_motion_compensation(self, frame, calib_file, oxts):
        for track in self.tracks_3d:
            track.ego_motion_compensation_3d(frame, calib_file, oxts)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def adaptive_birth_death_memory_new(self, tracks_3d, tracks_2d):
        for track in tracks_3d:
            track.scores.append(track.additional_info[6])
            if len(track.scores) > 5:
                track.scores.pop(0)
            avg_3d_score = np.mean(track.scores)
            track._max_age =  track._max_age * (0.5 * self.sigmoid(10* avg_3d_score - 0.5)) + 15
        for track in tracks_2d:
            track.scores.append(track.score)
            if len(track.scores) > 3:
                track.scores.pop(0)
            avg_2d_score = np.mean(track.scores)
            track._max_age = track._max_age * (0.85 * self.sigmoid(10 * avg_2d_score + 0.2)) + 5

    def cal_track_state(self, tracks_3d, tracks_2d):
        for track in tracks_3d:
            center_3d_dist = np.sqrt((track.tlwh[0] + track.tlwh[2]) ** 2 + (track.tlwh[1] + track.tlwh[3]) ** 2)
            if len(track.his_center_dist) == 0:
                track.his_center_dist.append(center_3d_dist)
                track.track_score.append(1)
            else:
                track.his_center_dist.append(center_3d_dist)
                track.track_state = track.his_center_dist[1] / track.his_center_dist[0]
                track.his_center_dist.pop(0)
        for track in tracks_2d:
            center_2d_dist = np.sqrt((track._tlwh[0] + track._tlwh[2]) ** 2 + (track._tlwh[1] + track._tlwh[3]) ** 2)
            if len(track.his_center_dist) == 0:
                track.his_center_dist.append(center_2d_dist)
                track.track_score.append(1)
            else:
                track.his_center_dist.append(center_2d_dist)
                track.track_state = track.his_center_dist[1] / track.his_center_dist[0]
                track.his_center_dist.pop(0)

    def update(self, dets_3d_fusion, dets_3d_only, dets_2d_only):
        self.adaptive_birth_death_memory_new(self.tracks_3d, self.tracks_2d)

        # 1st Level of Association
        matched_fusion_idx, unmatched_dets_fusion_idx, unmatched_trks_fusion_idx = associate_dets_to_trks_fusion(
            dets_3d_fusion, self.tracks_3d, self.cost_3d, self.threshold_3d, metric='match_3d')
        # print("1st len:", len(matched_fusion_idx), len(unmatched_dets_fusion_idx), len(unmatched_trks_fusion_idx))
        for detection_idx, track_idx in matched_fusion_idx:
            self.tracks_3d[track_idx].update_3d(dets_3d_fusion[detection_idx])
            self.tracks_3d[track_idx].state = 2
            self.tracks_3d[track_idx].fusion_time_update = 0
        for track_idx in unmatched_trks_fusion_idx:
            self.tracks_3d[track_idx].fusion_time_update += 1
            self.tracks_3d[track_idx].mark_missed()
        for detection_idx in unmatched_dets_fusion_idx:
            self.initiate_trajectory_3d(dets_3d_fusion[detection_idx])

        # 2nd Level of Association
        self.unmatch_tracks_3d1 = [t for t in self.tracks_3d if t.time_since_update > 0]
        matched_only_idx, unmatched_dets_only_idx, _ = associate_dets_to_trks_fusion(
            dets_3d_only, self.unmatch_tracks_3d1, self.cost_3d, self.threshold_3d, metric='match_3d')
        # print("2nd len:", len(matched_only_idx), len(unmatched_dets_only_idx))
        index_to_delete = []
        for detection_idx, track_idx in matched_only_idx:
            for index, t in enumerate(self.tracks_3d):
                if t.track_id_3d == self.unmatch_tracks_3d1[track_idx].track_id_3d:
                    t.update_3d(dets_3d_only[detection_idx])
                    index_to_delete.append(track_idx)
                    break
        self.unmatch_tracks_3d1 = [self.unmatch_tracks_3d1[i] for i in range(len(self.unmatch_tracks_3d1)) if i not in index_to_delete]
        # print("self.unmatch_tracks_3d1:", len(self.unmatch_tracks_3d1))
        for detection_idx in unmatched_dets_only_idx:
            self.initiate_trajectory_3d(dets_3d_only[detection_idx])
        self.unmatch_tracks_3d2 = [t for t in self.tracks_3d if t.time_since_update == 0 and t.hits == 1]
        self.unmatch_tracks_3d = self.unmatch_tracks_3d1 + self.unmatch_tracks_3d2

        # 3rd Level of Association
        matched, unmatch_dets, unmatch_trks = \
            associate_dets_to_trks(dets_2d_only, self.tracks_2d, self.cost_2d, self.threshold_2d, metric='match_2d')
        for detection_idx, track_idx in matched:
            self.tracks_2d[track_idx].update_2d(dets_2d_only[detection_idx])
        for track_idx in unmatch_trks:
            self.tracks_2d[track_idx].mark_missed()
        for detection_idx in unmatch_dets:
            self.initiate_trajectory_2d(dets_2d_only[detection_idx])

        # 4th Level of Association
        matched_track_2d, unmatch_tracks_2d = associate_2D_to_3D_tracking(self.tracks_2d, self.unmatch_tracks_3d, self.threshold_2d)
        # print("4th len:", len(matched_track_2d), len(unmatch_tracks_2d))
        index_to_delete2 = []
        for track_idx_2d, track_idx_3d in matched_track_2d:
            for i in range(len(self.tracks_3d)):
                if self.tracks_3d[i].track_id_3d == self.unmatch_tracks_3d[track_idx_3d].track_id_3d:
                    self.track2d_count -= 1
                    self.tracks_3d[i].age = self.tracks_2d[track_idx_2d].age + 1
                    self.tracks_3d[i].time_since_update = 0
                    if self.tracks_2d[track_idx_2d].hits >= 2:
                        self.tracks_3d[i].hits = self.tracks_2d[track_idx_2d].hits + 1
                    else:
                        self.tracks_3d[i].hits += 1
                    self.tracks_3d[i].state_update()
            index_to_delete2.append(track_idx_2d)
        self.tracks_2d = [self.tracks_2d[i] for i in range(len(self.tracks_2d)) if i not in index_to_delete2]

        for t in self.tracks_3d:
            if t.is_deleted():
                self.track3d_count -= 1
        self.tracks_3d = [t for t in self.tracks_3d if not t.is_deleted()]
        for t in self.tracks_2d:
            if t.is_deleted():
                self.track2d_count -= 1
        self.tracks_2d = [t for t in self.tracks_2d if not t.is_deleted()]

    def initiate_trajectory_3d(self, detection):
        self.kf_3d = KalmanBoxTracker(detection.bbox)
        self.additional_info = detection.additional_info
        pose = np.concatenate(self.kf_3d.kf.x[:7], axis=0)
        self.tracks_3d.append(Track_3D(pose, self.kf_3d, self.track_id_3d, self.min_frames, self.max_age, self.additional_info, detection.tlwh))
        # self.tracks_3d[self.track3d_count].xywh_omemory.append(detection.to_xywh().copy())
        # self.tracks_3d[self.track3d_count].xywh_pmemory.append(detection.to_xywh().copy())
        # self.tracks_3d[self.track3d_count].xywh_amemory.append(detection.to_xywh().copy())
        # delta_bbox = detection.to_xywh().copy() - detection.to_xywh().copy()
        # tmp_conds = np.concatenate((detection.to_xywh().copy(), delta_bbox))
        # self.tracks_3d[self.track3d_count].conds.append(tmp_conds)
        self.track_id_3d += 2
        self.track3d_count += 1

    def initiate_trajectory_2d(self, detection):
        self.tracks_2d.append(Track_2D(self.track_id_2d, self.min_frames, self.max_age, detection.tlwh, detection.score))
        self.tracks_2d[self.track2d_count].xywh_omemory.append(detection.to_xywh().copy())
        self.tracks_2d[self.track2d_count].xywh_pmemory.append(detection.to_xywh().copy())
        self.tracks_2d[self.track2d_count].xywh_amemory.append(detection.to_xywh().copy())
        delta_bbox = detection.to_xywh().copy() - detection.to_xywh().copy()
        tmp_conds = np.concatenate((detection.to_xywh().copy(), delta_bbox))
        self.tracks_2d[self.track2d_count].conds.append(tmp_conds)
        self.track_id_2d += 2
        self.track2d_count += 1
