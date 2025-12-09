import numpy as np
from tracking import kalman_filter_2d
from tracking.matching import associate_dets_to_trks_fusion, associate_dets_to_trks, associate_2D_to_3D_tracking, associate_dets_to_trks_by_extended_bbox_2d   #new add
from tracking.track_2d import Track_2D
from tracking.kalman_fileter_3d import  KalmanBoxTracker
from tracking.track_3d import Track_3D


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
        self.kf_2d = kalman_filter_2d.KalmanFilter()

        # new add 用于计算动态生命周期的尺度系数α和偏移系数β
        self.alpha = self.cfg[category].alpha
        self.beta = self.cfg[category].beta
        # new add 仿照bytetrack进行级联匹配（高置信度匹配+低置信度匹配）
        self.track_threshold_high = self.cfg.track_threshold_high
        self.track_threshold_low = self.cfg.track_threshold_low
        # new add 用于解决队列下标问题
        self.track2d_count = 0

    # 基于扩散模型的运动预测
    # model:D2MP
    # img_w, img_h:图片长宽
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

    def predict_3d(self):
        for track in self.tracks_3d:
            track.predict_3d(track.kf_3d)

    def predict_2d(self):
        for track in self.tracks_2d:
            track.predict_2d(self.kf_2d)

    def ego_motion_compensation(self, frame, calib_file, oxts):
        for track in self.tracks_3d:
            track.ego_motion_compensation_3d(frame, calib_file, oxts)

    def update(self, dets_3d_fusion, dets_3d_only, dets_2d_only):
        # 1st Level of Association
        matched_fusion_idx, unmatched_dets_fusion_idx, unmatched_trks_fusion_idx = associate_dets_to_trks_fusion(
            dets_3d_fusion, self.tracks_3d, self.cost_3d, self.threshold_3d, metric='match_3d')
        print("1st len:", len(matched_fusion_idx), len(unmatched_dets_fusion_idx), len(unmatched_trks_fusion_idx))
        for detection_idx, track_idx in matched_fusion_idx:
            self.tracks_3d[track_idx].update_3d(dets_3d_fusion[detection_idx])
            self.tracks_3d[track_idx].state = 2
            self.tracks_3d[track_idx].fusion_time_update = 0
        for track_idx in unmatched_trks_fusion_idx:
            self.tracks_3d[track_idx].fusion_time_update += 1
            self.tracks_3d[track_idx].mark_missed()
        for detection_idx in unmatched_dets_fusion_idx:
            self.initiate_trajectory_3d(dets_3d_fusion[detection_idx])

        # matched_fusion_idx, unmatched_dets_fusion_idx, unmatched_trks_fusion_idx = associate_dets_to_trks_fusion(
        #     dets_3d_fusion, self.tracks_3d, self.cost_3d, self.threshold_3d, metric='match_3d')
        # # print("1st len:", len(matched_fusion_idx), len(unmatched_dets_fusion_idx), len(unmatched_trks_fusion_idx))
        # for detection_idx, track_idx in matched_fusion_idx:
        #     self.tracks_3d[track_idx].update_3d(dets_3d_fusion[detection_idx])
        #     self.tracks_3d[track_idx].state = 2
        #     self.tracks_3d[track_idx].fusion_time_update = 0
        # for track_idx in unmatched_trks_fusion_idx:
        #     self.tracks_3d[track_idx].fusion_time_update += 1
        #     self.tracks_3d[track_idx].mark_missed()
        # for detection_idx in unmatched_dets_fusion_idx:
        #     self.initiate_trajectory_3d(dets_3d_fusion[detection_idx])

        # 如果更换相似度函数，需要修改first_da_threshold和second_da_threshold
        #  2nd Level of Association
        # self.unmatch_tracks_3d1 = [t for t in self.tracks_3d if t.time_since_update > 0]
        # dets_3d_only_high_score = []
        # dets_3d_only_low_score = []
        # first_da_threshold = 0.005
        # second_da_threshold = -1
        # for det in dets_3d_only:
        #     if det.score > self.track_threshold_low and det.score < self.track_threshold_high:
        #         dets_3d_only_low_score.append(det)
        #     if det.score >= self.track_threshold_high:
        #         dets_3d_only_high_score.append(det)
        # # 级联匹配第一阶段，高置信度匹配
        # print("1:", len(self.unmatch_tracks_3d1))
        # matched_3d_only_idx_first, unmatched_3d_dets_only_idx_first, unmatched_3d_trks_idx_first = associate_dets_to_trks_fusion(
        #     dets_3d_only_high_score, self.unmatch_tracks_3d1, self.cost_3d, first_da_threshold, metric='match_3d')
        # index_to_delete_first = []
        # for detection_idx, track_idx in matched_3d_only_idx_first:
        #     for index, t in enumerate(self.tracks_3d):
        #         if t.track_id_3d == self.unmatch_tracks_3d1[track_idx].track_id_3d:
        #             t.update_3d(dets_3d_only[detection_idx])
        #             index_to_delete_first.append(track_idx)
        #             break
        # # for detection_idx_first, track_idx_first in matched_3d_only_idx_first:
        # #     self.tracks_3d[track_idx_first].update_3d(dets_3d_only[detection_idx_first])
        # #     index_to_delete_first.append(track_idx_first)
        # # 级联匹配第二阶段，低置信度匹配
        # self.unmatch_tracks_3d1 = [self.unmatch_tracks_3d1[i] for i in range(len(self.unmatch_tracks_3d1)) if i not in index_to_delete_first]
        # print("2:", len(self.unmatch_tracks_3d1))
        # index_to_delete_second = []
        # matched_3d_only_idx_second, unmatched_3d_dets_only_idx_second, unmatched_3d_trks_idx_second = associate_dets_to_trks_fusion(
        #     dets_3d_only_low_score, self.unmatch_tracks_3d1, self.cost_3d, second_da_threshold, metric='match_3d')
        # for detection_idx, track_idx in matched_3d_only_idx_second:
        #     for index, t in enumerate(self.tracks_3d):
        #         if t.track_id_3d == self.unmatch_tracks_3d1[track_idx].track_id_3d:
        #             t.update_3d(dets_3d_only[detection_idx])
        #             index_to_delete_second.append(track_idx)
        #             break
        # # for detection_idx_second, track_idx_second in matched_3d_only_idx_second:
        # #     self.tracks_3d[track_idx_second].update_3d(dets_3d_only[detection_idx_second])
        # #     index_to_delete_second.append(track_idx_second)
        # for detection_idx in unmatched_3d_dets_only_idx_second:
        #     self.initiate_trajectory_3d(dets_3d_only[detection_idx])
        # self.unmatch_tracks_3d1 = [self.unmatch_tracks_3d1[i] for i in range(len(self.unmatch_tracks_3d1)) if i not in index_to_delete_second]
        # print("3:", len(self.unmatch_tracks_3d1))
        # self.unmatch_tracks_3d2 = [t for t in self.tracks_3d if t.time_since_update == 0 and t.hits == 1]
        # self.unmatch_tracks_3d = self.unmatch_tracks_3d1 + self.unmatch_tracks_3d2

        self.unmatch_tracks_3d1 = [t for t in self.tracks_3d if t.time_since_update > 0]
        # matched_only_idx, unmatched_dets_only_idx, _ = associate_dets_to_trks_fusion(
        #     dets_3d_only, self.unmatch_tracks_3d1, self.cost_3d, self.threshold_3d, metric='match_3d')
        matched_only_idx, unmatched_dets_only_idx, _ = associate_dets_to_trks_fusion(
            dets_3d_only, self.unmatch_tracks_3d1, self.cost_3d, self.threshold_3d, metric='match_3d')
        print("2nd len:", len(matched_only_idx), len(unmatched_dets_only_idx))
        index_to_delete = []
        for detection_idx, track_idx in matched_only_idx:
            for index, t in enumerate(self.tracks_3d):
                if t.track_id_3d == self.unmatch_tracks_3d1[track_idx].track_id_3d:
                    t.update_3d(dets_3d_only[detection_idx])
                    index_to_delete.append(track_idx)
                    break
        self.unmatch_tracks_3d1 = [self.unmatch_tracks_3d1[i] for i in range(len(self.unmatch_tracks_3d1)) if i not in index_to_delete]
        print("self.unmatch_tracks_3d1:", len(self.unmatch_tracks_3d1))
        for detection_idx in unmatched_dets_only_idx:
            self.initiate_trajectory_3d(dets_3d_only[detection_idx])
        self.unmatch_tracks_3d2 = [t for t in self.tracks_3d if t.time_since_update == 0 and t.hits == 1]
        self.unmatch_tracks_3d = self.unmatch_tracks_3d1 + self.unmatch_tracks_3d2


        # 3rd Level of Association
        # new add 仿照bytetrack级联匹配
        # dets_2d_high_score = []
        # dets_2d_low_score = []
        # first_da_threshold = 0.5
        # second_da_threshold = 0.3
        # for det in dets_2d_only:
        #     if det.score > self.track_threshold_low and det.score < self.track_threshold_high:
        #         dets_2d_low_score.append(det)
        #     else:
        #         dets_2d_high_score.append(det)
        # # 级联匹配第一阶段，高置信度匹配
        # matched_first, unmatch_trks_first, unmatch_dets_first = \
        #     associate_dets_to_trks_fusion(self.tracks_2d, dets_2d_high_score, self.cost_2d, first_da_threshold, metric='match_2d')
        # for track_idx, detection_idx in matched_first:
        #     # self.tracks_2d[track_idx].update_2d(self.kf_2d, dets_2d_only[detection_idx])
        #     self.tracks_2d[track_idx].update_2d(dets_2d_only[detection_idx])
        # # 级联匹配第二阶段，低置信度匹配
        # leftover_tracks = [self.tracks_2d[i] for i in unmatch_trks_first]
        # matched_second, unmatch_trks_second, unmatch_dets_second = \
        #     associate_dets_to_trks_fusion(leftover_tracks, dets_2d_low_score, self.cost_2d, second_da_threshold, metric='match_2d')
        # for track_idx, detection_idx in matched_second:
        #     # self.tracks_2d[track_idx].update_2d(self.kf_2d, dets_2d_only[detection_idx])
        #     self.tracks_2d[track_idx].update_2d(dets_2d_only[detection_idx])
        # for track_idx in unmatch_trks_second:
        #     self.tracks_2d[track_idx].mark_missed()
        # for detection_idx in unmatch_dets_first:
        #     self.initiate_trajectory_2d(dets_2d_only[detection_idx])
        # # new add
        # for t in self.tracks_2d:
        #     if t.is_deleted():
        #         self.track2d_count -= 1
        # self.tracks_2d = [t for t in self.tracks_2d if not t.is_deleted()]

        # new add 扩大2Dbbox进行二次匹配
        # matched_first, unmatch_trks_first, unmatch_dets_first = \
        #     associate_dets_to_trks_fusion(self.tracks_2d, dets_2d_only, self.cost_2d, self.threshold_2d, metric='match_2d')
        # for track_idx, detection_idx in matched_first:
        #     # self.tracks_2d[track_idx].update_2d(self.kf_2d, dets_2d_only[detection_idx])
        #     self.tracks_2d[track_idx].update_2d(dets_2d_only[detection_idx])
        # unmatch_trks_first = [self.tracks_2d[unmatch_trks_first[i]] for i in range(len(unmatch_trks_first))]
        # unmatch_dets_first = [dets_2d_only[unmatch_dets_first[i]] for i in range(len(unmatch_dets_first))]
        # matched_second, unmatch_trks_second, unmatch_dets_second = \
        #     associate_dets_to_trks_by_extended_bbox_2d(unmatch_trks_first, unmatch_dets_first, self.cost_2d, 0.3, metric='match_2d')
        # for track_idx, detection_idx in matched_second:
        #     # self.tracks_2d[track_idx].update_2d(self.kf_2d, dets_2d_only[detection_idx])
        #     self.tracks_2d[track_idx].update_2d(dets_2d_only[detection_idx])
        # for track_idx in unmatch_trks_second:
        #     self.tracks_2d[track_idx].mark_missed()
        # for detection_idx in unmatch_dets_second:
        #     self.initiate_trajectory_2d(dets_2d_only[detection_idx])
        # # new add
        # for t in self.tracks_2d:
        #     if t.is_deleted():
        #         self.track2d_count -= 1
        # self.tracks_2d = [t for t in self.tracks_2d if not t.is_deleted()]

        # matched, unmatch_trks, unmatch_dets = \
        #     associate_dets_to_trks_fusion(self.tracks_2d, dets_2d_only, self.cost_2d, self.threshold_2d, metric='match_2d')
        # matched, unmatch_dets, unmatch_trks = \
        #     associate_dets_to_trks_fusion(dets_2d_only, self.tracks_2d, self.cost_2d, self.threshold_2d, metric='match_2d')
        matched, unmatch_dets, unmatch_trks = \
            associate_dets_to_trks(dets_2d_only, self.tracks_2d, self.cost_2d, self.threshold_2d, metric='match_2d')
        print("3rd len:", len(matched), len(unmatch_dets), len(unmatch_trks))
        # for track_idx, detection_idx in matched:
        #     # self.tracks_2d[track_idx].update_2d(self.kf_2d, dets_2d_only[detection_idx])
        #     self.tracks_2d[track_idx].update_2d(dets_2d_only[detection_idx])
        for detection_idx, track_idx in matched:
            # self.tracks_2d[track_idx].update_2d(self.kf_2d, dets_2d_only[detection_idx])
            self.tracks_2d[track_idx].update_2d(dets_2d_only[detection_idx])
        for track_idx in unmatch_trks:
            self.tracks_2d[track_idx].mark_missed()
        for detection_idx in unmatch_dets:
            self.initiate_trajectory_2d(dets_2d_only[detection_idx])
        # new add
        for t in self.tracks_2d:
            if t.is_deleted():
                self.track2d_count -= 1
        # !!!!!
        self.tracks_2d = [t for t in self.tracks_2d if not t.is_deleted()]

        # 4th Level of Association
        matched_track_2d, unmatch_tracks_2d = associate_2D_to_3D_tracking(self.tracks_2d, self.unmatch_tracks_3d, self.threshold_2d)
        print("4th len:", len(matched_track_2d), len(unmatch_tracks_2d))
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
        self.tracks_3d = [t for t in self.tracks_3d if not t.is_deleted()]

    def initiate_trajectory_3d(self, detection):
        self.kf_3d = KalmanBoxTracker(detection.bbox)
        self.additional_info = detection.additional_info
        pose = np.concatenate(self.kf_3d.kf.x[:7], axis=0)
        self.tracks_3d.append(Track_3D(pose, self.kf_3d, self.track_id_3d, self.min_frames, self.max_age, self.additional_info))
        self.track_id_3d += 2

    def initiate_trajectory_2d(self, detection):
        # mean, covariance = self.kf_2d.initiate(detection.to_xyah())
        self.tracks_2d.append(Track_2D(self.track_id_2d, self.min_frames, self.max_age, detection.tlwh, detection.score))
        self.tracks_2d[self.track2d_count].xywh_omemory.append(detection.to_xywh().copy())
        self.tracks_2d[self.track2d_count].xywh_pmemory.append(detection.to_xywh().copy())
        self.tracks_2d[self.track2d_count].xywh_amemory.append(detection.to_xywh().copy())
        delta_bbox = detection.to_xywh().copy() - detection.to_xywh().copy()
        tmp_conds = np.concatenate((detection.to_xywh().copy(), delta_bbox))
        self.tracks_2d[self.track2d_count].conds.append(tmp_conds)
        self.track_id_2d += 2
        self.track2d_count += 1