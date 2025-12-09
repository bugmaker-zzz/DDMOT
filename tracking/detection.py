import numpy as np


class Detection_2D(object):
    def __init__(self, tlwh, score):
        '''
        :param tlwh:  top_left x   top_left y    width   height
        :param additional_info:
        '''
        self.tlwh = tlwh
        self.score = score
        # self.feature = np.asarray(feature, dtype=np.float32)
        self.center = self.to_xywh()[:2]

    def to_x1y1x2y2(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """
        Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
    def to_xyah_new(self, tlwh):
        """
        Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
    def to_xywh(self):
        ret = self.tlwh.copy()
        ret[:2] = ret[:2] + ret[2:] / 2
        return ret
# new add 
class Detection_2D_Fusion(object):
    def __init__(self, x1y1x2y2, score):
        '''
        :param tlwh:  top_left x   top_left y    width   height
        :param additional_info:
        '''
        self.x1y1x2y2 = x1y1x2y2
        self.tlwh = np.asarray([self.x1y1x2y2[0], self.x1y1x2y2[1], self.x1y1x2y2[2] - self.x1y1x2y2[0], self.x1y1x2y2[3] - self.x1y1x2y2[1]], dtype=np.float64)
        self.score = score
        # self.feature = np.asarray(feature, dtype=np.float32)

    def to_x1y1x2y2(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """
        Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
    def to_xyah_new(self, tlwh):
        """
        Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
    def to_xywh(self):
        ret = self.tlwh.copy()
        ret[:2] = ret[:2] + ret[2:] / 2
        return ret

class Detection_3D_Fusion(object):
    def __init__(self, BBox_3D, additional_info):
        self.bbox = BBox_3D
        self.additional_info = additional_info
        # new add
        self.tlwh = np.asarray([self.additional_info[2], self.additional_info[3], self.additional_info[4] - self.additional_info[2], self.additional_info[5] - self.additional_info[3]], dtype=np.float64)
        self.center = np.asarray([self.bbox[0], self.bbox[1], self.bbox[2]])

    # new add
    def to_xywh(self):
        ret = self.tlwh.copy()
        ret[:2] = ret[:2] + ret[2:] / 2
        return ret

class Detection_3D_only(object):
    def __init__(self, BBox_3D, additional_info):
        self.bbox = BBox_3D
        self.additional_info = additional_info
        self.score = self.additional_info[6]
        # new add
        self.tlwh = np.asarray([self.additional_info[2], self.additional_info[3], self.additional_info[4] - self.additional_info[2], self.additional_info[5] - self.additional_info[3]], dtype=np.float64)
        self.center = np.asarray([self.bbox[0], self.bbox[1], self.bbox[2]])

    # new add
    def to_xywh(self):
        ret = self.tlwh.copy()
        ret[:2] = ret[:2] + ret[2:] / 2
        return ret

