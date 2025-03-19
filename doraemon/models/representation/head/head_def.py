import yaml
from .arcface import ArcFace
from .circleloss import CircleLoss
from .mv_softmax import MV_Softmax
from .magface import MagFace

class HeadFactory:
    """Factory to produce head according to the head_conf.yaml
    
    Attributes:
        head_type(str): which head will be produce.
        head_param(dict): parsed params and it's value.
    """
    def __init__(self, head_config):
        for k, v in head_config.items(): self.head_type, self.head_param = k, v

    def get_head(self):
        if self.head_type == 'arcface':
            feat_dim = self.head_param['feat_dim'] # dimension of the output features, e.g. 512 
            num_class = self.head_param['num_class'] # number of classes in the training set.
            margin_arc = self.head_param['margin_arc'] # cos(theta + margin_arc).
            margin_am = self.head_param['margin_am'] # cos_theta - margin_am.
            scale = self.head_param['scale'] # the scaling factor for cosine values.
            head = ArcFace(feat_dim, num_class, margin_arc, margin_am, scale)

        elif self.head_type == 'magface':
            feat_dim = self.head_param['feat_dim'] # dimension of the output features, e.g. 512
            num_class = self.head_param['num_class'] # number of classes in the training set.
            margin_am = self.head_param['margin_am'] # cos_theta - margin_am.
            scale = self.head_param['scale'] # the scaling factor for cosine values.
            l_a = self.head_param['l_a']
            u_a = self.head_param['u_a']
            l_margin = self.head_param['l_margin']
            u_margin = self.head_param['u_margin']
            lamda = self.head_param['lamda']
            head = MagFace(feat_dim, num_class, margin_am, scale, l_a, u_a, l_margin, u_margin, lamda)

        elif self.head_type == 'circleloss':
            feat_dim = self.head_param['feat_dim'] # dimension of the output features, e.g. 512 
            num_class = self.head_param['num_class'] # number of classes in the training set.
            margin = self.head_param['margin'] # O_p = 1 + margin, O_n = -margin.
            gamma = self.head_param['gamma'] # the scale facetor.
            head = CircleLoss(feat_dim, num_class, margin, gamma)

        elif self.head_type == 'mv-softmax':
            feat_dim = self.head_param['feat_dim'] # dimension of the output features, e.g. 512 
            num_class = self.head_param['num_class'] # number of classes in the training set.
            is_am = self.head_param['is_am'] # am-softmax for positive samples.
            margin = self.head_param['margin'] # margin for positive samples.
            mv_weight = self.head_param['mv_weight'] # weight for hard negtive samples.
            scale = self.head_param['scale'] # the scaling factor for cosine values.
            head = MV_Softmax(feat_dim, num_class, is_am, margin, mv_weight, scale)

        else:
            raise NotImplemented("only arcface, magface, circleloss and mv-softmax are supported now !")
        return head
