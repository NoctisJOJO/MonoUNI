import torch
import torch.nn as nn
import numpy as np

from lib.backbones.resnet import resnet50
from lib.backbones.dla import dla34
from lib.backbones.dlaup import DLAUp
from lib.backbones.dlaup import DLAUpv2

import torchvision.ops.roi_align as roi_align
from lib.losses.loss_function import extract_input_from_tensor
from lib.helpers.decode_helper import _topk,_nms

def weights_init_xavier(m):
    classname = m.__class__.__name__  # 获取该模块的类名
    if classname.find('Linear') != -1:  # 
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
 
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        try:
            if m.bias:
                nn.init.constant_(m.bias, 0.0)
        except:
            nn.init.constant_(m.bias, 0.0)

class MonoUNI(nn.Module):
    def __init__(self, backbone='dla34', neck='DLAUp', downsample=4, mean_size=None,cfg=None):
        assert downsample in [4, 8, 16, 32]
        super().__init__()


        self.backbone = globals()[backbone](pretrained=True, return_levels=True)
        self.head_conv = 256  # default setting for head conv
        self.mean_size = nn.Parameter(torch.tensor(mean_size,dtype=torch.float32),requires_grad=False)
        self.cls_num = mean_size.shape[0]
        channels = self.backbone.channels  # channels list for feature maps generated by backbone
        self.first_level = int(np.log2(downsample))
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.feat_up = globals()[neck](channels[self.first_level:], scales_list=scales)
        self.cfg = cfg
        self.bin_size = 1
        if self.cfg['multi_bin']:
            min_f = 2250
            max_f = 2800
            interval_min = torch.tensor(self.cfg['interval_min']) - 4.5
            self.interval_min = interval_min / max_f
            interval_max = torch.tensor(self.cfg['interval_max']) + 4.5 
            self.interval_max = interval_max / min_f
            self.bin_size = 5





        # initialize the head of pipeline, according to heads setting.
        self.heatmap = nn.Sequential(nn.Conv2d(channels[self.first_level], self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(self.head_conv, self.cls_num, kernel_size=1, stride=1, padding=0, bias=True))
        self.offset_2d = nn.Sequential(nn.Conv2d(channels[self.first_level], self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(self.head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True))
        self.size_2d = nn.Sequential(nn.Conv2d(channels[self.first_level], self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(self.head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True))

        self.offset_3d = nn.Sequential(nn.Conv2d(channels[self.first_level] *2 +self.cls_num+2, self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm2d(self.head_conv),
                                     nn.ReLU(inplace=True),nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(self.head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True))
        self.size_3d = nn.Sequential(nn.Conv2d(channels[self.first_level]  *2 +self.cls_num+2, self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm2d(self.head_conv),
                                     nn.ReLU(inplace=True),nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(self.head_conv, 3, kernel_size=1, stride=1, padding=0, bias=True))
        self.heading = nn.Sequential(nn.Conv2d(channels[self.first_level]  *2 +self.cls_num+2, self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm2d(self.head_conv),
                                     nn.ReLU(inplace=True),nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(self.head_conv, 24, kernel_size=1, stride=1, padding=0, bias=True))

        self.vis_depth = nn.Sequential(nn.Conv2d(channels[self.first_level] *2 +2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(self.head_conv, self.bin_size, kernel_size=1, stride=1, padding=0, bias=True))
        self.att_depth = nn.Sequential(nn.Conv2d(channels[self.first_level] *2 +2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(self.head_conv, self.bin_size, kernel_size=1, stride=1, padding=0, bias=True))
        self.vis_depth_uncer = nn.Sequential(nn.Conv2d(channels[self.first_level] *2 +2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
                                             nn.LeakyReLU(inplace=True),
                                             nn.Conv2d(self.head_conv, self.bin_size, kernel_size=1, stride=1, padding=0, bias=True))
        self.att_depth_uncer = nn.Sequential(nn.Conv2d(channels[self.first_level] *2 +2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
                                             nn.LeakyReLU(inplace=True),
                                             nn.Conv2d(self.head_conv, self.bin_size, kernel_size=1, stride=1, padding=0, bias=True))
        if self.cfg['multi_bin']:
            self.depth_bin = nn.Sequential(nn.Conv2d(channels[self.first_level] *2  +2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
                                                nn.LeakyReLU(inplace=True),nn.AdaptiveAvgPool2d(1),
                                                nn.Conv2d(self.head_conv, 10, kernel_size=1, stride=1, padding=0, bias=True))
            self.depth_bin.apply(weights_init_xavier)
        # init layers
        self.heatmap[-1].bias.data.fill_(-2.19)
        self.fill_fc_weights(self.offset_2d)
        self.fill_fc_weights(self.size_2d)

        self.vis_depth.apply(weights_init_xavier)
        self.att_depth.apply(weights_init_xavier)
        self.offset_3d.apply(weights_init_xavier)
        self.size_3d.apply(weights_init_xavier)
        self.heading.apply(weights_init_xavier)
        self.vis_depth_uncer.apply(weights_init_xavier)
        self.att_depth_uncer.apply(weights_init_xavier)

    def forward(self, input, coord_ranges,calibs, targets=None, K=100, mode='train', calib_pitch_sin=None, calib_pitch_cos=None):
        device_id = input.device
        BATCH_SIZE = input.size(0)

        feat = self.backbone(input)
        feat = self.feat_up(feat[self.first_level:])
        '''
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(feat)
        '''
        ret = {}
        ret['heatmap']=self.heatmap(feat)
        ret['offset_2d']=self.offset_2d(feat)
        ret['size_2d']=self.size_2d(feat)
        #two stage
        assert(mode in ['train','val','test'])
        if mode=='train':   #extract train structure in the train (only) and the val mode
            inds,cls_ids = targets['indices'],targets['cls_ids']
            masks = targets['mask_2d']
        else:    #extract test structure in the test (only) and the val mode
            inds,cls_ids = _topk(_nms(torch.clamp(ret['heatmap'].sigmoid(), min=1e-4, max=1 - 1e-4)), K=K)[1:3]
            masks = torch.ones(inds.size()).type(torch.bool).to(device_id)
        ret.update(self.get_roi_feat(feat,inds,masks,ret,calibs,coord_ranges,cls_ids,mode, calib_pitch_sin, calib_pitch_cos))
        return ret

    def get_roi_feat_by_mask(self,feat,box2d_maps,inds,mask,calibs,coord_ranges,cls_ids,mode, calib_pitch_sin=None, calib_pitch_cos=None):
        BATCH_SIZE,_,HEIGHT,WIDE = feat.size()
        device_id = feat.device
        num_masked_bin = mask.sum()
        res = {}
        if num_masked_bin!=0:
            #get box2d of each roi region
            scale_box2d_masked = extract_input_from_tensor(box2d_maps,inds,mask)
            #get roi feature
            # print(torch.max(box2d_masked[:,0]))
            # print(torch.max(box2d_masked[:,1]))
            # print(torch.max(box2d_masked[:,2]))
            # print(torch.max(box2d_masked[:,3]))
            # print(torch.max(box2d_masked[:,4]))
            roi_feature_masked = roi_align(feat,scale_box2d_masked,[7,7])


            box2d_masked_copy = torch.zeros_like(scale_box2d_masked)
            box2d_masked_copy[:,0] = scale_box2d_masked[:,0]
            # box2d_masked_copy[:,1] = 0
            # box2d_masked_copy[:,2] = 0 
            box2d_masked_copy[:,3] = 239
            box2d_masked_copy[:,4] = 127
            roi_feature_global = roi_align(feat,box2d_masked_copy,[7,7])
            roi_feature_masked_ = torch.cat((roi_feature_masked,roi_feature_global),1)
            


            # #get coord range of each roi
            coord_ranges_mask2d = coord_ranges[scale_box2d_masked[:,0].long()]

            #map box2d coordinate from feature map size domain to original image size domain
            box2d_masked = torch.cat([scale_box2d_masked[:,0:1],
                       scale_box2d_masked[:,1:2]/WIDE  *(coord_ranges_mask2d[:,1,0:1]-coord_ranges_mask2d[:,0,0:1])+coord_ranges_mask2d[:,0,0:1],
                       scale_box2d_masked[:,2:3]/HEIGHT*(coord_ranges_mask2d[:,1,1:2]-coord_ranges_mask2d[:,0,1:2])+coord_ranges_mask2d[:,0,1:2],
                       scale_box2d_masked[:,3:4]/WIDE  *(coord_ranges_mask2d[:,1,0:1]-coord_ranges_mask2d[:,0,0:1])+coord_ranges_mask2d[:,0,0:1],
                       scale_box2d_masked[:,4:5]/HEIGHT*(coord_ranges_mask2d[:,1,1:2]-coord_ranges_mask2d[:,0,1:2])+coord_ranges_mask2d[:,0,1:2]],1)
            roi_calibs = calibs[box2d_masked[:,0].long()]
            roi_sin = calib_pitch_sin[box2d_masked[:,0].long()]
            roi_cos = calib_pitch_cos[box2d_masked[:,0].long()]
            # #project the coordinate in the normal image to the camera coord by calibs
            coords_in_camera_coord = torch.cat([self.project2rect(roi_calibs,torch.cat([box2d_masked[:,1:3],torch.ones([num_masked_bin,1]).to(device_id)],-1))[:,:2],
                                          self.project2rect(roi_calibs,torch.cat([box2d_masked[:,3:5],torch.ones([num_masked_bin,1]).to(device_id)],-1))[:,:2]],-1)
            box2d_v1 = box2d_masked[:,2:3]
            box2d_v2 = box2d_masked[:,4:5]
            v_maps = torch.cat([box2d_v1+i*(box2d_v2-box2d_v1)/(7-1) for i in range(7)],-1).unsqueeze(2).repeat([1,1,7]).unsqueeze(1).repeat([1,self.bin_size,1,1])
            coords_in_camera_coord = torch.cat([box2d_masked[:,0:1],coords_in_camera_coord],-1)
            # #generate coord maps
            coord_maps = torch.cat([torch.cat([coords_in_camera_coord[:,1:2]+i*(coords_in_camera_coord[:,3:4]-coords_in_camera_coord[:,1:2])/6 for i in range(7)],-1).unsqueeze(1).repeat([1,7,1]).unsqueeze(1),
                                torch.cat([coords_in_camera_coord[:,2:3]+i*(coords_in_camera_coord[:,4:5]-coords_in_camera_coord[:,2:3])/6 for i in range(7)],-1).unsqueeze(2).repeat([1,1,7]).unsqueeze(1)],1)

            # #concatenate coord maps with feature maps in the channel dim
            cls_hots = torch.zeros(num_masked_bin,self.cls_num).to(device_id)
            cls_hots[torch.arange(num_masked_bin).to(device_id),cls_ids[mask].long()] = 1.0
            
            roi_feature_masked = torch.cat([roi_feature_masked_,coord_maps,cls_hots.unsqueeze(-1).unsqueeze(-1).repeat([1,1,7,7])],1)



            scale_depth = torch.clamp((scale_box2d_masked[:,4]-scale_box2d_masked[:,2])*4*2.109375, min=1.0) / \
                          torch.clamp(box2d_masked[:,4]-box2d_masked[:,2], min=1.0)
            scale_depth = scale_depth.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            #compute 3d dimension offset

            size3d_offset = self.size_3d(roi_feature_masked)[:,:,0,0]
            


            vis_depth = self.vis_depth(roi_feature_masked)
            att_depth = self.att_depth(roi_feature_masked)
            vis_depth_uncer = self.vis_depth_uncer(roi_feature_masked)
            att_depth_uncer = self.att_depth_uncer(roi_feature_masked)


            
            if self.cfg['multi_bin']:
                depth_bin = self.depth_bin(roi_feature_masked)[:,:,0,0]
                res['depth_bin']= depth_bin
                vis_depth = torch.sigmoid(vis_depth)

                fx = roi_calibs[:,0,0]
                fy = roi_calibs[:,1,1]
                cy = roi_calibs[:,1,2]
                cy = cy.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                fp = torch.sqrt(1.0/(fx * fx) + 1.0/(fy * fy)) / 1.41421356
                fp = fp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                fy = fy.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                tan_maps = (v_maps - cy) * 1.0 / fy
                pitch_sin = roi_sin.view(roi_calibs.shape[0], 1, 1, 1)
                pitch_cos = roi_cos.view(roi_calibs.shape[0], 1, 1, 1)
                norm_theta = (pitch_cos - pitch_sin * tan_maps).float()
                
                interval_min = self.interval_min.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(vis_depth.shape[0],1,1,1).to(vis_depth.device)
                interval_max = self.interval_max.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(vis_depth.shape[0],1,1,1).to(vis_depth.device)
                if mode=='train':
                    vis_depth_ =   vis_depth * (interval_max-interval_min) + interval_min
                    vis_depth_ = vis_depth_ * norm_theta / fp * scale_depth 
                    ins_depth = vis_depth_ + att_depth
                    ins_depth_uncer = torch.logsumexp(torch.stack([vis_depth_uncer, att_depth_uncer], -1), -1)
                else:
                    depth_bin_1 = torch.softmax(depth_bin[:,:2],-1)
                    depth_bin_2 = torch.softmax(depth_bin[:,2:4],-1)
                    depth_bin_3 = torch.softmax(depth_bin[:,4:6],-1)
                    depth_bin_4 = torch.softmax(depth_bin[:,6:8],-1)
                    depth_bin_5 = torch.softmax(depth_bin[:,8:10],-1)
                    depth_bin = torch.cat((depth_bin_1[:,1:2],depth_bin_2[:,1:2],depth_bin_3[:,1:2],depth_bin_4[:,1:2],depth_bin_5[:,1:2]),-1)
                    _,depth_bin_max_index = torch.max(depth_bin,-1)
                    vis_depth_ =   vis_depth * (interval_max-interval_min) + interval_min
                    vis_depth_ = vis_depth_ * norm_theta / fp * scale_depth 
                    vis_depth = vis_depth_[torch.arange(depth_bin_max_index.shape[0]),depth_bin_max_index]
                    att_depth = att_depth[torch.arange(depth_bin_max_index.shape[0]),depth_bin_max_index]
                    vis_depth_uncer = vis_depth_uncer[torch.arange(depth_bin_max_index.shape[0]),depth_bin_max_index]
                    att_depth_uncer = att_depth_uncer[torch.arange(depth_bin_max_index.shape[0]),depth_bin_max_index]
                    ins_depth = vis_depth + att_depth
                    ins_depth_uncer = torch.logsumexp(torch.stack([vis_depth_uncer, att_depth_uncer], -1), -1)
            else:
                vis_depth = (-vis_depth).exp().squeeze(1)
                att_depth = att_depth.squeeze(1)
                vis_depth = vis_depth * scale_depth.squeeze(-1)
                vis_depth_uncer = vis_depth_uncer[:, 0, :, :]
                att_depth_uncer = att_depth_uncer[:, 0, :, :]
                ins_depth = vis_depth + att_depth
                ins_depth_uncer = torch.logsumexp(torch.stack([vis_depth_uncer, att_depth_uncer], -1), -1)


            

            res['train_tag'] = torch.ones(num_masked_bin).type(torch.bool).to(device_id)
            res['heading'] = self.heading(roi_feature_masked)[:,:,0,0]
            res['vis_depth'] = vis_depth
            res['att_depth'] = att_depth
            res['ins_depth'] = ins_depth
            res['vis_depth_uncer'] = vis_depth_uncer
            res['att_depth_uncer'] = att_depth_uncer
            res['ins_depth_uncer'] = ins_depth_uncer
            res['offset_3d'] = self.offset_3d(roi_feature_masked)[:,:,0,0]
            res['size_3d']= size3d_offset
            
        else:
            res['offset_3d'] = torch.zeros([1,2]).to(device_id)
            res['size_3d'] = torch.zeros([1,3]).to(device_id)
            res['train_tag'] = torch.zeros(1).type(torch.bool).to(device_id)
            res['heading'] = torch.zeros([1,24]).to(device_id)
            res['vis_depth'] = torch.zeros([1,7,7]).to(device_id)
            res['att_depth'] = torch.zeros([1,7,7]).to(device_id)
            res['ins_depth'] = torch.zeros([1,7,7]).to(device_id)
            res['vis_depth_uncer'] = torch.zeros([1,5,7,7]).to(device_id)
            res['att_depth_uncer'] = torch.zeros([1,5,7,7]).to(device_id)
            res['ins_depth_uncer'] = torch.zeros([1,5,7,7]).to(device_id)
            if self.cfg['multi_bin']:
                res['depth_bin'] = torch.zeros([1,10]).to(device_id)

        return res

    def get_roi_feat(self,feat,inds,mask,ret,calibs,coord_ranges,cls_ids,mode, calib_pitch_sin=None, calib_pitch_cos=None):
        BATCH_SIZE,_,HEIGHT,WIDE = feat.size()
        device_id = feat.device
        coord_map = torch.cat([torch.arange(WIDE).unsqueeze(0).repeat([HEIGHT,1]).unsqueeze(0),\
                        torch.arange(HEIGHT).unsqueeze(-1).repeat([1,WIDE]).unsqueeze(0)],0).unsqueeze(0).repeat([BATCH_SIZE,1,1,1]).type(torch.float).to(device_id)
        box2d_centre = coord_map + ret['offset_2d']
        box2d_maps = torch.cat([box2d_centre-ret['size_2d']/2,box2d_centre+ret['size_2d']/2],1)
        box2d_maps = torch.cat([torch.arange(BATCH_SIZE).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat([1,1,HEIGHT,WIDE]).type(torch.float).to(device_id),box2d_maps],1)
        #box2d_maps is box2d in each bin
        res = self.get_roi_feat_by_mask(feat,box2d_maps,inds,mask,calibs,coord_ranges,cls_ids,mode, calib_pitch_sin, calib_pitch_cos)
        return res


    def project2rect(self,calib,point_img):
        c_u = calib[:,0,2]
        c_v = calib[:,1,2]
        f_u = calib[:,0,0]
        f_v = calib[:,1,1]
        b_x = calib[:,0,3]/(-f_u) # relative
        b_y = calib[:,1,3]/(-f_v)
        x = (point_img[:,0]-c_u)*point_img[:,2]/f_u + b_x
        y = (point_img[:,1]-c_v)*point_img[:,2]/f_v + b_y
        z = point_img[:,2]
        centre_by_obj = torch.cat([x.unsqueeze(-1),y.unsqueeze(-1),z.unsqueeze(-1)],-1)
        return centre_by_obj

    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    import torch
    net = CenterNet3D()
    print(net)
    input = torch.randn(4, 3, 384, 1280)
    print(input.shape, input.dtype)
    output = net(input)
