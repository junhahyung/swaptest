import os
import sys
import torch
from torch import nn, optim
from kornia import morphology
import torch.nn.functional as F
from torchvision import transforms

sys.path.append('/home/nas4_user/jaeseonglee/faceswap/vtoonify_tu/backup') 
# d3d_path ='/home/nas4_user/jaeseonglee/faceswap/vtoonify_tu/backup/model/Deep3DFaceRecon_pytorch'
# sys.path.append(d3d_path)
from vgg19 import VGGLoss
from eyegaze import gaze_est_tensor
from model.bisenet.model import BiSeNet
from networks.generator import Generator
from networks.pixelcnnpp import UNetModel
from networks.discriminator import Discriminator
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision


sys.path.append('./rt_gene/rt_gene/src')
from rt_gene.estimate_gaze_pytorch import GazeEstimator
from rt_gene.extract_landmarks_method_base import LandmarkMethodBase




# id_coeffs = coeffs[:, :80]
# exp_coeffs = coeffs[:, 80: 144]
# tex_coeffs = coeffs[:, 144: 224]
# angles = coeffs[:, 224: 227]
# gammas = coeffs[:, 227: 254]
# translations = coeffs[:, 254:]

ACCUM = 0.5 ** (32 / (10 * 1000))

def crop_as_bfm(img, interm_wh, coords):
    B = img.shape[0]
    # LRUB order
    cropped_list = []
    for i in range(B):
        resized_img = F.interpolate(img[i,...][None,...], (int(interm_wh[i][1]),int(interm_wh[i][0])))
        #cropped_img = resized_img[...,int(coords[i][2]):int(coords[i][3]),int(coords[i][0]):int(coords[i][1])]
        cropped_img = torchvision.transforms.functional.crop(resized_img, int(coords[i][2]), int(coords[i][0]) ,224,224)
        cropped_list.append(cropped_img)
    return torch.cat(cropped_list,0)
def backup_tensor(img, mask, interm_wh, coords):
    
    
    w_batch, h_batch = interm_wh[...,0:1], interm_wh[...,1:]
    w0 = h0 = 1024
    left_batch,right_batch,up_batch,below_batch = torch.chunk(coords, 4, dim=-1)
    #import pdb;pdb.set_trace()
    B = img.shape[0]
    original_image_stack = []; original_mask_stack = []
    for i in range(B):
        w = w_batch[i]; h=h_batch[i]

        #print(w, h)
        #print(w_batch, h_batch)

        below = below_batch[i]; up = up_batch[i]; right = right_batch[i]; left=left_batch[i]
        #print(w,'???')
        img_canvas = torch.zeros((1,3,h,w)).to(img.device)
        mask_canvas = torch.zeros((1,1,h,w)).to(img.device)
        
        start_h=0; start_w=0; end_h=int(below-up); end_w=int(right-left)
        right_interm = right; left_interm = left; up_interm = up; below_interm = below
        if right>w:
            end_w -= int(abs(right-w))
            right_interm = w
        if below>h:
            end_h -= int(abs(below-h))
            below_interm = h 
        if left<0:
            start_w += int(abs(left))
            left_interm = 0
        if up<0:
            start_h += int(abs(up))
            up_interm = 0
        #import pdb;pdb.set_trace()
        img_canvas[...,int(up_interm):int(below_interm),int(left_interm):int(right_interm)] \
                    = img[i:i+1, :, start_h : end_h, start_w : end_w]#.unsqueeze(0)
        mask_canvas[...,int(up_interm):int(below_interm),int(left_interm):int(right_interm)] \
                    = mask[i:i+1, :, start_h : end_h, start_w : end_w]#.unsqueeze(0)
        #img_canvas[...,int(left):int(right),int(up):int(below)] = img
        ####
        img_canvas_original = F.interpolate(img_canvas,(256,256))
        mask_canvas_original = F.interpolate(mask_canvas,(256,256))
        original_image_stack.append(img_canvas_original)
        original_mask_stack.append(mask_canvas_original)
    
    return torch.cat(original_image_stack,0), torch.cat(original_mask_stack,0)


def requires_grad(net, flag=True):
    for p in net.parameters():
        p.requires_grad = flag

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def denorm(x):
    return (x / 2 + 0.5).clamp(0, 1)

def get_parse(parsemap, type, device):
    # index starts from 1 bc bg is 0
    # ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r', 
    # 'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
    if type=='partial':
        designate_list = [2,3,4,5,10,12,13]# 11이 이빨인데 이건 안넣는게 나을 듯
    elif type=='whole':
        designate_list = [1,2,3,4,5,10,11,12,13]
    elif type=='upper':# used to exclusion part
        #designate_list = [0, 17, 18]#! bg/hair/hat
        designate_list = [17, 18]
        designate_list = [0,6,17,18]#! eg/hair/hat/bg
    elif type=='aguzi':
        designate_list = [11]
    parsemap = parsemap.argmax(1).unsqueeze(-1).detach().cpu().numpy()
    
    for idx, i in enumerate(designate_list):
        if idx==0:
            result_map = (parsemap==i)
        else:
            result_map += (parsemap==i)
    KERNEL = torch.ones(7,7).to(device)
    binary_mask = F.interpolate(torch.tensor(result_map, dtype=torch.float32).permute(0,3,1,2).to(device), 256)

    if type=='partial':
        dilated_mask = morphology.dilation(binary_mask, KERNEL)
    else:
        dilated_mask = binary_mask
    return dilated_mask

class Trainer(nn.Module):
    def __init__(self, args, device, rank, use_ddp=True):
        super(Trainer, self).__init__()

        self.args = args
        self.batch_size = args.batch_size
        self.parser_type = args.parser_type
        
        if self.args.use_texture_cond :
            use_texture_cond=True
        
        else:
            use_texture_cond=False
        if self.args.gaze_img_cond:
            input_cond = True
        else:
            input_cond = False
        
        self.gen = UNetModel(in_channels=args.in_channels, use_texture_cond=use_texture_cond, geo_type=self.args.geo_type, old=args.old).to(device)
        self.dis = Discriminator(args.size, args.channel_multiplier, input_cond = input_cond).to(device)

        if self.args.turn_on_ema:
            self.gen_ema = UNetModel(in_channels=args.in_channels, use_texture_cond=use_texture_cond, old=args.old).to(device)
            self.gen_ema.eval()
            accumulate(self.gen_ema, self.gen, 0)
        # distributed computing
        if use_ddp:
            self.gen = DDP(self.gen, device_ids=[rank], find_unused_parameters=True)
            self.dis = DDP(self.dis, device_ids=[rank], find_unused_parameters=True)
        #self.gaze_estimator = DDP(self.gaze_estimator, device_ids=[rank], find_unused_parameters=True)
        #self.landmark_estimator = DDP(self.landmark_estimator, device_ids=[rank], find_unused_parameters=True)

        g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
        d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

        self.g_optim = optim.Adam(
            self.gen.parameters(),
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
        )

        self.d_optim = optim.Adam(
            self.dis.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio)
        )

        self.criterion_vgg = VGGLoss().to(rank)
        if self.parser_type == 'bisenet':
            BiSeNet_path = '/home/nas4_user/jaeseonglee/faceswap/vtoonify_tu/backup/checkpoint/faceparsing.pth'
            
            self.parsingpredictor = BiSeNet(n_classes=19)
            self.parsingpredictor.load_state_dict(torch.load(BiSeNet_path, map_location=lambda storage, loc: storage))
            self.parsingpredictor.to(device).eval()
        elif self.parser_type == 'segnext':
            pass
        else:
            pass

        if args.render_realtime:
            sys.path.append('/home/nas4_user/jaeseonglee/faceswap/vtoonify_tu/backup/model/Deep3DFaceRecon_pytorch')
            from options.test_options import TestOptions
            from models import create_model
            d3d_opt = TestOptions().parse()
            d3d_opt.use_opengl=False
            self.d3d = create_model(d3d_opt)

            self.d3d.setup(d3d_opt)
            # self.d3d.device = device #! IDK this or next line is correct;;
            self.d3d.device = rank
            self.d3d.parallelize()
            self.d3d.eval()

        if args.id_loss:
            sys.path.append('/home/nas4_user/jaeseonglee/faceswap/vtoonify_tu/backup')
            from id_loss import IDLoss #!09/08 일단 살려놓자
            self.criterion_id = IDLoss().to(rank)
        if args.parameter_loss_deca:
            sys.path.append('/home/nas4_user/jaeseonglee/faceswap/vtoonify_tu/backup')
            from model.DECA.decalib.deca import DECA
            from model.DECA.decalib.utils.tensor_cropper import Cropper
            from model.DECA.decalib.datasets.detectors import batch_FAN
            from model.DECA.decalib.utils.config import cfg as deca_cfg
            from model.mobile_face_net import load_face_landmark_detector
            deca_cfg.rasterizer_type = 'standard'
            
            self.deca = DECA(config=deca_cfg, device=device,rast=False).eval()
            
            self.landmark_detector_deca = load_face_landmark_detector()
            self.landmark_detector_deca = self.landmark_detector_deca.to(rank)
            self.landmark_detector_deca.eval()
        requires_grad(self.parsingpredictor, False)
 
        if self.args.identity_color_jitter:
            self.color_transform = transforms.Compose([
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.01),
        ])
    def g_nonsaturating_loss(self, fake_pred):
        return F.softplus(-fake_pred).mean()

    def d_nonsaturating_loss(self, fake_pred, real_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()

    def gen_update(self, img_target, eye_cond =None, render_backup_meta = None):
        self.gen.train()
        self.gen.zero_grad()

        requires_grad(self.gen, True)
        requires_grad(self.dis, False)
        # if self.args.gaze_loss:
        #     with torch.no_grad():
        #         gaze_target, success, gaze_log_target = gaze_est_tensor(img_target, self.landmark_estimator, self.gaze_estimator)
        # else:
        #     gaze_target=None; success=None
       
        mask_target_19 = self.parsingpredictor(2*torch.clamp(F.interpolate(img_target,512), -1, 1))[0] # (b, 19, 512, 512)
        if self.args.deactivate_bisenet:# always true
            pass
        else:
            mask_target = get_parse(mask_target_19, self.args.seg_type, img_target.device)

        parameter=None
        
        # id_coeffs = coeffs[:, :80] #! 80
        # exp_coeffs = coeffs[:, 80: 144] #! 64
        # tex_coeffs = coeffs[:, 144: 224] 
        # angles = coeffs[:, 224: 227] #! 3
        # gammas = coeffs[:, 227: 254] #! 27
        # translations = coeffs[:, 254:] #! 3
        #!  144 + 3 + 27 + 3 = 144+33=177
        if self.args.geo_type=='mesh' or self.args.geo_type=='parameter' or self.args.geo_type=='both':

            if self.args.render_realtime:#여기걸림
                # assert render_backup_meta is not None
                # assert physic_cond is None; assert aux_mask is None
                render_coeffs = render_backup_meta['coeffs'] # b, 257
                #여기에서 z값 바꾸는 코드 넣으면됨 
                rc_copy = render_coeffs.clone()

                #! RANDOM SCALING FOR RASTERIZED MESH
                if self.args.random_scale:
                    scale_delta = torch.rand(1)*5-4 # -4 ~ 2
                    rc_copy[:,256] += scale_delta.item()
                # if self.args.geo_type == 'mesh':
                if True:
                    if self.args.rasterize_whitened:
                        # print(rc_copy.device,'rcdevice')
                        # print(self.d3d.device,'d3ddevice')
                        rendered_mesh, scaled_mask = self.d3d.decode(rc_copy, no_albedo=True)
                    else:
                        rendered_mesh, scaled_mask = self.d3d.decode(rc_copy) #! b,3,224,224
                else:
                    pass
                #! TARGET TEXTURE CONDITION
                
                if not self.args.use_texture_cond: #If use this option, texture is defined from target image. If not, just brutally from rasterize mesh 
                    img_target_tex = None
                    
                else:
                    # img_target_tex = img_target
                    mask_bisenet = get_parse(mask_target_19, self.args.seg_type, img_target.device)
                    img_target_tex = img_target * mask_bisenet #! segmenting face part is more logical 0917
                    rendered_mesh = transforms.functional.rgb_to_grayscale(rendered_mesh, num_output_channels=3)
                    #rendered_mesh = (rendered_mesh-rendered_mesh.min())/(rendered_mesh.max()-rendered_mesh.min())
                    B = rendered_mesh.shape[0]
                    rendered_mesh_max = rendered_mesh.contiguous().reshape(B,-1).max(axis=1)[0].reshape(B,1,1,1)
                    rendered_mesh_min = rendered_mesh.contiguous().reshape(B,-1).min(axis=1)[0].reshape(B,1,1,1)
                    rendered_mesh = (rendered_mesh-rendered_mesh_min)/(rendered_mesh_max-rendered_mesh_min) #! minor modificataion11/8

                
                _, original_mask = self.d3d.decode(render_coeffs)
                _, original_mask_backup = backup_tensor(rendered_mesh, original_mask, \
                            render_backup_meta['interm_wh'], render_backup_meta['coords'])
                
                aux_mask = original_mask_backup

                physic_cond, _ = backup_tensor(rendered_mesh*(scaled_mask), scaled_mask, \
                            render_backup_meta['interm_wh'], render_backup_meta['coords'])
                
                #! MASK CONFUSION

                if self.args.perforation_confusion:
                    rolled_shape = torch.roll(render_coeffs[...,:80].clone(), 1, 0) #! Batch direction random roll
                    render_coeffs_rolled = torch.cat([rolled_shape,render_coeffs[...,80:]], -1)
                    _, rolled_mask = self.d3d.decode(render_coeffs_rolled) #! b,1,224,224
                    _, rolled_mask_backup = backup_tensor(rendered_mesh, rolled_mask, \
                                render_backup_meta['interm_wh'], render_backup_meta['coords'])
                
                    aux_mask += rolled_mask_backup
                    
                    

                KERNEL = torch.ones(7,7).to(physic_cond.device)
                aux_mask = morphology.dilation(aux_mask, KERNEL)

                # OCCLUSION ADDITION AND MASK CONVEXIZATION
                
                mask_occlusion = get_parse(mask_target_19, 'upper', img_target.device)
                intersection_occlusion = mask_occlusion * aux_mask[:,0:1,:,:]
                aux_mask[:,0:1,:,:] -= intersection_occlusion
                
                if self.args.deactivate_bisenet:# always true
                    KERNEL_aguzi = torch.ones(9,9).to(physic_cond.device)
                    mask_aguzi = get_parse(mask_target_19, 'aguzi', img_target.device)
                    mask_aguzi = morphology.dilation(mask_aguzi, KERNEL_aguzi)
                    aux_mask[:,0:1,:,:] += mask_aguzi
                    mask_target = aux_mask[:,0:1,:,:].bool().float()
                
                else:
                    mask_target += aux_mask[:,0:1,:,:]
                    mask_target = mask_target.bool().float()

            physic_cond = physic_cond.clip(0,1) * 2 -1
            img_target_masked = img_target * (1-mask_target) 

        
            if self.args.identity_color_jitter:
                face_source = self.criterion_id.extract_feats(self.color_transform((img_target+1)/2)*2-1).detach()
            else: 
                face_source = self.criterion_id.extract_feats(img_target).detach()
       
        if self.args.geo_type=='both':
            parameter = torch.cat([rc_copy[...,:144],rc_copy[...,224:]],-1)
            img_swap = self.gen(img_target_masked, timesteps = None, physic_cond = physic_cond, detail_cond = face_source, texture_cond = img_target_tex, gaze_condition=eye_cond, parameter=parameter)

        elif self.args.geo_type=='mesh':
            img_swap = self.gen(img_target_masked, timesteps = None, physic_cond = physic_cond, detail_cond = face_source, texture_cond = img_target_tex, gaze_condition=eye_cond, parameter=parameter)
        else:
            assert parameter is None
            parameter = torch.cat([rc_copy[...,:144],rc_copy[...,224:]],-1)
            img_swap = self.gen(img_target_masked, timesteps = None, physic_cond = None, detail_cond = face_source, texture_cond = img_target_tex, gaze_condition=eye_cond, parameter=parameter)
        if eye_cond is not None:
            img_swap_pred = self.dis(img_swap, eye_cond)
        else:
            img_swap_pred = self.dis(img_swap)
        

        vgg_loss = self.criterion_vgg(img_swap, img_target).mean()
        l1_loss = F.l1_loss(img_swap, img_target)
        gan_g_loss = self.g_nonsaturating_loss(img_swap_pred)

        g_loss = vgg_loss + l1_loss + gan_g_loss # 11.12 vgg 1->0.1 / 11.13 0.1->0.05 / 11.14 0.05->1
        loss_dict={}

        loss_dict['vgg_loss'] = vgg_loss
        loss_dict['l1_loss'] = l1_loss
        loss_dict['gan_g_loss'] = gan_g_loss


        if self.args.id_loss:
            face_swap = self.criterion_id.extract_feats(img_swap)
            if self.args.identity_color_jitter:
                face_source = self.criterion_id.extract_feats(img_target).detach()
            id_loss = (1 - F.cosine_similarity(face_source, face_swap)).mean()
            g_loss += id_loss # 11.12 1->10 / 11.13 10->5 / 11.16 5->1
            loss_dict['id_loss'] = id_loss

        if self.args.parameter_loss:
            interm_wh, coords = render_backup_meta['interm_wh'], render_backup_meta['coords']
            img_swap_cropped_denorm = crop_as_bfm((img_swap+1)/2, interm_wh, coords)

            coeffs_pred = self.d3d.encode(img_swap_cropped_denorm)
            # id_coeffs = coeffs[:, :80]
            # exp_coeffs = coeffs[:, 80: 144]
            # tex_coeffs = coeffs[:, 144: 224]
            # angles = coeffs[:, 224: 227]
            # gammas = coeffs[:, 227: 254]
            # translations = coeffs[:, 254:]
            coeffs_pred_partial = torch.cat([coeffs_pred[:,80:144],coeffs_pred[:,224:227]],-1)
            render_coeffs_partial = torch.cat([render_coeffs[:,80:144],render_coeffs[:,224:227]],-1)
            param_loss = F.l1_loss(coeffs_pred_partial,render_coeffs_partial.detach())
            g_loss += 10 * param_loss #11.20 100->10
            loss_dict['param_loss'] = param_loss
            
        if self.args.parameter_loss_deca:
            assert self.args.parameter_loss == False
            target_cropped_deca = self.landmark_detector_deca.align_face(
                                inputs=denorm(img_target), scale=1.25, inverse=False, target_size=224)
            codedict_target= self.deca.encode(target_cropped_deca,use_detail=False)
            swap_cropped_deca = self.landmark_detector_deca.align_face(
                                inputs=denorm(img_swap), scale=1.25, inverse=False, target_size=224)
            codedict_swap = self.deca.encode(swap_cropped_deca,use_detail=False)

            param_loss = F.l1_loss(codedict_target['pose'],codedict_swap['pose'])
            param_loss += F.l1_loss(codedict_target['exp'],codedict_swap['exp'])
            param_loss += F.l1_loss(codedict_target['shape'],codedict_swap['shape'])
            g_loss += 10 * param_loss #11.20 100->10
            loss_dict['param_deca_loss'] = param_loss
        g_loss.backward()
        self.g_optim.step()

        if self.args.turn_on_ema:
            accumulate(self.gen_ema, self.gen, ACCUM)
        if self.args.parameter_loss_deca:
            return loss_dict, img_swap, swap_cropped_deca 
        if self.args.parameter_loss:
            return loss_dict, img_swap, img_swap_cropped_denorm
        else: 
            return loss_dict, img_swap

    def dis_update(self, img_real, img_recon, eye_cond, eye_cond_false):
        self.dis.zero_grad()

        requires_grad(self.gen, False)
        requires_grad(self.dis, True)
        if eye_cond is not None  and eye_cond_false is not None:
            real_true = self.dis(img_real, eye_cond)
            real_false = self.dis(img_real, eye_cond_false)
            recon_true = self.dis(img_recon.detach(), eye_cond)
            recon_false = self.dis(img_recon.detach(), eye_cond_false)

            d_loss = F.softplus(-real_true).mean() + F.softplus(real_false).mean() + F.softplus(recon_true).mean() + F.softplus(recon_false).mean()

        else:
            real_img_pred = self.dis(img_real)
            recon_img_pred = self.dis(img_recon.detach())

            d_loss = self.d_nonsaturating_loss(recon_img_pred, real_img_pred)
        
        d_loss.backward()
        self.d_optim.step()

        return d_loss

    def sample(self, img_source, img_target, eye_cond=None, render_backup_meta=None):
        # id_coeffs = coeffs[:, :80] #! 80
        # exp_coeffs = coeffs[:, 80: 144] #! 64
        # tex_coeffs = coeffs[:, 144: 224] 
        # angles = coeffs[:, 224: 227] #! 3
        # gammas = coeffs[:, 227: 254] #! 27
        # translations = coeffs[:, 254:] #! 3
        #!  144 + 3 + 27 + 3 = 144+33=177
        with torch.no_grad():
            if self.args.turn_on_ema:
                self.gen_ema.eval()
            else:
                self.gen.eval()
            mask_target_19 = self.parsingpredictor(2*torch.clamp(F.interpolate(img_target,512), -1, 1))[0]
            # if self.args.gaze_loss:
            #     gaze_condition, success, _ = gaze_est_tensor(img_target, self.landmark_estimator, self.gaze_estimator) # success false 인 곳은 그냥 가운데 보도록(0,0) 후처리???
            # else:
            #     gaze_condition = None; success=None
            if self.args.geo_type=='parameter':
                pass
                render_coeffs_source = render_backup_meta['coeffs_source']
                render_coeffs_target = render_backup_meta['coeffs_target']
                mask_target = get_parse(mask_target_19, self.args.seg_type, img_target.device)
                img_target_masked = img_target * (1-mask_target)
                img_target_tex = img_target * (mask_target)
                physic_cond = None
                face_source = self.criterion_id.extract_feats(img_target).detach()
                parameter = torch.cat([render_coeffs_source[...,:80],render_coeffs_target[...,80:144],render_coeffs_target[...,224:]],-1)

                # if self.args.identity_color_jitter:
                #         face_source = self.criterion_id.extract_feats(self.color_transform(img_source)).detach()
                # else: 
                face_source = self.criterion_id.extract_feats(img_source).detach()
               
                img_swap = self.gen(img_target_masked, timesteps = None, physic_cond = physic_cond, detail_cond = face_source, texture_cond = img_target_tex, gaze_condition=gaze_condition, parameter=parameter)
                return img_swap, img_target_masked

            if self.args.geo_type=='mesh' or self.args.geo_type=='both':
                if render_backup_meta is not None:
                    render_coeffs_source = render_backup_meta['coeffs_source']
                    render_coeffs_target = render_backup_meta['coeffs_target']


                    #if self.args.perforation_confusion:
                    render_coeffs_swapped = render_coeffs_target.clone()
                    render_coeffs_swapped[...,:80] = render_coeffs_source[...,:80] # shape
                    render_coeffs_swapped[...,144:224] = render_coeffs_source[...,144:224] # albedo

                
                    if self.args.rasterize_whitened:
                        rendered_mesh, original_mask = self.d3d.decode(render_coeffs_swapped,no_albedo=True) 
                    else:
                        rendered_mesh, original_mask = self.d3d.decode(render_coeffs_swapped) 
                        
                    rendered_mesh_target, mask_target = self.d3d.decode(render_coeffs_target) #! b,3,224,224
                    
                    if not self.args.use_texture_cond: #If use this option, texture is defined from target image. If not, just brutally from rasterize mesh 
                        img_target_tex = None
                        
                    else:
                        # img_target_tex = img_target
                        mask_bisenet = get_parse(mask_target_19, self.args.seg_type, img_target.device)
                        img_target_tex = img_target * mask_bisenet #! segmenting face part is more logical 0917
                        rendered_mesh = transforms.functional.rgb_to_grayscale(rendered_mesh, num_output_channels=3)
                        #rendered_mesh = (rendered_mesh-rendered_mesh.min())/(rendered_mesh.max()-rendered_mesh.min())
                        B = rendered_mesh.shape[0]
                        rendered_mesh_max = rendered_mesh.contiguous().reshape(B,-1).max(axis=1)[0].reshape(B,1,1,1)
                        rendered_mesh_min = rendered_mesh.contiguous().reshape(B,-1).min(axis=1)[0].reshape(B,1,1,1)
                        rendered_mesh = (rendered_mesh-rendered_mesh_min)/(rendered_mesh_max-rendered_mesh_min) #! minor modificataion11/8
                    

                    #if self.args.perforation_confusion:
                    physic_cond, mask_from_source = backup_tensor(rendered_mesh*original_mask, original_mask, \
                                render_backup_meta['interm_wh'], render_backup_meta['coords'])

                    _, mask_from_target = backup_tensor(rendered_mesh_target*mask_target, mask_target, \
                                render_backup_meta['interm_wh'], render_backup_meta['coords'])
                
            

                    if self.args.perforation_confusion:
                        full_mask = (mask_from_source + mask_from_target).bool().float()
                    else:
                        full_mask = mask_from_target.bool().float()
                    
                    KERNEL = torch.ones(11,11).to(physic_cond.device)
                    full_mask = morphology.dilation(full_mask, KERNEL)

                    mask_occlusion = get_parse(mask_target_19, 'upper', img_target.device)
                    intersection_occlusion = mask_occlusion * full_mask[:,0:1,:,:]
                    full_mask[:,0:1,:,:] -= intersection_occlusion#hair-> out/ bg-> mayb zero set(no operation)
                    
                    if self.args.deactivate_bisenet:

                        KERNEL_aguzi = torch.ones(9,9).to(physic_cond.device)
                        mask_aguzi = get_parse(mask_target_19, 'aguzi', img_target.device)
                        mask_aguzi = morphology.dilation(mask_aguzi, KERNEL_aguzi)
                        full_mask[:,0:1,:,:] += mask_aguzi
                        full_mask = full_mask[:,0:1,:,:].bool().float()


                    else:
                        raise NotImplementedError
                    
                    physic_cond = physic_cond.clip(0,1) * 2 - 1

                    KERNEL = torch.ones(3,3).to(physic_cond.device)
                    full_mask = morphology.dilation(full_mask, KERNEL)

                    img_target_masked = img_target * (1-full_mask) 
                    
                    #from torchvision.utils import save_image; save_image((img_target_masked+1)/2.,'test2.png')
                    #import pdb;pdb.set_trace()
                
                    
                    # if self.args.identity_color_jitter:
                    #     face_source = self.criterion_id.extract_feats(self.color_transform(img_source)).detach()
                    # else: 
                    face_source = self.criterion_id.extract_feats(img_source).detach()
                    parameter = None
                    if self.args.turn_on_ema:
                        img_swap = self.gen_ema(img_target_masked, timesteps = None, physic_cond = physic_cond, detail_cond = face_source, texture_cond = img_target_tex, gaze_condition=eye_cond, old=self.args.old) 
                    else:
                        if self.args.geo_type=='both':
                            parameter = torch.cat([render_coeffs_source[...,:80],render_coeffs_target[...,80:144],render_coeffs_target[...,224:]],-1)
                            img_swap = self.gen(img_target_masked, timesteps = None, physic_cond = physic_cond, detail_cond = face_source, texture_cond = img_target_tex, gaze_condition=eye_cond, parameter=parameter, old=self.args.old)

                        elif self.args.geo_type=='mesh':
                            img_swap = self.gen(img_target_masked, timesteps = None, physic_cond = physic_cond, detail_cond = face_source, texture_cond = img_target_tex, gaze_condition=eye_cond, parameter=parameter, old=self.args.old)
                        else:
                            parameter = torch.cat([render_coeffs_swapped[...,:144],render_coeffs_swapped[...,224:]],-1)
                            img_swap = self.gen(img_target_masked, timesteps = None, physic_cond = None, detail_cond = face_source, texture_cond = img_target_tex, gaze_condition=eye_cond, parameter=parameter, old=self.args.old)
                    return img_swap, img_target_masked, physic_cond


    # 일단은 똑같음
    def sample_vid(self, img_source, img_target, eye_cond=None, render_backup_meta=None):
        # id_coeffs = coeffs[:, :80] #! 80
        # exp_coeffs = coeffs[:, 80: 144] #! 64
        # tex_coeffs = coeffs[:, 144: 224] 
        # angles = coeffs[:, 224: 227] #! 3
        # gammas = coeffs[:, 227: 254] #! 27
        # translations = coeffs[:, 254:] #! 3
        #!  144 + 3 + 27 + 3 = 144+33=177
        with torch.no_grad():
            if self.args.turn_on_ema:
                self.gen_ema.eval()
            else:
                self.gen.eval()
            mask_target_19 = self.parsingpredictor(2*torch.clamp(F.interpolate(img_target,512), -1, 1))[0]
            # if self.args.gaze_loss:
            #     gaze_condition, success, _ = gaze_est_tensor(img_target, self.landmark_estimator, self.gaze_estimator) # success false 인 곳은 그냥 가운데 보도록(0,0) 후처리???
            # else:
            #     gaze_condition = None; success=None
            if self.args.geo_type=='parameter':
                pass
                render_coeffs_source = render_backup_meta['coeffs_source']
                render_coeffs_target = render_backup_meta['coeffs_target']
                mask_target = get_parse(mask_target_19, self.args.seg_type, img_target.device)
                img_target_masked = img_target * (1-mask_target)
                img_target_tex = img_target * (mask_target)
                physic_cond = None
                face_source = self.criterion_id.extract_feats(img_target).detach()
                parameter = torch.cat([render_coeffs_source[...,:80],render_coeffs_target[...,80:144],render_coeffs_target[...,224:]],-1)

                # if self.args.identity_color_jitter:
                #         face_source = self.criterion_id.extract_feats(self.color_transform(img_source)).detach()
                # else: 
                face_source = self.criterion_id.extract_feats(img_source).detach()
               
                img_swap = self.gen(img_target_masked, timesteps = None, physic_cond = physic_cond, detail_cond = face_source, texture_cond = img_target_tex, gaze_condition=gaze_condition, parameter=parameter)
                return img_swap, img_target_masked

            if self.args.geo_type=='mesh' or self.args.geo_type=='both':
                if render_backup_meta is not None:
                    render_coeffs_source = render_backup_meta['coeffs_source']
                    render_coeffs_target = render_backup_meta['coeffs_target']


                    #if self.args.perforation_confusion:
                    render_coeffs_swapped = render_coeffs_target.clone()
                    render_coeffs_swapped[...,:80] = render_coeffs_source[...,:80] # shape
                    render_coeffs_swapped[...,144:224] = render_coeffs_source[...,144:224] # albedo

                
                    if self.args.rasterize_whitened:
                        rendered_mesh, original_mask = self.d3d.decode(render_coeffs_swapped,no_albedo=True) 
                    else:
                        rendered_mesh, original_mask = self.d3d.decode(render_coeffs_swapped) 
                        
                    rendered_mesh_target, mask_target = self.d3d.decode(render_coeffs_target) #! b,3,224,224
                    
                    if not self.args.use_texture_cond: #If use this option, texture is defined from target image. If not, just brutally from rasterize mesh 
                        img_target_tex = None
                        
                    else:
                        # img_target_tex = img_target
                        mask_bisenet = get_parse(mask_target_19, self.args.seg_type, img_target.device)
                        img_target_tex = img_target * mask_bisenet #! segmenting face part is more logical 0917
                        rendered_mesh = transforms.functional.rgb_to_grayscale(rendered_mesh, num_output_channels=3)
                        #rendered_mesh = (rendered_mesh-rendered_mesh.min())/(rendered_mesh.max()-rendered_mesh.min())
                        B = rendered_mesh.shape[0]
                        rendered_mesh_max = rendered_mesh.contiguous().reshape(B,-1).max(axis=1)[0].reshape(B,1,1,1)
                        rendered_mesh_min = rendered_mesh.contiguous().reshape(B,-1).min(axis=1)[0].reshape(B,1,1,1)
                        rendered_mesh = (rendered_mesh-rendered_mesh_min)/(rendered_mesh_max-rendered_mesh_min) #! minor modificataion11/8
                    

                    #if self.args.perforation_confusion:
                    physic_cond, mask_from_source = backup_tensor(rendered_mesh*original_mask, original_mask, \
                                render_backup_meta['interm_wh'], render_backup_meta['coords'])

                    _, mask_from_target = backup_tensor(rendered_mesh_target*mask_target, mask_target, \
                                render_backup_meta['interm_wh'], render_backup_meta['coords'])
                
                    
            

                    if self.args.perforation_confusion:
                        full_mask = (mask_from_source + mask_from_target).bool().float()
                    else:
                        full_mask = mask_from_target.bool().float()
                    
                    KERNEL = torch.ones(11,11).to(physic_cond.device)
                    full_mask = morphology.dilation(full_mask, KERNEL)

                    mask_occlusion = get_parse(mask_target_19, 'upper', img_target.device)
                    intersection_occlusion = mask_occlusion * full_mask[:,0:1,:,:]
                    full_mask[:,0:1,:,:] -= intersection_occlusion#hair-> out/ bg-> mayb zero set(no operation)
                    
                    if self.args.deactivate_bisenet:

                        KERNEL_aguzi = torch.ones(9,9).to(physic_cond.device)
                        mask_aguzi = get_parse(mask_target_19, 'aguzi', img_target.device)
                        mask_aguzi = morphology.dilation(mask_aguzi, KERNEL_aguzi)
                        full_mask[:,0:1,:,:] += mask_aguzi
                        full_mask = full_mask[:,0:1,:,:].bool().float()


                    else:
                        raise NotImplementedError
                    
                    physic_cond = physic_cond.clip(0,1) * 2 - 1

                    KERNEL = torch.ones(3,3).to(physic_cond.device)
                    full_mask = morphology.dilation(full_mask, KERNEL)

                    img_target_masked = img_target * (1-full_mask) 
                    
                    #from torchvision.utils import save_image; save_image((img_target_masked+1)/2.,'test2.png')
                    #import pdb;pdb.set_trace()
                
                    
                    # if self.args.identity_color_jitter:
                    #     face_source = self.criterion_id.extract_feats(self.color_transform(img_source)).detach()
                    # else: 
                    face_source = self.criterion_id.extract_feats(img_source).detach()
                    parameter = None
                    if self.args.turn_on_ema:
                        img_swap = self.gen_ema(img_target_masked, timesteps = None, physic_cond = physic_cond, detail_cond = face_source, texture_cond = img_target_tex, gaze_condition=eye_cond, old=self.args.old) 
                    else:
                        if self.args.geo_type=='both':
                            parameter = torch.cat([render_coeffs_source[...,:80],render_coeffs_target[...,80:144],render_coeffs_target[...,224:]],-1)
                            img_swap = self.gen(img_target_masked, timesteps = None, physic_cond = physic_cond, detail_cond = face_source, texture_cond = img_target_tex, gaze_condition=eye_cond, parameter=parameter, old=self.args.old)

                        elif self.args.geo_type=='mesh':
                            img_swap = self.gen(img_target_masked, timesteps = None, physic_cond = physic_cond, detail_cond = face_source, texture_cond = img_target_tex, gaze_condition=eye_cond, parameter=parameter, old=self.args.old)
                        else:
                            parameter = torch.cat([render_coeffs_swapped[...,:144],render_coeffs_swapped[...,224:]],-1)
                            img_swap = self.gen(img_target_masked, timesteps = None, physic_cond = None, detail_cond = face_source, texture_cond = img_target_tex, gaze_condition=eye_cond, parameter=parameter, old=self.args.old)
                    return img_swap, img_target_masked, physic_cond


    def resume(self, resume_ckpt):
        print("load model:", resume_ckpt)
        ckpt = torch.load(resume_ckpt)
        ckpt_name = os.path.basename(resume_ckpt)
        start_iter = int(os.path.splitext(ckpt_name)[0])
        try:
            self.gen_ema.module.load_state_dict(ckpt["gen"])
        except:
            print('There is no ema weights!')
        self.gen.module.load_state_dict(ckpt["gen"])
        self.dis.module.load_state_dict(ckpt["dis"])
        self.g_optim.load_state_dict(ckpt["g_optim"])
        self.d_optim.load_state_dict(ckpt["d_optim"])

        return start_iter

    def save(self, idx, checkpoint_path):
        save_dict = {
                "gen": self.gen.module.state_dict(),
                "dis": self.dis.module.state_dict(),
                "g_optim": self.g_optim.state_dict(),
                "d_optim": self.d_optim.state_dict(),
                "args": self.args
            }
        if self.args.turn_on_ema:
            save_dict['gen_ema'] = self.gen_ema.module.state_dict()
        torch.save(save_dict,
            f"{checkpoint_path}/{str(idx).zfill(6)}.pt"
        )
