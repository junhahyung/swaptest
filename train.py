import argparse
import os
import torch
from torch.utils import data
from dataset import FFHQ, CelebAHQ
import torchvision
import torchvision.transforms as transforms
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
os.environ["OMP_NUM_THREADS"] = '1'

def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def display_img(idx, img, name, writer):
    img = img.clamp(-1, 1)
    img = ((img - img.min()) / (img.max() - img.min())).data

    writer.add_images(tag='%s' % (name), global_step=idx, img_tensor=img)


def write_loss(i, loss_dict, writer):
    # writer.add_scalar('vgg_loss', vgg_loss.item(), i)
    # writer.add_scalar('l1_loss', l1_loss.item(), i)
    # writer.add_scalar('gen_loss', g_loss.item(), i)
    # writer.add_scalar('dis_loss', d_loss.item(), i)

    for k, v in loss_dict.items():
        writer.add_scalar(k, v.item(), i)
    writer.flush()


def ddp_setup(args, rank, world_size):
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = args.port

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main(rank, world_size, args):
    # init distributed computing
    ddp_setup(args, rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda")

    # make logging folder
    log_path = os.path.join(args.exp_path, args.exp_name + '/log')
    checkpoint_path = os.path.join(args.exp_path, args.exp_name + '/checkpoint')

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    writer = SummaryWriter(log_path)

    print('==> preparing dataset')
    transform = torchvision.transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    )


    if args.dataset == 'ffhq': #! this is for test
        dataset = FFHQ('train', transform = transform, augmentation=True, 
                 render_realtime = args.render_realtime, eye_gaze_cond=args.gaze_img_cond)
        dataset_test = CelebAHQ('test_all', transform = transform, eye_gaze_cond=args.gaze_img_cond)

    else:
        raise NotImplementedError

    loader = data.DataLoader(
        dataset,
        num_workers=8,
        batch_size=args.batch_size // world_size,
        sampler=data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True),
        pin_memory=True,
        drop_last=False,
    )

    loader_test = data.DataLoader(
        dataset_test,
        num_workers=8,
        batch_size=16,
        sampler=data.distributed.DistributedSampler(dataset_test, num_replicas=world_size, rank=rank, shuffle=False),
        pin_memory=True,
        drop_last=False,
    )

    loader = sample_data(loader)
    loader_test = sample_data(loader_test)

    print('==> initializing trainer')
    # Trainer
    trainer = Trainer(args, device, rank)

    # resume
    if args.resume_ckpt is not None:
        args.start_iter = trainer.resume(args.resume_ckpt)
        print('==> resume from iteration %d' % (args.start_iter))

    print('==> training')
    pbar = range(args.iter)
    for idx in pbar:
        i = idx + args.start_iter

        # laoding data
        physic_cond = None
        aux_mask = None
        render_backup_meta = None

        if args.render_realtime:
            render_backup_meta = dict()
            vis_cond = None
            vis_cond_false = None

            if args.gaze_img_cond:
                _, img_target, coeffs, interm_wh, coords, vis_cond, vis_cond_false = next(loader)
            else:
                _, img_target, coeffs, interm_wh, coords = next(loader)
            #physic_cond = rendered_source.to(rank, non_blocking=True)
            coeffs = coeffs.to(rank, non_blocking=True)
            interm_wh = interm_wh.to(rank, non_blocking=True)
            coords = coords.to(rank, non_blocking=True)
            #face_source = face_source.to(rank, non_blocking=True)
            img_target = img_target.to(rank, non_blocking=True)
            #aux_mask = aux_mask.to(rank, non_blocking=True)
            if args.gaze_img_cond:
                vis_cond = vis_cond.to(rank, non_blocking=True).float()
                vis_cond_false = vis_cond_false.to(rank, non_blocking=True).float()
            render_backup_meta['coeffs'] = coeffs; render_backup_meta['interm_wh'] = interm_wh; render_backup_meta['coords'] = coords
            
        else:
            raise NotImplementedError
          

    
        if args.parameter_loss or args.parameter_loss_deca:
            loss_dict, img_swap, crop_rt = trainer.gen_update(img_target, vis_cond, render_backup_meta)

        else:
            loss_dict, img_swap = trainer.gen_update(img_target, vis_cond, render_backup_meta)

        '''
        gaze_log_target, gaze_log_swap = gaze_log
        if gaze_log_target is not None:
            gaze_log_target_le, gaze_log_target_re = gaze_log_target

        if gaze_log_swap is not None:
            gaze_log_swap_le, gaze_log_swap_re = gaze_log_swap
        '''

        # update discriminator
        gan_d_loss = trainer.dis_update(img_target, img_swap, vis_cond, vis_cond_false)
        loss_dict['gan_d_loss'] = gan_d_loss
        if rank == 0:
            # write to log
            write_loss(idx, loss_dict, writer)

        # display
        if i % args.display_freq == 0 and rank == 0:
            # print("[Iter %d/%d] [vgg loss: %f] [l1 loss: %f] [g loss: %f] [d loss: %f]"
            #       % (i, args.iter, vgg_loss.item(), l1_loss.item(), gan_g_loss.item(), gan_d_loss.item()))
            loss_prompt = f'[Iteration {int(i)}/{int(args.iter)}] '
            for k,v in loss_dict.items():
                loss_prompt += f'[{k}: {float(v)}] '

            print(loss_prompt)
            if rank == 0:
                if args.dataset == 'ffhq_debug':
                    render_backup_meta = dict()
                    _, img_target, coeffs, interm_wh, coords = next(loader)
                    #physic_cond = rendered_source.to(rank, non_blocking=True)
                    coeffs = coeffs.to(rank, non_blocking=True)
                    interm_wh = interm_wh.to(rank, non_blocking=True)
                    coords = coords.to(rank, non_blocking=True)
                    #face_source = face_source.to(rank, non_blocking=True)
                    img_target = img_target.to(rank, non_blocking=True)
                    #aux_mask = aux_mask.to(rank, non_blocking=True)
                    render_backup_meta['coeffs'] = coeffs; render_backup_meta['interm_wh'] = interm_wh; render_backup_meta['coords'] = coords

                    img_swap, img_target_masked, img_target_masked_bfm, bfm_rendered, aux_mask, aux_mask_org, mask_hat_and_hair = trainer.sample(img_target, img_target, None, None, render_backup_meta)
                   


                    display_img(i, mask_hat_and_hair, 'Hat_hair_mask', writer)
                    display_img(i, img_target, 'GT', writer)
                    display_img(i, aux_mask, 'aux_mask_roll', writer)
                    display_img(i, aux_mask_org, 'aux_mask_org', writer)
                    display_img(i, bfm_rendered, 'bfm_rendered', writer)
                    display_img(i, img_swap, 'reconstructed', writer)
                    display_img(i, img_target_masked, 'target_masked_bisenet_bfm_no_hair_hat', writer)
                    display_img(i, img_target_masked_bfm, 'bgimg_from_bfm_with_shape_rolled_mask', writer)
                    #display_img(i, mix_rendered, 'mix_rendered', writer)
                    #display_img(i, img_swap, 'swap', writer)
                elif args.render_realtime:

                    '''
                    if gaze_log_target is not None:
                        display_img(i, gaze_log_target_le, 'gaze_left_target', writer)
                        display_img(i, gaze_log_target_re, 'gaze_right_target', writer)
                    if gaze_log_swap is not None:
                        display_img(i, gaze_log_swap_le, 'gaze_left_swap', writer)
                        display_img(i, gaze_log_swap_re, 'gaze_right_swap', writer)
                    '''

                    render_backup_meta = dict()
                    if args.gaze_img_cond:
                        _, img_source, img_target, coeffs_source, coeffs_target, interm_wh, coords, vis_cond = next(loader_test)
                    else:
                        _, img_source, img_target, coeffs_source, coeffs_target, interm_wh, coords = next(loader_test)
                    #physic_cond = rendered_source.to(rank, non_blocking=True)
                    coeffs_source = coeffs_source.to(rank, non_blocking=True)
                    coeffs_target = coeffs_target.to(rank, non_blocking=True)
                    interm_wh = interm_wh.to(rank, non_blocking=True)
                    coords = coords.to(rank, non_blocking=True)
                    #face_source = face_source.to(rank, non_blocking=True)
                    img_source = img_source.to(rank, non_blocking=True)
                    img_target = img_target.to(rank, non_blocking=True)
                    #aux_mask = aux_mask.to(rank, non_blocking=True)
                    if args.gaze_img_cond:
                        vis_cond = vis_cond.to(rank, non_blocking=True).float()
                    else:
                        vis_cond = None
                    render_backup_meta['coeffs_source'] = coeffs_source; render_backup_meta['coeffs_target'] = coeffs_target; render_backup_meta['interm_wh'] = interm_wh; render_backup_meta['coords'] = coords
                    if args.geo_type == 'parameter':
                        img_swap, img_target_masked = trainer.sample(img_source, img_target,vis_cond,  render_backup_meta)
                    
                    elif args.geo_type == 'both':
                        img_swap, img_target_masked, physic_cond = trainer.sample(img_source, img_target,vis_cond,  render_backup_meta)

                    else:
                        img_swap, img_target_masked, physic_cond = trainer.sample(img_source, img_target,vis_cond,  render_backup_meta)

                    display_img(i, img_source, 'source', writer)
                    display_img(i, img_target, 'target', writer)
                    display_img(i, img_swap, 'result', writer)
                    
                    display_img(i, img_target_masked, 'holed_target', writer)
                    if args.geo_type == 'parameter':
                        pass
                    else:
                        display_img(i, img_target_masked+physic_cond, 'overlay', writer)
                        display_img(i, physic_cond, 'eroded_bfm', writer)
                        if args.gaze_img_cond:
                            display_img(i, vis_cond, 'eye cond', writer)
                    if args.parameter_loss or args.parameter_loss_deca:
                        display_img(i, crop_rt*2-1, 'deca_or_bfm_crop', writer)
                   
                writer.flush()

        # save model
        if i % args.save_freq == 0 and rank == 0:
            trainer.save(i, checkpoint_path)

    return


if __name__ == "__main__":
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=800000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--display_freq", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=50000)
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--dataset", type=str, default='vggface')
    parser.add_argument("--exp_path", type=str, default='./exps_0916')
    parser.add_argument("--exp_name", type=str, default='v1')
    parser.add_argument("--addr", type=str, default='localhost')
    parser.add_argument("--port", type=str, default='12346')
    parser.add_argument("--seg_type", type=str, default='partial',choices=['partial','whole'])
    # parser.add_argument("--physic_cond", action='store_true')
    
    # parser.add_argument("--bfm_mask", action='store_true')
    parser.add_argument("--in_channels", type=int, default=3) #! for abl. 3
    parser.add_argument("--parser_type", type=str, default='bisenet',choices=['bisenet','segnext'])
    parser.add_argument("--render_realtime", action='store_true')
    parser.add_argument("--id_loss", action='store_true')
    parser.add_argument("--parameter_loss", action='store_true')
    parser.add_argument("--parameter_loss_deca", action='store_true')
    parser.add_argument("--deactivate_bisenet", action='store_true')

    parser.add_argument("--perforation_confusion", action='store_true') #1
    parser.add_argument("--random_scale", action='store_true')#2
    parser.add_argument("--use_texture_cond", action='store_true')#3
    parser.add_argument("--identity_color_jitter", action='store_true')#4

    parser.add_argument("--rasterize_whitened", action='store_true') #번외
    parser.add_argument("--turn_on_ema", action='store_true')
    #parser.add_argument("--gaze_loss", action='store_true')

    #parser.add_argument("--gaze_lambda", type=float, default=1.)
    parser.add_argument("--gaze_img_cond", action='store_true')
    parser.add_argument("--old", type=bool, default=False)

    #! from here extra
    parser.add_argument("--geo_type", type=str, default='mesh',choices=['mesh','parameter','keypoints', 'both'])

    opts = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    #assert n_gpus >= 2

    world_size = n_gpus
    print('==> training on %d gpus' % n_gpus)
    mp.spawn(main, args=(world_size, opts,), nprocs=world_size, join=True)
