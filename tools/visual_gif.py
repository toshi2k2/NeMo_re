import sys
sys.path.append("../")
sys.path.append('../code/lib')
sys.path.append('/home/adam/NeMo/code/lib')
sys.path.append('/home/adam/NeMo/exp')


import numpy as np
from MeshUtils import *
import os
from PIL import Image
import BboxTools as bbt
import seaborn as sns
import matplotlib.pyplot as plt
import io


def get_img(theta, elevation, azimuth, distance):
    C = camera_position_from_spherical_angles(distance, elevation, azimuth, degrees=False, device=device)
    R, T = campos_to_R_T(C, theta, device=device)
    image = phong_renderer(meshes_world=meshes.clone(), R=R, T=T)
    image = image[:, ..., :3]
    box_ = bbt.box_by_shape(crop_size, (render_image_size // 2,) * 2)
    bbox = box_.bbox
    image = image[:, bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], :]
    image = torch.squeeze(image).detach().cpu().numpy()
    image = np.array((image / image.max()) * 255).astype(np.uint8)
    return image


def visual_on_plot(plot_img, theta, elevation, azimuth, max_range=(-3.3, 3.3), width=2):
    h, w, _ = plot_img.shape

    plot_img = plot_img.copy()

    # print(azimuth)
    pos_ = int((azimuth / (max_range[1] - max_range[0]) + 0.5) * w)
    # print(pos_)
    box_ = bbt.Bbox2D([(0, h), (pos_ - width // 2, pos_ + width // 2)], image_boundary=plot_img.shape)
    bbt.draw_bbox(plot_img, box_, fill=(0, 0, 255))

    pos_ = int((elevation / (max_range[1] - max_range[0]) + 0.5) * w)
    box_ = bbt.Bbox2D([(0, h), (pos_ - width // 2, pos_ + width // 2)], image_boundary=plot_img.shape)
    bbt.draw_bbox(plot_img, box_, fill=(240, 0, 0))

    pos_ = int((theta / (max_range[1] - max_range[0]) + 0.5) * w)
    box_ = bbt.Bbox2D([(0, h), (pos_ - width // 2, pos_ + width // 2)], image_boundary=plot_img.shape)
    bbt.draw_bbox(plot_img, box_, fill=(0, 200, 0))

    return plot_img


def get_one_image_from_plt(plot_functions, plot_args=tuple(), plot_kwargs=dict()):
    plt.cla()
    plt.clf()
    ax = plot_functions(*plot_args, **plot_kwargs)
    positions = ax.get_position()
    pos = [positions.y0, positions.y1, positions.x0, positions.x1]
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)
    img = np.array(im)
    h, w = img.shape[0:2]
    box = bbt.from_numpy([np.array([int(t[0] * h), int(t[1] * h), int(t[2] * w), int(t[3] * w)]) for t in [pos]][0])
    box = box.pad(1)
    box = box.shift((2, 1))
    img = box.apply(img)
    bbt.draw_bbox(img, bbt.full(img.shape).pad(-2), boundary=(0, 0, 0), boundary_width=11)
    # img = np.transpose(img, (1, 0, 2))
    return img


def alpha_merge_imgs(im1, im2, alpha=0.8):
    mask = np.zeros(im1.shape[0:2], dtype=np.uint8)
    mask[np.sum(im2, axis=2) > 0] = int(255 * alpha)
    im2 = np.concatenate((im2, mask.reshape((*im1.shape[0:2], 1))), axis=2)
    im2 = Image.fromarray(im2)
    im1 = Image.fromarray(im1)
    im1.paste(im2, (0, 0), im2)
    return np.array(im1)


batch_size = 20

image_sizes = {'car': (256, 672), 'bus': (320, 800), 'motorbike': (512, 512), 'boat': (512, 1216),
               'bicycle': (608, 608), 'aeroplane': (320, 1024), 'sofa': (352, 736), 'tvmonitor': (480, 480),
               'chair': (544, 384), 'diningtable': (320, 800), 'bottle': (512, 736), 'train': (256, 608)}
distance_render = {'car': 5, 'bus': 5.5, 'motorbike': 4.5, 'bottle': 5.75, 'boat': 8, 'bicycle': 5.2, 'aeroplane': 7,
                   'sofa': 5.4, 'tvmonitor': 5.5, 'chair': 4, 'diningtable': 7, 'train': 3.75}


cate = 'train'
# cate = 'chair'
mesh_d = 'single'
# occ_level = 'FGL1_BGL1'
occ_level = ''
# occ_mesh_d = 'd5'
occ_mesh_d = ''

render_image_size = max(image_sizes[cate])
crop_size = image_sizes[cate]
down_smaple_rate = 8


if len(occ_level) == 0:
    # record_path = 'resunetpre_3D512_points1saved_model_%s_799_clutter_%s' % (cate, mesh_d)
    record_path = '../data/PASCAL3D_NeMo/images/' + cate
    save_dir = './final_visual_%s_799_clutter_%s' % (cate, mesh_d)
    # anno_path = '../../PASCAL3D/PASCAL3D_distcrop/annotations/%s/%s.npz' % (cate, '%s')
    anno_path = '../data/PASCAL3D_NeMo/annotations/%s/%s.npz' % (cate, '%s')
    # img_path = '../../PASCAL3D/PASCAL3D_distcrop/images/%s/%s.JPEG' % (cate, '%s')
    img_path = '../data/PASCAL3D_NeMo/images/%s/%s.JPEG' % (cate, '%s')
    # plot_path = '../../junks/scan_align_resunet_d4/%s.png'
    plot_path = '../exp/aligns_final/' + cate + '_' + mesh_d + '/%s.png'
else:
    record_path = occ_level + '_resunetpre_3D512_points1saved_model_%s_799_clutter_%s' % (cate, mesh_d)
    save_dir = './final_visual_' + occ_level + '_resunetpre_3D512_points1saved_model_%s_799_clutter_%s' % (cate, mesh_d)
    anno_path = '../../PASCAL3D/PASCAL3D_distcrop/annotations/%s/%s.npz' % (cate, '%s')
    img_path = '../../PASCAL3D/PASCAL3D_OCC_distcrop/images/' + cate + occ_level + '/%s.JPEG'
    # plot_path = '../../junks/scan_align_resunet_d4/%s.png'
    plot_path = '../../junks/aligns_final/' + cate + occ_level + '_' + mesh_d + '/%s.png'

# mesh_path = '../data/PASCAL3D+_release1.1/CAD_%s/' % occ_mesh_d + cate
# mesh_path = '../data/PASCAL3D+_release1.1/CAD_%s/' % mesh_d + cate
mesh_path = '../data/PASCAL3D+_release1.1/CAD/%s/' % cate
if len(occ_level) > 0:
    occ_features_path = '../saved_features/' + cate + '_occ/' + occ_level + '_resunetpre_3D512_points1saved_model_%s_799_%s.npz' % (cate, occ_mesh_d)
else:
    # occ_features_path = '../saved_features/' + cate + '/resunetpre_3D512_points1saved_model_%s_799_%s.npz' % (cate, occ_mesh_d)
    occ_features_path = '../exp/Features_' + mesh_d + '/' + cate + '/' + occ_level + 'saved_feature_%s_%s.npz' % (cate, mesh_d)
occ_features = np.load(occ_features_path)
subtypes = ['mesh%02d' % i for i in range(1, 1 + len(os.listdir(mesh_path)))]

device = 'cuda:0'
map_shape = (image_sizes[cate][0] // down_smaple_rate, image_sizes[cate][1] // down_smaple_rate)


if __name__ == '__main__':
    print(save_dir)
    set_distance = distance_render[cate]
    os.makedirs(save_dir, exist_ok=True)
    cameras = OpenGLPerspectiveCameras(device=device, fov=12.0)
    raster_settings = RasterizationSettings(
        image_size=render_image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )
    raster_settings1 = RasterizationSettings(
        image_size=render_image_size // down_smaple_rate,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings1
    )
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    phong_renderer = MeshRenderer(

        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, lights=lights, cameras=cameras)
    )
    img_list = os.listdir(record_path)
    img_list = [t.split('.')[0] for i, t in enumerate(img_list)]

    idx = 0
    for image_name in img_list:
        idx += 1
        if os.path.exists(os.path.join(save_dir, image_name + '.gif')):
            continue

        if image_name != 'n03896233_8719':
            continue

        # if idx % 3 != 0:
        #     continue

        print(image_name)

        # records = np.load(os.path.join(record_path, image_name + '.npy'))  #where is this?
        records = np.load(os.path.join('/home/adam/NeMo/junk/snow/'+image_name+'.npy'), allow_pickle=True)

        img = np.array(Image.open(img_path % image_name).resize(crop_size[::-1]))
        plot_ = np.array(Image.open(plot_path % image_name).resize((crop_size[1] + 10, crop_size[0] + 10)))[:, :, 0:3]
        plot_ = bbt.full(plot_.shape).pad(-5).apply(plot_)

        annos = np.load(anno_path % image_name)
        this_idx = annos['cad_index']
        theta_gt, elevation_gt, azimuth_gt, distance_gt = float(annos['theta']), float(annos['elevation']), float(annos['azimuth']), float(annos['distance'])

        # print(records, theta_gt, elevation_gt)
        theta_pred, elevation_pred, azimuth_pred, distance_pred = records[-1]
        # theta_pred, distance_pred, elevation_pred, azimuth_pred = records

        if np.max([np.abs(theta_pred - theta_gt), np.abs(elevation_pred - elevation_gt), np.abs(azimuth_pred - azimuth_gt)]) > 0.2:
            continue

        k = annos['cad_index'] - 1
        xvert, xface = load_off(mesh_path + '/%02d.off' % (k + 1), to_torch=True)
        # xvert, xface = load_off('../data/PASCAL3D+_release1.1/CAD/car' + '/%02d.off' % (k + 1), to_torch=True)
        
        # subtype = subtypes[k]
        subtype = 'mesh01'
        # for k,v in occ_features.items():
        #     print(k)
        # print(subtypes)
        # exit()
        name_list = occ_features['names_%s' % subtype]
        feature_bank = torch.from_numpy(occ_features['memory_%s' % subtype])
        clutter_bank = torch.from_numpy(occ_features['clutter_%s' % subtype])
        inter_module = MeshInterpolateModule(xvert, xface, feature_bank, rasterizer, post_process=center_crop_fun(map_shape, (render_image_size // down_smaple_rate, ) * 2))
        inter_module = inter_module.cuda()
        clutter_bank = clutter_bank.cuda()
        clutter_bank = normalize(torch.mean(clutter_bank, dim=0, keepdim=True), dim=1)
        predicted_map = occ_features[image_name]
        predicted_map = torch.from_numpy(predicted_map).to(device)
        clutter_score = torch.nn.functional.conv2d(predicted_map.unsqueeze(0),
                                                   clutter_bank.unsqueeze(2).unsqueeze(3)).squeeze()

        x3d, xface = load_off('../data/PASCAL3D+_release1.1/CAD/' + cate + '/%02d.off' % this_idx)

        verts = torch.from_numpy(x3d).to(device)
        verts = pre_process_mesh_pascal(verts)
        faces = torch.from_numpy(xface).to(device)

        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = Textures(verts_rgb.to(device))
        meshes = Meshes(verts=[verts], faces=[faces], textures=textures)

        get_images = []

        img_gt = get_img(theta_gt, elevation_gt, azimuth_gt, float(distance_render[cate]))

        meshes = meshes.extend(batch_size)

        visual_static = np.concatenate([img_gt, img], axis=0)

        # for record in records:
        with torch.no_grad():
            for j in range(records.shape[0] // batch_size):
                theta_pred, elevation_pred, azimuth_pred, distance_pred = [], [], [], []

                plot_batchs = []
                for k in range(batch_size):
                    theta_pred_, elevation_pred_, azimuth_pred_, distance_pred_ = np.array(records[j * batch_size + k]).astype(float).tolist()
                    this_plot = visual_on_plot(plot_, theta_pred_ - theta_gt, elevation_pred_ - elevation_gt, (np.pi + azimuth_pred_ - azimuth_gt) % (2 * np.pi) - np.pi) # loss landscape
                    theta_pred.append(theta_pred_)
                    elevation_pred.append(elevation_pred_)
                    azimuth_pred.append(azimuth_pred_)
                    distance_pred.append(distance_pred_)

                    plot_batchs.append(this_plot)

                theta_pred = torch.from_numpy(np.array(theta_pred, dtype=np.float32)).to(device)
                elevation_pred = torch.from_numpy(np.array(elevation_pred, dtype=np.float32)).to(device)
                azimuth_pred = torch.from_numpy(np.array(azimuth_pred, dtype=np.float32)).to(device)
                distance_pred = torch.from_numpy(np.array(distance_pred, dtype=np.float32)).to(device)

                img_ = get_img(theta_pred, elevation_pred, azimuth_pred, distance_pred)

                C = camera_position_from_spherical_angles(distance_pred, elevation_pred, azimuth_pred, degrees=False, device=device)

                theta = theta_pred
                projected_map = inter_module(C, theta)

                # print(img_.shape)
                for k in range(batch_size):
                    object_score = torch.sum(projected_map[k] * predicted_map, dim=0)

                    # print(object_score)

                    object_mask = torch.abs(object_score) > 1e-5
                    occluder_mask = clutter_score > object_score

                    out = torch.zeros_like(object_mask).type(torch.float32)
                    out += object_mask.type(torch.float32) * 1
                    out -= object_mask.type(torch.float32) * occluder_mask.type(torch.float32) * 2

                    occ_pred = get_one_image_from_plt(sns.heatmap, plot_args=(out.squeeze().cpu().numpy(),),
                                                      plot_kwargs=dict(cbar=True, vmax=1, vmin=-1, square=True,
                                                                       cmap='RdYlGn'))
                    occ_pred = np.array(Image.fromarray(occ_pred).resize(crop_size[::-1]))

                    # get_image = np.concatenate((np.concatenate([img_[k], plot_batchs[k]], axis=0), visual_static), axis=1)
                    get_image = np.concatenate((np.concatenate([img, occ_pred[:, :, 0:3]], axis=0), np.concatenate([alpha_merge_imgs(img, img_[k]), plot_batchs[k]], axis=0)), axis=1)
                    get_images.append(Image.fromarray(get_image))
        # get_images[-1].show()
        # get_images = get_images[::-1]
        # break
        get_images[0].save(os.path.join(save_dir, image_name + '.gif'), save_all=True, append_images=get_images[1::])


