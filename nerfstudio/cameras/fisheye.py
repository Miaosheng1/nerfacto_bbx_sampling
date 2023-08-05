import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.cameras import camera_utils

class Fisheye():
    def __init__(self,fisheye_meta):
        self.fisheye_imgs = fisheye_meta['imgs']
        self.poses = fisheye_meta['pose']
        self.mask = fisheye_meta['mask']
        self.meta = fisheye_meta['meta']
        self.device = "cuda"
        self.num_imgs = self.fisheye_imgs.shape[0]
        self.height = self.fisheye_imgs.shape[1]
        self.width = self.fisheye_imgs.shape[2]

        self.k1 = torch.ones(self.num_imgs,1).to(self.device) * self.meta['k1_03']
        self.k1[::2] = self.meta["k1_02"]
        self.k2 = torch.ones(self.num_imgs,1).to(self.device) * self.meta['k2_03']
        self.k2[::2] = self.meta["k2_02"]
        self.gamma1 = torch.ones(self.num_imgs,1).to(self.device) * self.meta['gamma1_03']
        self.gamma1[::2] = self.meta["gamma1_02"]
        self.gamma2= torch.ones(self.num_imgs,1).to(self.device) * self.meta['gamma2_03']
        self.gamma2[::2] = self.meta["gamma2_02"]
        self.u0 = torch.ones(self.num_imgs,1).to(self.device) * self.meta['u0_03']
        self.u0[::2] = self.meta['u0_02']
        self.v0 = torch.ones(self.num_imgs,1).to(self.device) * self.meta['v0_03']
        self.v0[::2] = self.meta['v0_02']
        self.mirror = torch.ones(self.num_imgs,1).to(self.device) * self.meta['mirror_03']
        self.mirror[::2] = self.meta['mirror_02']
        self.p1 = torch.ones(self.num_imgs,1).to(self.device) * 1.3477698472982495e-03
        self.p1[::2] = 4.2223943394772046e-04
        self.p2 = torch.ones(self.num_imgs,1).to(self.device) * -7.0340482615055284e-04
        self.p2[::2] = 4.2462134260997584e-04

        return

    def generate_fisheye_ray(self,ray_indices) -> RayBundle:

        camera_indices = ray_indices[:, 0] # camera indices
        k1 = self.k1[camera_indices]
        k2 = self.k2[camera_indices]
        gamma1 = self.gamma1[camera_indices]
        gamma2 = self.gamma2[camera_indices]
        u0 = self.u0[camera_indices]
        v0 = self.v0[camera_indices]
        mirror = self.mirror[camera_indices]
        p1 =self.p1[camera_indices]
        p2 =self.p2[camera_indices]

        pixels_y = ray_indices[:, :1] + 0.5 # row indices
        pixels_x = ray_indices[:, 1:2] + 0.5 # col indices

        iter = 10000
        z = torch.linspace(0.0, 1.0, iter).to(self.device).type(torch.float32).unsqueeze(dim=0)
        z = z.repeat(camera_indices.shape[0],1)
        z_after = torch.sqrt(1 - z ** 2) / (z + mirror)
        z_dist = torch.stack([z, z_after]).permute(1, 0, 2)  ##[batch,2,10000]

        ro2 = torch.linspace(0.0, 1.0, iter).to(self.device).type(torch.float32).unsqueeze(dim=0)
        dis_cofficient = 1 + k1 * ro2 + k2 * ro2 * ro2
        ro2_after = torch.sqrt(ro2) * (1 + k1 * ro2 + k2 * ro2 * ro2)
        map_dist = torch.stack([dis_cofficient, ro2_after]).permute(1, 0, 2)  ## [batch,2,10000]

        x = (pixels_x.to(self.device) - u0) / gamma1
        y = (pixels_y.to(self.device) - v0) / gamma2

        ## Undistortion for coordinates
        dist = torch.sqrt(x * x + y * y)
        index = torch.abs(map_dist[:, 1, :] - dist).argmin(dim=-1)
        coor_index = torch.arange(index.shape[0]).to(index)
        index = torch.stack([coor_index,index],dim=0).permute(1,0)
        x /= map_dist[index[:,0], 0, index[:,1]].unsqueeze(dim=-1)
        y /= map_dist[index[:,0], 0, index[:,1]].unsqueeze(dim=-1)

        ## Undistortion z value
        z_after = torch.sqrt(x * x + y * y)
        index = torch.abs(z_dist[:, 1, :] - z_after).argmin(dim=-1)
        coor_index = torch.arange(index.shape[0]).to(index)
        index = torch.stack([coor_index, index], dim=0).permute(1, 0)
        x *= (z_dist[index[:,0], 0, index[:,1]].unsqueeze(dim=-1) + mirror)
        y *= (z_dist[index[:,0], 0, index[:,1]].unsqueeze(dim=-1) + mirror)

        x = x.squeeze(dim=-1)
        y = y.squeeze(dim=-1)
        xy = torch.stack((x, y))
        xys = xy.permute(1, 0)

        print("undistortion value:{}".format(xys))
        exit()
        return


