import smplx
import torch
import torch.optim as optim
import vposer
import numpy as np
import pyvista as pv

bm = smplx.create('./models','smplx')
vp = vposer.create('./models/vposer/V02_05_epoch=13_val_loss=0.03.ckpt',2)
vpbm = vposer.VPoserBodyModel(bm,vp)

f = bm.faces.astype('int')
@torch.no_grad()
def fk(model,**kwargs):
    body = model(**kwargs)
    v = body.vertices[0].numpy()
    mesh = pv.make_tri_mesh(v,f)
    return mesh

sample = np.load('./assets/amass_sample.npz')
with torch.no_grad(): body_sample = bm(body_pose=torch.tensor(sample['poses'][:1,3:66]).float())
mesh_sample = pv.make_tri_mesh(body_sample.vertices[0].numpy(),f)
mesh_estimate = fk(vpbm)

for cls in [optim.SGD,optim.Adam,optim.LBFGS]:
    vpbm.reset_params()
    parameters = [vpbm.body_pose_z]
    optimizer = cls(parameters,lr=1)

    def closure():
        optimizer.zero_grad()
        body_estimate = vpbm()
        loss = (body_estimate.joints-body_sample.joints).square().sum()
        loss.backward()
        return loss

    pl = pv.Plotter(window_size=[200,200])
    pl.add_mesh(mesh_sample,color='grey')
    pl.add_mesh(mesh_estimate)
    pl.view_xy()
    pl.open_gif(f"./assets/ik_{cls.__name__}.gif")

    niter = 60
    for i in range(niter):
        loss = optimizer.step(closure)
        print(f'{i}/{niter} loss={loss.item()}')
        pl.update_coordinates(fk(vpbm).points)
        pl.render()
        pl.write_frame()
    pl.close()  
