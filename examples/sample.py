import smplx
import torch
import vposer
import numpy as np
import pyvista as pv

bm = smplx.build_layer('./models','smplx')
vp = vposer.create('./models/vposer/V02_05_epoch=13_val_loss=0.03.ckpt',2)
vpbm = vposer.VPoserBodyModel(bm,vp)

with torch.no_grad():
    bodies = vpbm.sample_bodies(50)
    f = bm.faces.astype('int')
    pl = pv.Plotter()
    for i in range(50):
        v = bodies.vertices[i].numpy()+[i%10,(i//10)*2,0]
        mesh = pv.make_tri_mesh(v,f)
        pl.add_mesh(mesh)
    pl.view_xy()
    pl.parallel_projection = True
    pl.parallel_scale = 5
    pl.show(screenshot='./assets/samples.png')