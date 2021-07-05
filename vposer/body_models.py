import numpy as np
import torch
import torch.nn as nn

__all__ = [
    'VPoserModelOutput',
    'VPoserBodyModel'
]

class VPoserModelOutput(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __repr__(self) -> str:
        return self.__dict__.__repr__()

    def get(self, key, default=None):
        return self.__dict__.get(key,default=default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

class VPoserBodyModel(nn.Module):
    def __init__(self, bm, vp, create_body_pose_z=None, body_pose_z=None):
        ''' VPoserBodyModel constructor

            Parameters
            ----------
            bm: smplx model or layer
            vp: VPoserV1 or VposerV2
            create_body_pose_z: bool, optional
                If no given, set false when the body model is a layer.
                If set ture, a built-in parameter `body_pose_z` will be created. (default=None)
            body_pose_z: torch.tensor, optinal, `(bm.batch_szie,vp.latentD)`
                Valid when `create_body_pose_z` is true.
                If given, initialize `body_pose_z` with it.
                If not given, initialize `body_pose_z` with zero values.(default=None)
        '''
        super(VPoserBodyModel,self).__init__()
        assert bm.name() in ['SMPL','SMPL+H','SMPL-X'], f'Body model {bm.name()} is not supported!'
        self.bm = bm
        self.vp = vp
        self.is_layer = ('Layer' in bm.__class__.__name__)
        
        if create_body_pose_z is None: create_body_pose_z = not self.is_layer
        if create_body_pose_z:
            device, dtype = self.bm.shapedirs.device, self.bm.shapedirs.dtype
            
            if body_pose_z is None:
                body_pose_z = torch.zeros([self.bm.batch_size, self.vp.latentD],
                                        requires_grad=True,dtype=dtype,device=device)
            else:
                assert len(body_pose_z) == self.bm.batch_size
                if torch.is_tensor(body_pose_z):
                    default_body_pose = body_pose_z.clone().detach()
                else:
                    default_body_pose = torch.tensor(body_pose_z, dtype=dtype, device=device)
            
            self.register_parameter('body_pose_z', nn.Parameter(body_pose_z, requires_grad=True))
        
        for name,param in self.bm.named_parameters():
            setattr(self,name,param)
        
    def forward(self, body_pose_z=None, **kwargs):
        ''' Parameters
            ----------
            body_pose_z: torch.tensor, optinal, `(bm.batch_szie,vp.latentD)`
                If given, use it instead of built-in `body_pose_z`
                If not given, use built-in `body_pose_z`
                
            Returns
            ----------
            output: smplx output
        '''
        dtype = self.bm.shapedirs.dtype
        device = self.bm.shapedirs.device
        batch_size = self.bm.batch_size
        for var in kwargs.values():
            if var is None:
                continue
            batch_size = max(batch_size, len(var))
        if body_pose_z is None:
            if hasattr(self,'body_pose_z'):
                body_pose_z = self.body_pose_z
            else:
                assert self.is_layer or batch_size == self.bm.batch_size,\
                    'Input batch size is inconsistant with body model.'
                body_pose_z = torch.zeros([batch_size, self.vp.latentD],
                                        dtype=dtype,device=device)
        
        decode_results = self.vp.decode(body_pose_z)
        if not self.is_layer:
            body_pose = decode_results['pose_body']
        else:
            body_pose = decode_results['pose_body_matrot']
        
        shape = body_pose.shape
        num_joints = self.bm.NUM_BODY_JOINTS
        if num_joints > shape[1]: 
            body_pose = torch.cat(
                [body_pose,torch.zeros((shape[0],num_joints-shape[1],shape[2]),device=device)],
                dim=1
            )
        body_pose = body_pose.view(shape[0],-1)
        bm_out = self.bm(body_pose=body_pose,**kwargs)
        return VPoserModelOutput(body_pose_z=body_pose_z,**bm_out)
    
    @torch.no_grad()
    def reset_params(self, **params_dict) -> None:
        if hasattr(self,'body_pose_z'):
            self.body_pose_z.fill_(0)
        self.bm.reset_params()
    
    def sample_bodies(self, num_poses=None, seed=None):
        '''Parameters
            ----------
            num_poses: int, optinal, 
                Batch size of samples, valid only when the body model is layer
            seed: random seed
            Returns
            ----------
            output: smplx output
        '''
        np.random.seed(seed)
        
        dtype = self.bm.shapedirs.dtype
        device = self.bm.shapedirs.device
        batch_size = self.bm.batch_size
        if self.is_layer and num_poses is not None:
            batch_size = num_poses
        
        self.eval()
        with torch.no_grad():
            Zgen = torch.tensor(np.random.normal(0., 1., size=(batch_size, self.vp.latentD)),
                                dtype=dtype, device=device)
        
        return self.forward(body_pose_z=Zgen)
        