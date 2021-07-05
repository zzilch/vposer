import numpy as np
import torch
import torch.nn as nn

__all__ = [
    'VPoserBodyModel',
    'VPoserBodyLayer'
]

class VPoserBodyModel(nn.Module):
    def __init__(self, bm, vp, input_type='aa',create_body_pose_z=True, body_pose_z=None):
        ''' VPoserBodyModel constructor

            Parameters
            ----------
            bm: smplx model or layer
            vp: VPoserV1 or VposerV2
            input_type: str
                Rotation type of body pose, `aa` for smplx models and `matrot` for smplx layers.
                (default='aa')
            create_body_pose_z: bool
                If set ture, a built-in parameter `body_pose_z` will be created. (default=False)
            body_pose_z: torch.tensor, optinal, `(bm.batch_szie,vp.latentD)`
                Valid when `create_body_pose_z` is true.
                If given, initialize `body_pose_z` with it.
                If not given, initialize `body_pose_z` with zero values.(default=None)
        '''
        super(VPoserBodyModel,self).__init__()
        assert bm.name() in ['SMPL','SMPL-H','SMPL-X'], f'{bm.name()} is not supported!'
        assert input_type in ['aa','matrot'], f'Rotation type {input_type} is not supported!'
        self.bm = bm
        self.vp = vp
        self.input_type = input_type
        
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
        if body_pose_z is None:
            body_pose_z = self.body_pose_z
        
        decode_results = self.vp.decode(body_pose_z)
        if self.input_type == 'aa':
            body_pose = decode_results['pose_body']
        elif self.input_type == 'matrot':
            body_pose = decode_results['pose_body_matrot']
        
        shape,device = body_pose.shape,body_pose.device
        num_joints = self.bm.NUM_BODY_JOINTS
        if num_joints>21: 
            body_pose = torch.cat(
                [body_pose,torch.zeros((shape[0],num_joints-shape[1],shape[2]),device=device)],
                dim=1
            )
        body_pose = body_pose.view(shape[0],-1)
        return self.bm(body_pose=body_pose,**kwargs)
    
    @torch.no_grad()
    def reset_params(self, **params_dict) -> None:
        if 'body_pose_z' in dict(self.named_parameters()):
            self.body_pose_z.fill_(0)
        self.bm.reset_params()
    
    def sample_bodies(self, seed=None):
        np.random.seed(seed)
        
        dtype = self.body_pose_z.dtype
        device = self.body_pose_z.device
        self.eval()
        with torch.no_grad():
            Zgen = torch.tensor(np.random.normal(0., 1., size=(self.bm.batch_size, self.vp.latentD)),
                                dtype=dtype, device=device)
        
        return self.forward(body_pose_z=Zgen)
    
class VPoserBodyLayer(VPoserBodyModel):
    def __init__(self, bm, vp):
        super(VPoserBodyLayer,self).__init__(bm, vp, 
                                             body_pose_z=None, input_type='matrot',create_body_pose_z=False)
        ''' VPoserBodyLayer constructor

            Parameters
            ----------
            bm: smplx model or layer
            vp: VPoserV1 or VposerV2
            input_type: str
                Rotation type of body pose, `aa` for smplx models and `matrot` for smplx layers.
                (default='aa')
            create_body_pose_z: bool
                If set ture, a built-in parameter `body_pose_z` will be created. (default=False)
            body_pose_z: torch.tensor, optinal, `(bm.batch_szie,vp.latentD)`
                Valid when `create_body_pose_z` is true.
                If given, initialize `body_pose_z` with it.
                If not given, initialize `body_pose_z` with zero values.(default=None)
        '''
    
    def forward(self, body_pose_z=None, **kwargs):
        ''' Parameters
            ----------
            body_pose_z: torch.tensor, optinal, `(batch_szie,vp.latentD)`
                If given, use it instead of built-in `body_pose_z`
                If not given, use `bm.batch_size` and filled with zero values.
                
            Returns
            ----------
            output: smplx output
        '''
        dtype=self.bm.shapedirs.dtype
        device=self.bm.shapedirs.device
        if body_pose_z is None:
            batch_size = self.bm.batch_size
            body_pose_z = torch.zeros([batch_size, self.vp.latentD],
                                      dtype=dtype,device=device)
            
        return super().forward(body_pose_z=body_pose_z)
    
    def sample_bodies(self, num_poses=None, seed=None):
        np.random.seed(seed)
        
        if num_poses is None:
            num_poses = self.bm.batch_size
        dtype = self.bm.shapedirs.dtype
        device = self.bm.shapedirs.device
        self.eval()
        with torch.no_grad():
            Zgen = torch.tensor(np.random.normal(0., 1., size=(num_poses, self.vp.latentD)),
                                dtype=dtype, device=device)
        
        return self(body_pose_z=Zgen)
        