import os
import argparse
import torch

def clean_v2(vposer_dir):
    from human_body_prior.tools.configurations import load_config
    from human_body_prior.models.vposer_model import VPoser
    
    def load_model(model_ps,trained_weigths_fname, model_code=None, remove_words_in_model_weights=None, load_only_ps=False, disable_grad=True, custom_ps = None):
        if load_only_ps: return model_ps
        if custom_ps is not None: model_ps = custom_ps
        assert model_code is not None, ValueError('mode_code should be provided')
        model_instance = model_code(model_ps)
        if disable_grad: # i had to do this. torch.no_grad() couldnt achieve what i was looking for
            for param in model_instance.parameters():
                param.requires_grad = False
        state_dict = torch.load(trained_weigths_fname)['state_dict']
        if remove_words_in_model_weights is not None:
            words = '{}'.format(remove_words_in_model_weights)
            state_dict = {k.replace(words, '') if k.startswith(words) else k: v for k, v in state_dict.items()}

        ## keys that were in the model trained file and not in the current model
        instance_model_keys = list(model_instance.state_dict().keys())
        trained_model_keys = list(state_dict.keys())
        wts_in_model_not_in_file = set(instance_model_keys).difference(set(trained_model_keys))
        ## keys that are in the current model not in the training weights
        wts_in_file_not_in_model = set(trained_model_keys).difference(set(instance_model_keys))
        # assert len(wts_in_model_not_in_file) == 0, ValueError('Some model weights are not present in the pretrained file. {}'.format(wts_in_model_not_in_file))

        state_dict = {k:v for k, v in state_dict.items() if k in instance_model_keys}
        model_instance.load_state_dict(state_dict, strict=False) # Todo fix the issues so that we can set the strict to true. The body model uses unnecessary registered buffers
        model_instance.eval()

        return model_instance, model_ps

    model_ps = load_config(f'{vposer_dir}/V02_05.yaml')
    trained_weigths_fnames = [
        f'{vposer_dir}/snapshots/V02_05_epoch=08_val_loss=0.03.ckpt',
        f'{vposer_dir}/snapshots/V02_05_epoch=13_val_loss=0.03.ckpt']
    for trained_weigths_fname in trained_weigths_fnames:
        fn = trained_weigths_fname.split('/')[-1]
        vp,ps = load_model(model_ps,trained_weigths_fname, model_code=VPoser,
                                remove_words_in_model_weights='vp_model.',
                                disable_grad=True)
        torch.save(vp.state_dict(),f'{vposer_dir}/{fn}')


def clean_v1(vposer_dir):
    from vposer import VPoserV1 as VPoser
    vposer = VPoser(512,32,[1,21,3])
    state_dict = torch.load(f'{vposer_dir}/snapshots/TR00_E096.pt',map_location='cpu')
    vposer.load_state_dict(state_dict)
    torch.save(vposer.state_dict(),f'{vposer_dir}/TR00_E096.pt')


def main(args):
    clean_v1(f'{args.vposer_dir}/vposer_v1_0')
    clean_v2(f'{args.vposer_dir}/V02_05')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("vposer_dir",type=str)
    args = parser.parse_args()
    main(args)