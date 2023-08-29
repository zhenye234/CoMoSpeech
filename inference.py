import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write

import torch
import math
import params
from model.tts import Comospeech
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

import sys
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN


HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default='text.txt', help='path to a file with texts to synthesize')
    parser.add_argument('-c', '--checkpoint', type=str, default='', help='path to a checkpoint of Grad-TTS')
    parser.add_argument('-t', '--timesteps', type=int,  default=1, help='number of sampling timesteps')
 
    args = parser.parse_args()
    
 
    
    print('Initializing Grad-TTS...')
 
    if params.teacher:
        generator = Comospeech(len(symbols)+1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, ).cuda()
 
    else:
        generator = Comospeech(len(symbols)+1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats,teacher=False).cuda()
 

    generator.load_state_dict(torch.load(args.checkpoint, map_location=lambda loc, storage: loc))
    _ = generator.cuda().eval()
    print(f'Number of parameters: {generator.nparams}')
    
    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()
    
    with open(args.file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    cmu = cmudict.CMUDict('./resources/cmu_dictionary')
    tall=[]
    save_dir='out/'
    os.makedirs(save_dir,exist_ok=True) 
    with torch.no_grad():
        for i, text in enumerate(texts):
            print(f'Synthesizing {i} text...', end=' ')
            x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
            x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
            
            t = dt.datetime.now()
            y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=args.timesteps)
            t = (dt.datetime.now() - t).total_seconds()

            print(f' RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')
            
 
            audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
            
            write(save_dir+f'sample_{str(i).zfill(3)}.wav', 22050, audio)

    print('Done. Check out `out` folder for samples.')
