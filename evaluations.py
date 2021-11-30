import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from scipy.misc import imread, imresize
from PIL import Image
import nltk
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
from speaksee.evaluation import Bleu, Meteor, Rouge, Cider, Spice
from speaksee.evaluation import PTBTokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3,4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def caption_image_beam_search(encoder, decoder, image, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image: input image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    #img = imread(image_path)
    #if len(img.shape) == 2:
    #    img = img[:, :, np.newaxis]
    #    img = np.concatenate([img, img, img], axis=2)
    #img = imresize(img, (256, 256))
    #img = img.transpose(2, 0, 1)
    #img = img / 255.
    #img = torch.FloatTensor(img).to(device)
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    #transform = transforms.Compose([normalize])
    #print(img)
    #print(img.shape)
    #image = transform(img)  # (3, 256, 256)

    # Encode
    #image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    encoder_out = torch.cat((encoder_out, encoder_out), axis=3)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        #print('next_word_inds:', next_word_inds)
        #print('word_map[\'<end>\']:', word_map['<end>'])
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
        #print('incomplete_inds:', incomplete_inds)
        #print('complete_inds:', complete_inds)
        # Set aside complete sequences
        #print(complete_inds)
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    #if len(complete_seqs) == 0:
    #    return (0, 0)
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas

if __name__ == '__main__':

    # data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])    
    data_folder = '../../data/OutputDataset/' 
    data_name = 'PCCD5_min_word_freq'
    val_loader = torch.utils.data.DataLoader(
    CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
    batch_size=1, shuffle=True, pin_memory=True) # PCCD
    
    # models
    decoder=torch.load(r'./models/decoder_saved_epoch_33')
    decoder=decoder.to(device)
    decoder.eval()
    
    encoder = torch.load(r'./models/encoder_saved_epoch_33')
    encoder = encoder.to(device)
    encoder.eval()
    
    # Load word map (word2ix)
    wm_dir = '/workspace/data/OutputDataset/WORDMAP.json'
    with open(wm_dir, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
    #print(type(rev_word_map))   # dict {1:'A',...}
    #print(len(rev_word_map))    # 8551
    
    # get evls
    for idx, (img, caps, caps_len, _) in enumerate(val_loader):
        try:
            seq, alphas = caption_image_beam_search(encoder, decoder, img.to(device), word_map, 5)
        except:
            continue
        #print(seq)       
        #print(list(np.array(caps.tolist())[0])[:caps_len.tolist()[0][0]]) 
    
        tagt_cap = []
        pred_cap = []
    
        for i in (list(np.array(caps.tolist())[0])[:caps_len.tolist()[0][0]])[1:-1]:
            tagt_cap.append(rev_word_map[i])
        
        for j in seq[1:-1]:
            pred_cap.append(rev_word_map[j])
        
        if pred_cap.count('.')==1:
            pass
        else:
            tmp1 = []
            tmp2 = []
            for i in pred_cap:
                tmp1.append(i)
                if i=='.':
                    tmp2.append(tmp1)
                    tmp1 = []
            pred_cap = tmp2
        
        #print('target caption:\n', tagt_cap)
        #print('pred caption:\n', pred_cap)
        #print('tagt:', tagt_cap.count('.'))
        #print('pred:', pred_cap.count('.'))        
        #break
        #print('bleu:', sentence_bleu(pred_cap, tagt_cap))

        '''
        cap_tagt = {}
        cap_pred = {}
        
        for idx, cap in enumerate(pred_cap):
            cap_pred[idx] = cap
        
        print('length of tagt: ', len(tagt_cap))
        print('length of pred: ', len(pred_cap))
        print('tagt:', tagt_cap)
        print('pred:', pred_cap[0])
        cap_tagt[0] = tagt_cap
        cap_pred[0] = pred_cap[0]
        
        val_bleu, _ = Bleu(n=4).compute_score(cap_tagt, cap_pred)
'''


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    