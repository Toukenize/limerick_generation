import torch
import string
import re
import numpy as np
import argparse

from utils import *
from config import *
from torch import nn
from torch.nn import functional as F

def doc2idx_and_pad(token_list, vocab, max_len):
    
    token_pad = (0, max_len - len(token_list))
    token_idx = torch.tensor(vocab.doc2idx(token_list), dtype=torch.long)
    token_idx = F.pad(token_idx, token_pad, 'constant')
    token_idx[token_idx < 0] = 0  # deal with rare cases where there are missing characters

    return token_idx
    
def word2char(token):
    subtokens = list(iter(token))
    return subtokens
    
def obtain_next_word(outputs, temperature, vocab, masked_idx=[]):
    
    word_weights = outputs.div(temperature).exp().cpu()[0,-1] # Pick the last sets of weights
        
    if word_weights.argmax() == 1:
        new_word_idx = torch.tensor([1], dtype=torch.long).unsqueeze(0).to(DEVICE)
        new_word = vocab[new_word_idx.item()]
        assert new_word == '<linebreak>', 'Check index of <linebreak>'
        return new_word_idx, new_word 
    
    if len(masked_idx) > 0:
        assert (word_weights >= 0).all().item(), 'Not all word weights are positive'
        word_mask = torch.zeros_like(word_weights)
        word_mask[masked_idx] = 1                                             # Set those idx in masked_idx to 1 (i.e. selected)
        word_weights = word_weights.where(word_mask == 1, torch.tensor(0.))   # For vocab not in masked_idx, set weight to 0
    
    new_word_idx = torch.multinomial(word_weights, 1)                         # This is randomized based on word_weights distribution
    new_word_idx = new_word_idx.unsqueeze(0).to(DEVICE)
    
    new_word = vocab[new_word_idx.item()]

    return new_word_idx, new_word
    
def obtain_most_probable_next_word(outputs, vocab):

    new_word_idx = outputs.cpu()[0,-1].argmax()
    new_word = vocab[new_word_idx.item()]
    
    return new_word_idx.item(), new_word
    
def obtain_sound(new_word):
    
    last_sound =  vocab_to_sound.get(new_word) or ['<missing>']  # Get final sound token
    last_sound = last_sound[-1]
    sound_token = sound_vocab.token2id[last_sound]
    sound_idx = torch.tensor([sound_token]).unsqueeze(0).to(DEVICE)
    
    #print(sound_idx)
    return sound_idx

def obtain_syllable_length(new_word):
    if new_word != '<linebreak>':
        sylla_len = len(vocab_to_syllable.get(new_word) or []) / SYLLA_NORM
    else:
        sylla_len = 0.0

    sylla_len = torch.tensor([sylla_len]).unsqueeze(0).to(DEVICE)
    #print(sylla_len)
    return sylla_len
    
def obtain_chars(new_word):
    if new_word == '<linebreak>':
        chars = torch.zeros(16, dtype=torch.long).unsqueeze(0).unsqueeze(0).to(DEVICE)
    else:
        chars = doc2idx_and_pad(word2char(new_word), char_vocab, CHAR_LEN).unsqueeze(0).unsqueeze(0).to(DEVICE)
    #print(chars)
    return chars
    
def initialize_inputs():
    words = torch.tensor([[2]]).to(DEVICE)
    sounds = torch.tensor([[2]]).to(DEVICE)
    sylla_lens = torch.zeros(1, dtype=torch.float).unsqueeze(0).to(DEVICE)
    chars = torch.zeros(CHAR_LEN, dtype=torch.long).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    return words, sounds, sylla_lens, chars

def repeated_whole_sequence_inference(model, max_len, temperature, manual_dropout=0.7):
    
    model.eval()
    
    word_idx, sound_idx, sylla_len, chars_idx = initialize_inputs()
    
    sentence = []
    sounds = []
    
    num_words = 0
    num_lines = 0
        
    while (num_lines < 5) and (num_words < max_len):
                
        with torch.no_grad():
            outputs, sound_outputs = model.forward(word_idx, sound_idx, sylla_len, chars_idx, init_hidden=True, manual_dropout=manual_dropout)

        # Unique to multi_loss models
        pred_sound_idx, pred_sound_word = obtain_most_probable_next_word(sound_outputs, vocab=rhyme_sound_vocab)
        
        if pred_sound_word != '<nls>':
            words_w_correct_sound = sound_to_vocab[pred_sound_word]                
            correct_mask_idx = np.r_[words_w_correct_sound]
        else:
            correct_mask_idx = []
            
        new_word_idx, new_word = obtain_next_word(outputs, temperature, vocab=word_vocab, masked_idx=correct_mask_idx)
        
        new_sound_idx = obtain_sound(new_word)            # sound token of the predicted word NOT the predicted sound token
        new_sylla_len = obtain_syllable_length(new_word)
        new_chars_idx = obtain_chars(new_word)
        
        word_idx = torch.cat((word_idx, new_word_idx), dim=-1)
        sound_idx = torch.cat((sound_idx, new_sound_idx), dim=-1)
        sylla_len = torch.cat((sylla_len, new_sylla_len), dim=-1)
        chars_idx = torch.cat((chars_idx, new_chars_idx), dim=1)
    
        #print(word_idx.shape, sound_idx.shape, sylla_len.shape, chars_idx.shape)
        sounds.append(pred_sound_word)
        sentence.append(new_word)
        num_words += 1
        
        if new_word == '<linebreak>':
            num_lines += 1
        
    else:
        return sentence, sounds, outputs
        
def get_formated_limerick(example):
    string_output = ' '.join(example)
    string_output = re.sub(r'<linebreak>( |\b|)','\n',string_output)

    return string_output
    
def get_formated_limerick_sound(limerick, limerick_sound):

    formatted_sound = []
    linebreak = 0

    for l, s in zip(limerick, limerick_sound):

        if linebreak == 5:
            break

        if l == '<linebreak>':
            linebreak += 1
            formatted_sound.append('\n')
        else:
            formatted_sound.append(s)
    sound_string = ' '.join(formatted_sound)
    sound_string = re.sub('\n ','\n', sound_string)

    return sound_string
    
def print_limerick_and_sound(lim_str, sound_str):
    print('='*101)
    print('Limerick'.center(50) + '|' + 'Sound Tokens'.center(50))
    print('='*101)

    for lim, lim_sound in zip(lim_str.split('\n')[:-1], sound_str.split('\n')[:-1]):    
        print(lim.strip().center(50) + '|' + lim_sound.strip().center(50))
        
    print('='*101 + '\n')

def print_limerick(lim_str):
    print('='*101)
    print('Limerick'.center(101))
    print('='*101)

    for lim in lim_str.split('\n')[:-1]:    
        print(lim.strip().center(101))
    print('='*101 + '\n')
    
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", 
        type=str,   
        help="Select model for limerick generation.", 
        required=True,
        choices=['contx_gru','conv_gru']
        )

    parser.add_argument("--temp", 
        type=float, 
        help="Specify the temperature for Generation. (Default = 1.0, Recommended = 0.8 to 1.2)",
        default=1.0)
        
    parser.add_argument("--dropout", 
        type=float, 
        help="Specify the dropout rate before the final dense layer. (Default = 0.0, Recommended = 0.0 -  to maintain limerick structure)",
        default=0.0)

    args = parser.parse_args()

    return args

def main():
    prompt = 'Generate a limerick (Y - Yes, S - Yes w Sound Tokens, N - No)? ' + '>>> '
    
    while (user_input := input(prompt).strip().upper()) != 'N':

        if user_input not in ['Y','S']:
            print(f'Invalid Option "{user_input}"\n')
            continue
        
        else:
            limerick, limerick_sound, _ = repeated_whole_sequence_inference(
                                        model, 
                                        max_len=100, 
                                        temperature=1.0,
                                        manual_dropout=0)

            lim_str = get_formated_limerick(limerick)
            sound_str = get_formated_limerick_sound(limerick, limerick_sound)
            
            if user_input == 'Y':
                print_limerick(lim_str)
            else:
                print_limerick_and_sound(lim_str, sound_str)
    
    print('Exiting program')
    
    return 
    
    
if __name__ == "__main__":
    args = get_args()
    sound_vocab, word_vocab, char_vocab, rhyme_sound_vocab = get_vocab()
    sound_to_vocab = get_sound_to_vocab_lookup()
    vocab_to_sound = get_vocab_to_sound_lookup()
    vocab_to_syllable = get_vocab_to_syllable_lookup()
    model = get_model(args.model)
    main()