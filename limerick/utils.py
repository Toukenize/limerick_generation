import json
import os
import torch

from torch import nn
from torch.nn import functional as F
from gensim.corpora import Dictionary
from config import *

def get_vocab():
    with open(VOCAB_PATH, 'r') as handle:
        vocab = json.load(handle)
    
    with open(SPECIAL_TOKEN_PATH,'r') as handle:
        specials = json.load(handle)

    sound_vocab = Dictionary(vocab['sound'])
    word_vocab = Dictionary(vocab['word'])
    char_vocab = Dictionary(vocab['char'])
    rhyme_vocab = Dictionary(vocab['rhyme'])
    
    sound_vocab.patch_with_special_tokens(specials['other_special'])
    word_vocab.patch_with_special_tokens(specials['other_special'])
    char_vocab.patch_with_special_tokens(specials['other_special'])
    rhyme_vocab.patch_with_special_tokens(specials['rhyme_special'])
    
    print('Successfully loaded vocab & special tokens')
    
    return sound_vocab, word_vocab, char_vocab, rhyme_vocab
    
def get_sound_to_vocab_lookup():
    
    with open(SOUND_TO_VOCAB_LOOKUP_PATH, 'r') as handle:
        sound_to_vocab_lookup = json.load(handle)
    
    print('Successfully loaded sound to vocab lookup')

    return sound_to_vocab_lookup

def get_vocab_to_sound_lookup():
    
    with open(VOCAB_TO_SOUND_LOOKUP_PATH, 'r') as handle:
        sound_to_vocab_lookup = json.load(handle)
    
    print('Successfully loaded vocab to sound lookup')

    return sound_to_vocab_lookup

def get_vocab_to_syllable_lookup():
    
    with open(VOCAB_TO_SYLLABLE_LOOKUP_PATH, 'r') as handle:
        sound_to_vocab_lookup = json.load(handle)
    
    print('Successfully loaded vocab to syllable lookup')

    return sound_to_vocab_lookup

class Conv_GRU_Multi_Loss(nn.Module):
    
    def __init__(self, 
                 num_layers=1,
                 word_vocab_size=1000, word_embedding_size=32, 
                 char_vocab_size=50, char_embedding_size=16,
                 sound_vocab_size=50, sound_embedding_size=16,
                 rhyme_vocab_size=50,
                 hidden_size=32, dropout=0.5):
        
        super(Conv_GRU_Multi_Loss, self).__init__()
        
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_size, padding_idx=0)   
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_size, padding_idx=0)     
        self.sound_embedding = nn.Embedding(sound_vocab_size, sound_embedding_size, padding_idx=0)     
        
        # 2 channels are the stacked embeddings (both embeddings must be the same size)
        # Convulution over the embeddings (channel-wise), preserving the same sequence (no conv over sequence)
        
        self.conv2d = nn.Conv2d(2, 1, (1,7), padding=(0,3)) 
        
        self.gru = nn.GRU(word_embedding_size, hidden_size, num_layers,batch_first=True)
        self.char_gru = nn.GRU(char_embedding_size, hidden_size, num_layers,batch_first=True)
        
        # FC Layer Input Shape = Sound Hidden Size + Word Hidden Size + Syllable Length Dimension
        # Thus Input Shape = hidden_size + 1
        self.norm_1 = nn.LayerNorm(hidden_size + 1)
        self.fc = nn.Linear(hidden_size + 1, hidden_size)
        self.dropout = nn.Dropout(p = dropout)
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, word_vocab_size)
        self.aux_classifier = nn.Linear(hidden_size, rhyme_vocab_size)

        self.hidden_dim = hidden_size
        self.num_layers = num_layers
        
        #print(f'MultiEmbeddingGenerator initialized with {self.count_parameters():,} trainable parameters:\n')
        #print('Summary:')
        #print(self)
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(DEVICE)      
        
    def forward(self, 
                words, 
                sounds, 
                sylla_len,
                chars, 
                manual_dropout=0.0,
                init_hidden=True):
        
        # Characters' hidden state should always be reset to 0.
        # Because in this architecture, we only want to pass the spelling structure of individual word to the GRU layers.
        # Thus, the current hidden state for each word is independent from the last word.
        self.char_hidden = self.init_hidden(words.shape[0])

        if init_hidden:
            self.hidden = self.init_hidden(words.shape[0])
        
        word_embeds = self.word_embedding(words)
        sound_embeds = self.sound_embedding(sounds)
                
        stacked_embeds = torch.cat((word_embeds.unsqueeze(1), sound_embeds.unsqueeze(1)), dim=1)
        conv_embeds = self.conv2d(stacked_embeds)
        conv_embeds = conv_embeds.squeeze(1)
        
        for c in chars.permute(2,0,1):
            
            char_embeds = self.char_embedding(c)
            _, self.char_hidden = self.char_gru(char_embeds, self.char_hidden) # Only the final hidden state is used

        word_output, self.hidden = self.gru(conv_embeds, self.hidden + self.char_hidden)
        
        output = torch.cat([word_output, sylla_len.unsqueeze(-1)], dim=-1)

        batch_size, sequence_len, hidden_size = output.shape
        
        output = output.contiguous().view(batch_size * sequence_len, hidden_size)
        output = self.norm_1(output)
        output = self.fc(F.relu(output))
        output = self.norm_2(output)
        output = self.dropout(output)
        output = F.dropout(output, manual_dropout)
        
        word_output = self.classifier(output).view(batch_size, sequence_len, -1)
        sound_output = self.aux_classifier(output).view(batch_size, sequence_len, -1)
                
        return word_output, sound_output
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class Contx_GRU_Multi_Loss(nn.Module):
    
    def __init__(self, 
                 num_layers=1,
                 word_vocab_size=1000, word_embedding_size=32, 
                 char_vocab_size=50, char_embedding_size=16,
                 sound_vocab_size=50, sound_embedding_size=16,
                 rhyme_vocab_size=50,
                 hidden_size=32, dropout=0.5):
        
        super(Contx_GRU_Multi_Loss, self).__init__()
        
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_size, padding_idx=0)   
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_size, padding_idx=0)     
        self.sound_embedding = nn.Embedding(sound_vocab_size, sound_embedding_size, padding_idx=0)     
        
        self.word_gru = nn.GRU(word_embedding_size, hidden_size, num_layers,batch_first=True)
        self.char_gru = nn.GRU(char_embedding_size, hidden_size, num_layers,batch_first=True)
        self.sound_gru = nn.GRU(sound_embedding_size, hidden_size, num_layers,batch_first=True)
        
        # FC Layer Input Shape = Sound Hidden Size + Word Hidden Size + Syllable Length Dimension
        # Thus Input Shape = hidden_size * 2 + 1
        self.norm_1 = nn.LayerNorm(hidden_size + 1)
        self.fc = nn.Linear(hidden_size + 1, hidden_size)
        self.dropout = nn.Dropout(p = dropout)
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, word_vocab_size)
        self.aux_classifier = nn.Linear(hidden_size, rhyme_vocab_size)
        
        self.hidden_dim = hidden_size
        self.num_layers = num_layers
        
        #print(f'MultiEmbeddingGenerator initialized with {self.count_parameters():,} trainable parameters:\n')
        #print('Summary:')
        #print(self)
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(DEVICE)      
        
    def forward(self, 
                words, 
                sounds, 
                sylla_len,
                chars, 
                manual_dropout=0.0,
                init_hidden=True):
        
        # Characters' hidden state should always be reset to 0.
        # Because in this architecture, we only want to pass the spelling structure of individual word to the GRU layers.
        # Thus, the current hidden state for each word is independent from the last word.
        self.char_hidden = self.init_hidden(words.shape[0])

        if init_hidden:
            self.word_hidden = self.init_hidden(words.shape[0])
            self.sound_hidden = self.init_hidden(words.shape[0])
        
        word_embeds = self.word_embedding(words)
        sound_embeds = self.sound_embedding(sounds)
        
        for c in chars.permute(2,0,1):
            
            char_embeds = self.char_embedding(c)
            _, self.char_hidden = self.char_gru(char_embeds, self.char_hidden) # Only the final hidden state is used

        sound_output, self.sound_hidden = self.sound_gru(sound_embeds, self.sound_hidden + self.char_hidden)
        word_output, self.word_hidden = self.word_gru(word_embeds, self.word_hidden + self.char_hidden + 0.5 * self.sound_hidden)
        
        output = torch.cat([word_output, sylla_len.unsqueeze(-1)], dim=-1)

        batch_size, sequence_len, hidden_size = output.shape
        
        output = output.contiguous().view(batch_size * sequence_len, hidden_size)
        output = self.norm_1(output)
        output = self.fc(F.relu(output))
        output = self.norm_2(output)
        output = self.dropout(output)
        output = F.dropout(output, manual_dropout)
        word_output = self.classifier(output).view(batch_size, sequence_len, -1)
        sound_output = self.aux_classifier(output).view(batch_size, sequence_len, -1)
        
                
        return word_output, sound_output
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
def get_model(model_type):
    
    assert model_type in ['contx_gru','conv_gru'], 'Model type must be "contx_gru" or "conv_gru"'
    
    if model_type == 'contx_gru':
        model = Contx_GRU_Multi_Loss(**CONTX_GRU_PARAMS).cuda()
        model.load_state_dict(torch.load(CONTX_GRU_PATH))
    else:
        model = Conv_GRU_Multi_Loss(**CONV_GRU_PARAMS).cuda()
        model.load_state_dict(torch.load(CONV_GRU_PATH))
    
    print(f'Successfully loaded model {model_type}\n')

    return model