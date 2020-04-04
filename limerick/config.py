import os
import torch

SEQ_LEN = 64
SYLLABLE_LEN = 5
SOUND_LEN = 2
CHAR_LEN = 16
SYLLA_NORM = 10

WORD_VOCAB_LEN = 48644
CHAR_VOCAB_LEN = 29
SOUND_VOCAB_LEN = 950
RHYME_VOCAB_LEN = 663

VOCAB_PATH = os.path.join('..','vocab','vocab.json')
SPECIAL_TOKEN_PATH = os.path.join('..','vocab','special_tokens.json')
SOUND_TO_VOCAB_LOOKUP_PATH = os.path.join('..','vocab','sound_to_vocab_id.json')
VOCAB_TO_SOUND_LOOKUP_PATH = os.path.join('..','vocab','vocab_to_sound.json')
VOCAB_TO_SYLLABLE_LOOKUP_PATH = os.path.join('..','vocab','vocab_to_syllable.json')

CONTX_GRU_PATH = os.path.join('..','models','contx_gru_mo_100_es.pth')
CONV_GRU_PATH = os.path.join('..','models','conv_gru_mo_100_es.pth')

CONTX_GRU_PARAMS = dict(
    num_layers=1,
    word_vocab_size=WORD_VOCAB_LEN, word_embedding_size=64, 
    char_vocab_size=CHAR_VOCAB_LEN, char_embedding_size=8,
    sound_vocab_size=SOUND_VOCAB_LEN, sound_embedding_size=32,
    rhyme_vocab_size=RHYME_VOCAB_LEN,
    hidden_size=32, dropout=0.5)
    
CONV_GRU_PARAMS = dict(
    num_layers=1,
    word_vocab_size=WORD_VOCAB_LEN, word_embedding_size=64, 
    char_vocab_size=CHAR_VOCAB_LEN, char_embedding_size=8,
    sound_vocab_size=SOUND_VOCAB_LEN, sound_embedding_size=64,
    rhyme_vocab_size=RHYME_VOCAB_LEN,
    hidden_size=32, dropout=0.5)
    
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    