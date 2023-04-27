#hyper parameters
hypers = {
    
    'vocab_size': len(vocab),
    
    'vocab_char_size': len(vocab_char),
    
    'vocab_pos_size': len(vocab_pos),

    'embedding_dim': 512, #512
    
    'embedding_char_dim': 100, #512-100

    'lstm_hidden_dim': 128,
    
    'hidden_dim': 128,

    'number_of_tags': len(tags),

    'input_size': 768,

    'hidden_size': 0,

    'num_classes': 2,

    'learning_rate': 2e-3, #2e-4

    'batch_size': 16,

    'epochs': 1,

    'dropout_rate': 0.3,#0.3

    'print_step': 10,

    'device': 'cuda' if torch.cuda.is_available() else 'cpu'

}

print(hypers['device'])

class Dict2Class(object):
      
    def __init__(self, my_dict):
          
        for key in my_dict:
            setattr(self, key, my_dict[key])

hypers = Dict2Class(hypers)
