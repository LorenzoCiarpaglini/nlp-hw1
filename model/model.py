class EventDetection(nn.Module):

    def __init__(self, hypers):
        super(EventDetection, self).__init__()

        # self.embedding = nn.Embedding(hypers.vocab_size + 1, hypers.embedding_dim)
        self.embedding_pos = nn.Embedding(hypers.vocab_pos_size + 1, hypers.embedding_dim, padding_idx = vocab_pos['PAD'])
        self.embedding_char = nn.Embedding(hypers.vocab_char_size + 1, hypers.embedding_char_dim, padding_idx = vocab_char['PAD'])

        #the LSTM takens embedded sentence
        #self.lstm_char = nn.LSTM(hypers.embedding_char_dim * 2, hypers.lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.lstm_char = nn.LSTM(hypers.embedding_char_dim, hypers.lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(hypers.embedding_dim, hypers.lstm_hidden_dim, batch_first=True, bidirectional=True)
        
        #self.convLayer = ConvLayer()

        #fc layer transforms the output to give the final output layer
        self.fc = nn.Linear(hypers.lstm_hidden_dim * 2, hypers.number_of_tags)
        self.fc_word = nn.Linear(300, hypers.embedding_dim)
        self.fc_char = nn.Linear(hypers.lstm_hidden_dim * 2, hypers.embedding_dim)
        self.fc_concat = nn.Linear(hypers.embedding_dim * 3, hypers.embedding_dim)
        

        self.dropout = nn.Dropout(hypers.dropout_rate)

        self.relu = nn.ReLU()

    # def hidden_init(batch_size, hidden_size):
    #   h0 = torch.zeros(batch_size, hidden_size).requires_grad_(False).to(hypers.device)
    #   return h0
    
    def forward(self, s, chars, pos):
        
        # s = self.embedding(s)   # dim: batch_size x batch_max_len x embedding_dim
        pos = self.embedding_pos(pos)
        chars = self.embedding_char(chars)
        
        # batch_size x batch_max_len x embedding_dim ->  
        # batch_size x batch_max_len x 1(50) x embedding_dim
        # batch_size x batch_max_len x 50 x embedding_dim
        
        
        outputs = []
        for i in range(chars.shape[1]):
            out, _ = self.lstm_char(chars[:, i, :, :])
            #print(f'out_shape: {out.shape}')
            outputs.append(out)
        outputs = torch.stack(outputs).to(hypers.device)
        #print(f'out_shape: {outputs.shape}')
        outputs = outputs.permute(1,0,2,3)
        #print(f'out_shape: {outputs.shape}')
        outputs = outputs.mean(dim = 2)  #batch X words_len X (lstm * 2)
        #print(f'out_shape: {outputs.shape}')
        
        outputs = self.dropout(self.relu(self.fc_char(outputs))) #batch X words_len X embedding_dim 
        
        #s = s.unsqueeze(1)
        #pos = pos.unsqueeze(1)
        s = self.fc_word(s)
        
        concat = torch.cat((s, outputs, pos), dim = -1) #batch X words_len X embedding_dim * 3
        
        concat = self.dropout(self.relu(self.fc_concat(concat)))
        
        #s = self.embedding(s)   # dim: batch_size x batch_max_len x embedding_dim

        #run the LSTM along the sentences of length batch_max_len
        concat, _ = self.lstm(concat)     # dim: batch_size x batch_max_len x lstm_hidden_dim                

        #reshape the Variable so that each row contains one token
        concat = concat.reshape(-1, concat.shape[2])  # dim: batch_size*batch_max_len x lstm_hidden_dim

        #apply the fully connected layer and obtain the output for each token
        concat = self.fc(concat)          # dim: batch_size*batch_max_len x num_tags
        concat = self.dropout(concat)

        return F.log_softmax(concat, dim=1)   # 


def loss_fn(outputs, labels):
    #reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.view(-1)  

    #mask out 'PAD' tokens
    mask = (labels >= 0).float()

    #the number of tokens is the sum of elements in mask
    num_tokens = int(torch.sum(mask).item())

    #pick the values corresponding to labels and multiply by mask
    outputs = outputs[range(outputs.shape[0]), labels]*mask

    #cross entropy loss for all non 'PAD' tokens
    return -torch.sum(outputs)/num_tokens