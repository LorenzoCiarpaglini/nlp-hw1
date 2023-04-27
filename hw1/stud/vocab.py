def sentence_to_char(data_set):
    data_set_new = []
    
    for i, data in enumerate(data_set):
        data_set_new.append(data.copy())

        data_set_new[i]['tokens_char'] = []
        lengths = []
        
        for token in data['tokens']:
            chars = []
            
            for char in token:
                chars += [char]
                
            lengths.append(len(chars))
            data_set_new[i]['tokens_char'].append(chars)
            

        data_set_new[i]['char_len'] = lengths
  
    return data_set_new

def sentence_to_pos(data_set):
    data_set_new = []
    
    for i, data in enumerate(data_set):
        data_set_new.append(data.copy())

        data_set_new[i]['pos_tagged'] = []

        pos_tagged = nltk.pos_tag(data['tokens'])
        
        pos_tagged_filtered = list(map(lambda x: x[1], pos_tagged))

        data_set_new[i]['pos_tagged'] = pos_tagged_filtered
  
    return data_set_new

def createVocabolary(data_set):
  vocab = {}
  tags = {}
  # words_set = set()
  words_set = []
  tags_set = []

  for index, data in enumerate(data_set):
    # data = json.loads(data)
    
    for token in data['tokens']:
      # words_set.add(token)
      if token not in words_set:
        words_set.append(token)

    for label in data['labels']:
      # words_set.add(token)
      if label not in tags_set:
        tags_set.append(label)
        
    if index % 1000 == 0:
        print(index)
  
  for idx, word in enumerate(words_set):
    vocab[word] = idx
  
  vocab['UNK'] = idx + 1
  vocab['PAD'] = idx + 2

  for idx, tag in enumerate(tags_set):
    tags[tag] = idx

  return vocab, tags

def createVocabolaryChar(data_set, v0 = False):
  vocab_char = {}
  # words_set = set()
  chars_set = []


  for index, data in enumerate(data_set):
    # data = json.loads(data)
    
    if v0:
        for char in data['tokens_char']:
            if char not in chars_set:
              chars_set.append(char)
    else:
        for chars in data['tokens_char']:
          for char in chars:
            if char not in chars_set:
              chars_set.append(char)

    if index % 1000 == 0:
        print(index)

  for idx, char in enumerate(chars_set):
    vocab_char[char] = idx
  
  vocab_char['UNK'] = idx + 1
  vocab_char['PAD'] = idx + 2

  return vocab_char

def createVocabolaryPos(data_set):
  vocab_pos = {}
  # words_set = set()
  pos_set = []

  for index, data in enumerate(data_set):
    # data = json.loads(data)
        
    for pos in data['pos_tagged']:
      if pos not in pos_set:
        pos_set.append(pos)

    if index % 1000 == 0:
        print(index)

  for idx, pos in enumerate(pos_set):
    vocab_pos[pos] = idx
  
  #vocab_pos['UNK'] = idx + 1
  vocab_pos['PAD'] = idx + 1

  return vocab_pos

def createWordsFile(data_set):
  return

def map_to_dict(data_set, vocab, vocab_char, vocab_pos, tags, v0 = False):
    data_set_new = []

    for i, data in enumerate(data_set):
        data_set_new.append(data.copy())

        data_set_new[i]['tokens_mapped'] = []
        data_set_new[i]['chars_mapped'] = []
        data_set_new[i]['pos_mapped'] = []
        data_set_new[i]['labels_mapped'] = []

        for token in data['tokens']:
            data_set_new[i]['tokens_mapped'].append(vocab[token] if token in vocab else vocab['UNK'])
        if v0:
            for char in data['tokens_char']:
                data_set_new[i]['chars_mapped'].append(vocab_char[char] if char in vocab_char else vocab_char['UNK'])
        else:
            for chars in data['tokens_char']:
                chars_mapped = []
                
                for char in chars:
                    chars_mapped.append(vocab_char[char] if char in vocab_char else vocab_char['UNK'])
                    data_set_new[i]['chars_mapped'].append(chars_mapped)
        
        for pos in data['pos_tagged']:
            data_set_new[i]['pos_mapped'].append(vocab_pos[pos])  #possible that i encounter a pos tag not in vocab
        
        for label in data['labels']:
            data_set_new[i]['labels_mapped'].append(tags[label])

        data_set_new[i]['len'] = len(data['tokens'])
  
  return data_set_new


def get_max_len(data_set):
    longest = 0
    for i, elem in enumerate(data_set):
        if elem['len'] > longest:
        longest = elem['len']
    return longest

def get_max_char_len(data_set):
    longest = 0
    for i, elem in enumerate(data_set):
        if max(elem['char_len']) > longest:
            longest = max(elem['char_len'])
    return longest