def downloadDataset(dataset_prefix):
  data_path = os.path.join(DATASET_DIR, dataset_prefix)
  data_path += '.' + file_type_dir

  with open(data_path) as f:
    sentences = f.read().splitlines()

    for i, sentence in enumerate(sentences):
      sentences[i] = json.loads(sentence)
    return sentences


def getLongestAndAvgSentence(data_set):
    longest, idx_longest, avg = 0, 0, 0
    
    for i, data in enumerate(data_set):
        if longest < len(data['labels']):
            longest = len(data['labels'])
            idx_longest = i
        avg += len(data['labels'])
        
    return longest, idx_longest, avg/(i+1)
  
def count_sentences_without_event(data_set, upper_bound_len=25):
    idx_arr = []
    
    
    for i, data in enumerate(data_set):
        no_event = True
        for elem in data['labels']:
            if elem != 'O':
                no_event = False
        if no_event and len(data['labels']) > upper_bound_len:
            idx_arr.append(i)
            
            
    return idx_arr, len(idx_arr)

def clean_dataset(data_set, idx_to_remove):
    new_data_set = []
    for i, data in enumerate(data_set):
        if i not in idx_to_remove:
            new_data_set.append(data)
    return new_data_set