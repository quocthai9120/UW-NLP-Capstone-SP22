def trigram(train_data, unknown_chars):
    count_dict = {} # NOTE: this is a nested dictionary
    for line in train_data:
        tok_1, tok_2 = '<start>', '<start>'
        for token in line:
            if token in unknown_chars:
                token = '<unk>'
            base_str = tok_1 + tok_2
            if base_str not in count_dict:
                count_dict[base_str] = {} 
            count_dict[base_str][token] = count_dict[base_str][token] + 1 if token in count_dict[base_str] else 1
            tok_1 = tok_2
            tok_2 = token
        base_str = tok_1 + tok_2
        if base_str not in count_dict:
                count_dict[base_str] = {} 
        count_dict[base_str]['<stop>'] = count_dict[base_str]['<stop>'] + 1 if '<stop>' in count_dict[base_str] else 1
    sum_map = {}
    for sequence in count_dict:
        ch_map = count_dict[sequence]
        count_seq = sum(ch_map.values())
        sum_map[sequence] = count_seq
    
    return sum_map, count_dict

def bigram(train_data, unknown_chars):
    count_dict = {} # NOTE: this is a nested dictionary
    for line in train_data:
        prev_token = '<start>'
        for token in line:
            if token in unknown_chars:
                token = '<unk>'
            if prev_token not in count_dict:
                count_dict[prev_token] = {} 
            count_dict[prev_token][token] = count_dict[prev_token][token] + 1 if token in count_dict[prev_token] else 1
            prev_token = token
        if prev_token not in count_dict:
                count_dict[prev_token] = {} 
        count_dict[prev_token]['<stop>'] = count_dict[prev_token]['<stop>'] + 1 if '<stop>' in count_dict[prev_token] else 1
    sum_map = {}
    for sequence in count_dict:
        ch_map = count_dict[sequence]
        count_seq = sum(ch_map.values())
        sum_map[sequence] = count_seq
    
    return sum_map, count_dict


def unigram(train_data):
    N = 0;
    char_count = {}
    stop_count = 0
    for line in train_data:
        for token in line:
            N += 1
            char_count[token] = char_count[token] + 1 if token in char_count else 1
        stop_count += 1
    char_count['<stop>'] = stop_count
    N += stop_count

    unknown_chars = set()
    unknown_count = 0
    for ch in char_count:
        if char_count[ch] < 5: 
            unknown_chars.add(ch)
            unknown_count += char_count[ch]
    char_count['<unk>'] = unknown_count
    for ch in unknown_chars:
        del char_count[ch]

    return N, char_count, unknown_chars