def get_params(fileName):
    hyphen_idx = [idx for idx in range(len(fileName)) if fileName[idx] == "-"]
    print(hyphen_idx)
    return {
        'mode': int(fileName[hyphen_idx[0]+1:hyphen_idx[1]]),
        'stochastic_gradient': True if fileName[hyphen_idx[2]+1:hyphen_idx[3]] == "True" else False,
        'stochastic_value': int(fileName[hyphen_idx[4]+1:hyphen_idx[5]]),
        'threshCollectTill': float(fileName[hyphen_idx[6]+1:hyphen_idx[7]]),
        'threshTau': float(fileName[hyphen_idx[8]+1:hyphen_idx[9]]),
        'train_len': int(fileName[hyphen_idx[10]+1:hyphen_idx[11]]),
        'test_len': int(fileName[hyphen_idx[12]+1:-5])
    }