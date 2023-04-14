import matplotlib.pyplot as plt
from nll import * 

if __name__ == '__main__':

    train_len = 3000
    val_len = 1000
    test_len = 3000
    ############ Training error ##############################33

    # generate true data for a hawkes synthetic process omega=1; mu=0.2; alpha=0.8; history_start=0;
    omega = 1
    np.random.seed(0)
    def predict(mu, alpha, history: History):
        last_time = history.time_slots[-1] + 1e-10
        lambda_max = mu + alpha * \
            np.exp((last_time - history.time_slots) * -1 * omega).sum()
        while (True):
            u = np.random.uniform(0, 1)
            last_time = last_time - (np.log(1-u) / lambda_max)
            u2 = np.random.uniform(0, 1)
            value = (mu + alpha * np.exp((last_time - history.time_slots)
                     * -1 * omega).sum()) / lambda_max
            if u2 <= value:
                break
        return last_time


    train_data = [0]
    mu = 0.2
    alpha = 0.8
    prob_log = 0
    for i in range(1, train_len):
        val = predict(0.2, 0.8, History(train_data[:i]))
        train_data.append(val)

    temp = train_data.copy()
    val_data = []
    for i in range(val_len):
        val = predict(0.2, 0.8, History(temp))
        temp.append(val)
        val_data.append(val)
    
    test_data = []
    for i in range(test_len):
        val = predict(0.2, 0.8, History(temp))
        temp.append(val)
        test_data.append(val)

    assert len(train_data) == train_len
    assert len(val_data) == val_len
    assert len(test_data) == test_len


    # try to get likelihood of actual data when alpha and data are obtained through a non-data minimization of train_len params
    for mu in np.arange(0, 0.3, 0.02):
        # mu = mu
        alpha = 1 - mu
        print('mu', mu, 'alpha', alpha)
        
        T = np.max(train_data)
        prob_log = 0
        for i in range(1, train_len):
            prob_log += np.log(mu + alpha * np.exp((train_data[i] - History(train_data[:i]).time_slots) * -1 * omega).sum()) 
            prob_log += alpha/omega * (np.exp(-1 * omega * (T - train_data[i])) - 1)
        prob_log -= mu * (T - train_data[1])
        print('Train', mu, alpha, prob_log/(train_len-1))

        T = np.max(test_data)
        curr = train_data.copy() + val_data.copy()
        prob_log = 0
        for i in range(test_len):
            prob_log += np.log(mu + alpha * np.exp((test_data[i] - History(curr).time_slots) * -1 * omega).sum()) 
            prob_log += alpha/omega * (np.exp(-1 * omega * (T - test_data[i])) - 1)
            curr.append(test_data[i])
        prob_log -= mu * (T - np.max(val_data))
        # prob_log += np.log(mu + alpha * np.exp((test_data[i] - History(curr).time_slots) * -1 * omega).sum()) - mu * (test_data[i] - test_data[i-1]) + alpha/omega * (np.exp(-1 * omega * (test_data[i] - curr[-1])) - 1) 
        print('Test', mu, alpha, prob_log/test_len)


        T = np.max(test_data)
        full_data = train_data + val_data + test_data
        prob_log = 0
        for i in range(1, len(full_data)):
            prob_log += np.log(mu + alpha * np.exp((full_data[i] - History(full_data[:i]).time_slots) * -1 * omega).sum()) 
            prob_log += alpha/omega * (np.exp(-1 * omega * (T - full_data[i])) - 1)
        prob_log -= mu * T
        print('Full', mu, alpha, prob_log/len(full_data))


        T = np.max(val_data)
        full_data = train_data + val_data
        prob_log = 0
        for i in range(1, len(full_data)):
            prob_log += np.log(mu + alpha * np.exp((full_data[i] - History(full_data[:i]).time_slots) * -1 * omega).sum()) 
            prob_log += alpha/omega * (np.exp(-1 * omega * (T - full_data[i])) - 1)
        prob_log -= mu * T
        print('Train + Val', mu, alpha, prob_log/len(full_data))