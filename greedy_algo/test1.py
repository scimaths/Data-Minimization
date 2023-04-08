import matplotlib.pyplot as plt
from nll import * 

if __name__ == '__main__':

    train_len = 1500
    test_len = 10
    train_data = None
    with open('data_exp16_train.npy', 'rb') as f:
        train_data = list(np.load(f))
    train_data_history = History(train_data[:train_len])

    test_data = None
    with open('data_exp16_test.npy', 'rb') as f:
        test_data = list(np.load(f))
    test_data_history = History(train_data[train_len:test_len+train_len])

    # timestamps data t1, t2, ... tn
    # we model the excitation intensities that
    # may have caused the data

    # excitation intensities modeled by a hawkes model
    # trained without including sensitivities
    setting1 = Setting1(None, None, None, None, 1, init_num_epochs=1000, lr=1e-4, epoch_decay=None, budget=None, sensitivity=0.1)
    train_data = [0]
    mu = 0.2
    alpha = 0.8
    for i in range(1, train_len):
        val = setting1.predict(mu, alpha, History(train_data[:i]))
        train_data.append(val)

    omega = 1
    final_T = train_data[-1]
    setting1 = Setting1(None, None, None, final_T, omega, init_num_epochs=300, lr=5e-3, epoch_decay=None, budget=None, sensitivity=0.1)
    
    avg1 = 0
    avg2 = 0
    for j in range(15):
        mu, alpha, _ = setting1.do_forward(History(train_data[:(train_len-1)]), [train_data[train_len-1]])
        print(mu, alpha)
        mu = 0.2
        alpha=0.8
        error1 = 0
        error2 = 0
        pred = [0]#train_data[:30]
        for i in range(1,train_len):
            val = setting1.predict(mu, alpha, History(train_data[:i]))
            error1 += (val - train_data[i]) ** 2
            pred.append(val)
            val1 = setting1.predict(mu, alpha, History(train_data[:i]))
            error2 += (val - val1)**2
        print(error1/train_len)
        print(error2/train_len)
        avg1 += np.sqrt(error1/train_len)
        avg2 += np.sqrt(error2/train_len)
    print(avg1/15)
    print(avg2/15)
    intensities = list(map(lambda t: mu.item() + alpha.item() * np.sum(np.exp(-1 * omega * (t - train_data[:train_len+100])[(t - train_data[:train_len+100]) > 0])), train_data[train_len:train_len+100]))

    # excitation intensities modeled by a hawkes model
    # trained including sensitivities    
    mu, alpha, _ = setting1.do_forward_sensitivity(History(train_data[:(train_len-1)]), [train_data[train_len-1]])
    error = 0
    for i in range(1, train_len):
        val = setting1.predict(mu, alpha, History(train_data[:i]))
        error += (val - train_data[i]) ** 2
    print(error)
    intensities_sens = list(map(lambda t: mu.item() + alpha.item() * np.sum(np.exp(-1 * omega * (t - train_data[:train_len+100])[(t - train_data[:train_len+100]) > 0])), train_data[train_len:train_len+100]))
    
    plt.scatter(train_data[:train_len], np.ones((train_len,)))
    # plt.scatter(train_data[train_len:train_len+100], np.ones((train_len+100-train_len,)))
    plt.scatter(pred, np.ones((train_len)))
    # plt.plot(train_data[train_len:train_len+100], intensities, 'r', train_data[train_len:train_len+100], intensities_sens, 'g')
    # plt.plot()
    plt.savefig("oo.png")

    # for i in range(len(train_data[:train_len])):
    #     print(train_data[i], end=", ")
    
    # print()
    # for i in range(len(train_data[:train_len])):
    #     print(intensities[i], end=", ")

    # print()
    # for i in range(len(train_data[:train_len])):
    #     print(intensities_sens[i], end=", ")

    omega = 1
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


    avg = 0
    for j in range(15):
        curr1 = train_data[:30]
        curr2 = train_data[:30]
        error = 0
        for i in range(500):
            val1 = predict(0.2, 0.8, History(curr1[:i+30]))
            val2 = predict(0.2, 0.8, History(curr2[:i+30]))
            error += (val1-val2)**2
            curr1.append(val1)
            curr2.append(val2)
        print('err', np.sqrt(error/len(curr1)))
        avg += np.sqrt(error/len(curr1))

    print(avg/15)
    