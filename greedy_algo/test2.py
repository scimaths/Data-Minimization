import matplotlib.pyplot as plt
from nll import * 

if __name__ == '__main__':

    train_len = 1500
    test_len = 10
    train_data = None
    with open('data_exp16_train.npy', 'rb') as f:
        train_data = list(np.load(f))

    test_data = None
    with open('data_exp16_test.npy', 'rb') as f:
        test_data = list(np.load(f))


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


    train_data = [0]
    for i in range(1, train_len):
        val = predict(0.2, 0.8, History(train_data[:i]))
        train_data.append(val)

    avg = 0
    for j in range(30):
        curr1 = train_data[:30]
        curr2 = train_data[:30]
        error = 0
        for i in range(1000):
            val1 = predict(0.2, 0.8, History(curr1[:i+30]))
            val2 = predict(0.2, 0.8, History(curr2[:i+30]))
            error += (val1-val2)**2
            curr1.append(val1)
            curr2.append(val2)
        print('err', np.sqrt(error/len(curr1)))
        avg += np.sqrt(error/len(curr1))

    print(avg/30)



    omega = 1
    final_T = train_data[-1]
    setting1 = Setting1(None, None, None, final_T, omega, init_num_epochs=300, lr=5e-3, epoch_decay=None, budget=None, sensitivity=0.1)
    mu, alpha, _ = setting1.do_forward(History(train_data[:(train_len-1)]), [train_data[train_len-1]])
    
    print(mu, alpha)
    avg = 0
    for j in range(30):
        curr1 = train_data[:30]
        curr2 = train_data[:30]
        error = 0
        for i in range(1000):
            val1 = predict(mu.item(), alpha.item(), History(curr1[:i+30]))
            val2 = predict(0.2, 0.8, History(curr2[:i+30]))
            error += (val1-val2)**2
            curr1.append(val1)
            curr2.append(val2)
        print('err', np.sqrt(error/len(curr1)))
        avg += np.sqrt(error/len(curr1))

    print(avg/30)
    



    avg = 0
    for j in range(30):
        curr1 = [0]
        curr2 = [0]
        error = 0
        for i in range(1, 1000):
            val1 = predict(0.2, 0.8, History(curr1[:i]))
            val2 = predict(0.2, 0.8, History(curr2[:i]))
            error += (val1-val2)**2
            curr1.append(val1)
            curr2.append(val2)
        print('err', np.sqrt(error/len(curr1)))
        avg += np.sqrt(error/len(curr1))

    print(avg/30)



    omega = 1
    final_T = train_data[-1]
    setting1 = Setting1(None, None, None, final_T, omega, init_num_epochs=300, lr=5e-3, epoch_decay=None, budget=None, sensitivity=0.1)
    mu, alpha, _ = setting1.do_forward(History(train_data[:(train_len-1)]), [train_data[train_len-1]])
    
    print(mu, alpha)
    avg = 0
    for j in range(30):
        curr1 = [0]
        curr2 = [0]
        error = 0
        for i in range(1,1000):
            val1 = predict(mu.item(), alpha.item(), History(curr1[:i]))
            val2 = predict(0.2, 0.8, History(curr2[:i]))
            error += (val1-val2)**2
            curr1.append(val1)
            curr2.append(val2)
        print('err', np.sqrt(error/len(curr1)))
        avg += np.sqrt(error/len(curr1))

    print(avg/30)
    