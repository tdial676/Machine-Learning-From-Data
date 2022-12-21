import numpy
import random
from sklearn import svm
train = numpy.loadtxt('./features.train')
test = numpy.loadtxt('./features.test')

def one_v_one(data, num1, num2):
    new_data = []
    for thing in data:
        if thing[0] != num1 and thing[0] != num2:
            continue
        else:
            new_data.append(thing)
    return numpy.array(new_data)
otrain = one_v_one(train, 1, 5)
otest = one_v_one(test, 1, 5)
digits_train = otrain[:, 0]
X_train = otrain[:, 1:]
digits_test = otest[:, 0]
X_test = otest[:, 1:]

# digits_train = train[:, 0]
# X_train = train[:, 1:]
# digits_test = test[:, 0]
# X_test = test[:, 1:]

def make_binary(digit_class, digits):
    scores = []
    for score in digits:
        if (score != digit_class):
            scores.append(-1)
        else:
            scores.append(1)
    return numpy.array(scores)
def transform(x1, x2):
    return [1, x1, x2, x1 * x2, x1 ** 2, x2 ** 2]

def get_weight_reg(k, data, binary_train, Transform):
    i = 2 if (Transform == False) else 6
    Z = numpy.array(data)
    dagger = (numpy.dot(Z.T, Z) + numpy.dot(numpy.identity(i), k))
    w =  numpy.dot(numpy.dot(numpy.linalg.inv(dagger), Z.T), binary_train)
    return w

def Ein_lin(d_classifier, k, Transform):
    summ_err = 0
    binary_train = make_binary(d_classifier, digits_train)
    if (Transform == False):
        Z = X_train
    else:
        Z = []
        for point1 in X_train:
            Z.append(transform(point1[0], point1[1]))
    w = get_weight_reg(k, Z, binary_train, Transform)
    for d,s in zip(Z, binary_train):
        if (s != numpy.sign(numpy.dot(w, d))):
            summ_err += 1
    return summ_err/len(Z)

def Eout_lin(d_classifier, k, Transform):
    binary_test = make_binary(d_classifier, digits_test)
    binary_train = make_binary(d_classifier, digits_train)
    summ_err= 0
    if (Transform == False):
        Z_Test = X_test
        Z_Train = X_train
    else:
        Z_Train = []
        Z_Test = []
        for point1 in X_train:
            Z_Train.append(transform(point1[0], point1[1]))
        for point1 in X_test:
            Z_Test.append(transform(point1[0], point1[1]))
    w = get_weight_reg(k, Z_Train, binary_train, Transform)
    for d, s in zip(Z_Test, binary_test):
        if (s != numpy.sign(numpy.dot(w, d))):
            summ_err += 1
    return summ_err / len(Z_Test)

def q7():
    print("5 versus all our Ein is: ", Ein_lin(5, 1, False))
    print("6 versus all our Ein is: ", Ein_lin(6, 1, False))
    print("7 versus all our Ein is: ", Ein_lin(7, 1, False))
    print("8 versus all our Ein is: ", Ein_lin(8, 1, False))
    print("9 versus all our Ein is: ", Ein_lin(9, 1, False))
def q8():
    print("0 versus all our Eout is: ", Eout_lin(0, 1, True))
    print("1 versus all our Eout is: ", Eout_lin(1, 1, True))
    print("2 versus all our Eout is: ", Eout_lin(2, 1, True))
    print("3 versus all our Eout is: ", Eout_lin(3, 1, True))
    print("4 versus all our Eout is: ", Eout_lin(4, 1, True))
def q9():
    print('Num:     (Ein, Eout) With Transformation:                        (Ein, Eout) Without Transformation:      ')
    print('0  :     (',Ein_lin(0, 1, True),' , ', Eout_lin(0, 1, True),')  |     (',Ein_lin(0, 1, False),' , ', Eout_lin(0, 1, False))
    print('1  :     (',Ein_lin(1, 1, True),' , ', Eout_lin(1, 1, True),') |     (',Ein_lin(1, 1, False),'   , ',Eout_lin(1, 1, False))
    print('2  :     (',Ein_lin(2, 1, True),' , ', Eout_lin(2, 1, True),')  |     (',Ein_lin(2, 1, False),' , ', Eout_lin(2, 1, False))
    print('3  :     (',Ein_lin(3, 1, True),' , ', Eout_lin(3, 1, True),')  |     (',Ein_lin(3, 1, False),' , ', Eout_lin(3, 1, False))
    print('4  :     (',Ein_lin(4, 1, True),' , ', Eout_lin(4, 1, True),')  |     (',Ein_lin(4, 1, False),' , ', Eout_lin(4, 1, False))
    print('5  :     (',Ein_lin(5, 1, True),' , ', Eout_lin(5, 1, True),')  |     (',Ein_lin(5, 1, False),' , ', Eout_lin(5, 1, False))
    print('6  :     (',Ein_lin(6, 1, True),' , ', Eout_lin(6, 1, True),')  |     (',Ein_lin(6, 1, False),' , ', Eout_lin(6, 1, False))
    print('7  :     (',Ein_lin(7, 1, True),' , ', Eout_lin(7, 1, True),')  |     (',Ein_lin(7, 1, False),' , ', Eout_lin(7, 1, False))
    print('8  :     (',Ein_lin(8, 1, True),' , ', Eout_lin(8, 1, True),')  |     (',Ein_lin(8, 1, False),' , ', Eout_lin(8, 1, False))
    print('9  :     (',Ein_lin(9, 1, True),' , ', Eout_lin(9, 1, True),')  |     (',Ein_lin(9, 1, False),' , ', Eout_lin(9, 1, False))
    print(abs(Eout_lin(5, 1, True) - Eout_lin(5, 1, False))/ Eout_lin(5, 1, True) * 100)
    print(Eout_lin(5, 1, False) * 0.95)
def q10():
    print('lamda = 0.01 (Ein, Eout): ', Ein_lin(1, 0.01, True),',', Eout_lin(1, 0.01, True))
    print('lamda = 1 (Ein, Eout): ', Ein_lin(1, 1, True),',', Eout_lin(1, 1, True))
def q11():
    def z(x1, x2):
        return [1, (x2 ** 2) - (2 * x1) - 1, (x1 ** 2) - (2 * x2) + 1]
    X = [[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]]
    Y = [-1, -1, -1, 1, 1, 1, 1]
    Z = []
    for x1,x2 in X:
        Z.append(z(x1, x2))
    print(Z)
    classifier = svm.SVC(kernel='linear', C=numpy.inf)
    classifier.fit(Z, Y)
    print("Ein: ", 1 - classifier.score(Z, Y))
    print("Weight: ", classifier.coef_[0, 1:], 'Bias: ', classifier.intercept_)
    print('Normalized Weights: ',numpy.round(classifier.coef_[0, 1:]/2), 'Normalized Bias: ', (classifier.intercept_/2))

def q12():
    X = [[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]]
    Y = [-1, -1, -1, 1, 1, 1, 1]

    classifier = svm.SVC(kernel='poly', degree=2, coef0=1, C=numpy.inf)
    classifier.fit(X, Y)
    print("Number Of Support Vectors: ", sum(classifier.n_support_))
q11()
def q13(runs, points):
    def f(x1, x2):
        return numpy.sign(x2 - x1 + (0.25 * numpy.sin(numpy.pi * x1)))
    summ_fails = 0
    for r in range(runs):
        data = []
        score = []
        for i in range(points):
            point = [1, random.uniform(-1, 1), random.uniform(-1, 1)]
            data.append(point)
            score.append(f(point[1], point[2]))
        classifier = svm.SVC(kernel='rbf', C=numpy.inf, coef0=1, gamma=1.5)
        classifier.fit(data, score)
        err = 1 - classifier.score(data, score)
        summ_fails += 0 if (err == 0) else 1
    print("Non seperable this many times: ",summ_fails/runs)

def f(x1, x2):
    return numpy.sign(x2 - x1 + (0.25 * numpy.sin(numpy.pi * x1)))

def make_points(num, data):
    points = []
    for i in range(num):
        points.append(random.choice(data))
    return numpy.array(points)

def create_data(points):
    data = []
    score = []
    for i in range(points):
        point = [random.uniform(-1, 1), random.uniform(-1, 1)]
        data.append(point)
        score.append(f(point[0], point[1]))
    return [data, score]

def find_centers(data, K):
    centers = make_points(K, data)
    while True:
        clusters = [[] for j in range(K)]
        for q in data:
            curr_min = numpy.inf
            curr_min_idx = 0
            for i,p in enumerate(centers):
                distance = numpy.sqrt(sum((q - p) ** 2))
                if distance < curr_min:
                    curr_min_idx = i
                    curr_min = distance
            clusters[curr_min_idx].append(q)
        empty = 0
        for c in clusters:
            if len(c) == 0:
                empty += 1
        if (empty == 0):
            break
        else:
            centers = make_points(K, data)
    curr_idx = 0
    for c in clusters:
        curr_x_mean = 0
        curr_y_mean = 0
        for point in c:
            curr_x_mean += point[0]
            curr_y_mean += point[1]
        curr_x_mean /= len(c)
        curr_y_mean /= len(c)
        centers[curr_idx] = [curr_x_mean, curr_y_mean]
        curr_idx += 1
    return centers

def exp(gamma, x, c):
    return numpy.exp(-gamma * numpy.sum((x - c) ** 2))

def make_phi_matrix(data, K, gamma):
    phi_matrix = []
    centers = find_centers(data, K)
    for d in data:
        curr_point = [1]
        for c in centers:
            curr_point.append(exp(gamma, d, c))
        phi_matrix.append(curr_point)
    return [numpy.array(phi_matrix), centers]

def get_weight(data, K, gamma, Y):
    phi = make_phi_matrix(data, K, gamma)
    return [numpy.dot(numpy.linalg.pinv(phi[0]), Y), phi[1]]

def score_RBF(x, K, gamma, centers, w):
    tot = 0
    for i in range(1, K):
        tot += (w[i] * exp(gamma, x, centers[i]))
    return numpy.sign(tot + w[0])

def Eout_RBF(data, score, K, gamma, centers, w):
    summ_err = 0
    for d, s in zip(data, score):
        if (s != score_RBF(d, K, gamma, centers, w)):
            summ_err += 1
    return summ_err/(len(data) * K)

def Kernel_VS_Reg(K, gamma, runs):
    better = 0
    for r in range(runs):
        train_p = create_data(100)
        test_p = create_data(1000)
        d_train = train_p[0]
        s_train = train_p[1]
        d_test = test_p[0]
        s_test = test_p[1]
        #SVM
        classifier = svm.SVC(kernel='rbf', C=numpy.inf, coef0=1.0, gamma=1.5)
        classifier.fit(d_train, s_train)
        while 1 - classifier.score(d_train, s_train) != 0.0:
            train_p = create_data(100)
            d_train = train_p[0]
            s_train = train_p[1]
            classifier.fit(d_train, s_train)
        summ_err_k = (1 - classifier.score(d_test, s_test))

        #RBF
        w, centers = get_weight(d_train, K, gamma, s_train)
        summ_err_r = Eout_RBF(d_test, s_test, K, gamma, centers, w)

        if (summ_err_k < summ_err_r):
            better += 1
    return better/runs

def Reg_Ein_and_Eout(K, gamma, runs):
    summ_Ein = 0
    summ_Eout = 0
    for r in range(runs):
        train_p = create_data(100)
        test_p = create_data(1000)
        d_train = train_p[0]
        s_train = train_p[1]
        d_test = test_p[0]
        s_test = test_p[1]

        w, centers = get_weight(d_train, K, gamma, s_train)
        summ_Eout += Eout_RBF(d_test, s_test, K, gamma, centers, w)
        summ_Ein +=  Eout_RBF(d_train, s_train, K, gamma, centers, w)
    return (summ_Ein/runs, summ_Eout/runs)

def q14(gamma, K, runs):
    av = Kernel_VS_Reg(K, gamma, runs)
    print("Kernel Wins this many times: ", av)

def q15():
    q14(1.5, 12, 1000)

#IGNORE
# def q16(gamma, K, runs):
#     Ein, Eout = Reg_Ein_and_Eout(K, gamma, runs)
#     print("The average Ein for K=",K," is : ", Ein)
#     print("The average Eout for K=",K," is : ", Eout)

def q16v2(gamma, K1, K2, runs):
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    for i in range(runs):
        Ein1, Eout1 = Reg_Ein_and_Eout(K1, gamma, 1)
        Ein2, Eout2 = Reg_Ein_and_Eout(K2, gamma, 1)
        if (Ein2 < Ein1 and Eout2 > Eout1):
            a += 1
        if (Ein2 > Ein1 and Eout2 < Eout1):
            b += 1
        if (Ein2 > Ein1 and Eout2 > Eout1):
            c += 1
        if (Ein2 < Ein1 and Eout2 < Eout1):
            d += 1
        if (Ein2 == Ein1 and Eout2 == Eout1):
            e += 1
    print("Choice a: ",a)
    print("Choice b: ",b)
    print("Choice c: ",c)
    print("Choice d: ",d)
    print("Choice e: ",e)

#IGNORE
# def q17(gamma, K, runs):
#     Ein, Eout = Reg_Ein_and_Eout(K, gamma, runs)
#     print("The average Ein for gamma=",gamma," is : ", Ein)
#     print("The average Eout for gamma=",gamma," is : ", Eout)

def q17v2(gamma1, gamma2, K, runs):
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    for i in range(runs):
        Ein1, Eout1 = Reg_Ein_and_Eout(K, gamma1, 1)
        Ein2, Eout2 = Reg_Ein_and_Eout(K, gamma2, 1)
        if (Ein2 < Ein1 and Eout2 > Eout1):
            a += 1
        if (Ein2 > Ein1 and Eout2 < Eout1):
            b += 1
        if (Ein2 > Ein1 and Eout2 > Eout1):
            c += 1
        if (Ein2 < Ein1 and Eout2 < Eout1):
            d += 1
        if (Ein2 == Ein1 and Eout2 == Eout1):
            e += 1

    print("Choice a: ",a)
    print("Choice b: ",b)
    print("Choice c: ",c)
    print("Choice d: ",d)
    print("Choice e: ",e)

def zero_Ein(gamma, K, runs):
    Ein_zero = 0
    for r in range(runs):
        train_p = create_data(100)
        d_train = train_p[0]
        s_train = train_p[1]

        w, centers = get_weight(d_train, K, gamma, s_train)
        if Eout_RBF(d_train, s_train, K, gamma, centers, w) == 0:
            Ein_zero += 1
    return Ein_zero/runs

def q18(gamma, K, runs):
    av_Ein_zero = zero_Ein(gamma, K, runs)
    print("The average times Ein is zero is: ", av_Ein_zero)
