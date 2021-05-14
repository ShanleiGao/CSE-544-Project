# %% [markdown]
# # Personal Info
# - Name: Shanlei Gao
# - Student ID: 112516422
# - Email: gao.shanlei@stonybrook.edu
# %% [markdown]
# # Load Data

# %%
import matplotlib.pyplot as plt
import numpy as np
import math
import datetime
import csv
import scipy.stats


# %%
# Get number of days passed since 2020-01-22
def date_index(date_string):
    year = int(date_string[0:4])
    month = int(date_string[5:7])
    day = int(date_string[8:10])
    d1 = datetime.datetime(2020, 1, 22)
    d2 = datetime.datetime(year, month, day)
    return (d2 - d1).days


# %%
de_confirmed = np.loadtxt("Datasets/States Data/5.csv", delimiter=",", encoding='utf-8-sig', usecols=((1)), skiprows=1)
fl_confirmed = np.loadtxt("Datasets/States Data/5.csv", delimiter=",", encoding='utf-8-sig', usecols=((2)), skiprows=1)
de_deaths = np.loadtxt("Datasets/States Data/5.csv", delimiter=",", encoding='utf-8-sig', usecols=((3)), skiprows=1)
fl_deaths = np.loadtxt("Datasets/States Data/5.csv", delimiter=",", encoding='utf-8-sig', usecols=((4)), skiprows=1)

date = []
csv_file=open('Datasets/States Data/5.csv')
csv_reader_lines = csv.reader(csv_file)
for one_line in csv_reader_lines:
    if one_line[0]!='Date':
        date.append(date_index(one_line[0]))


# %%
# Get daily data
de_confirmed_daily = [0]
fl_confirmed_daily = [0]
de_deaths_daily = [0]
fl_deaths_daily = [0]
for i in range(1, len(de_confirmed)):
    de_confirmed_daily.append(de_confirmed[i] - de_confirmed[i-1])
    fl_confirmed_daily.append(fl_confirmed[i] - fl_confirmed[i-1])
    de_deaths_daily.append(de_deaths[i] - de_deaths[i-1])
    fl_deaths_daily.append(fl_deaths[i] - fl_deaths[i-1])

plt.figure(figsize=(12,10))
plt.plot(date, de_confirmed_daily, label='DE confirmed')
plt.plot(date, de_deaths_daily, label='DE deaths')
plt.plot(date, fl_confirmed_daily, label='FL confirmed')
plt.plot(date, fl_deaths_daily, label='FL deaths')
plt.xlabel('Day')
plt.ylabel('Number')
plt.legend(loc='upper left')
plt.show()

# %% [markdown]
# # Clean Dataset

# %%
def tukey_outlier(input, alpha):
    sorted_data = sorted(input)
    n = len(input)
    count = 0

    q1 = sorted_data[math.ceil(n / 4)]
    q3 = sorted_data[math.ceil((3 * n) / 4)]
    iqr = q3 - q1

    for i in range(len(input)):
        if input[i] > q3 + alpha * iqr or input[i] < q1 - alpha * iqr:
            count += 1
            if input[i] != 0: # Except 0s
                input[i] = (input[i-1] + input[i-2])/2

    return count


# %%
outlier_alpha = 1.5
print('DE confirmed:', tukey_outlier(de_confirmed_daily, outlier_alpha), 'outliers detected.')
print('FL confirmed:', tukey_outlier(fl_confirmed_daily, outlier_alpha), 'outliers detected.')
print('DE deaths:', tukey_outlier(de_deaths_daily, outlier_alpha), 'outliers detected.')
print('FL deaths:', tukey_outlier(fl_deaths_daily, outlier_alpha), 'outliers detected.')


# %%
def daily_to_cum(daily):
    sum = 0
    cum = []
    for x in daily:
        sum += x
        cum.append(sum)
    return cum 


# %%
de_confirmed = daily_to_cum(de_confirmed_daily)
de_deaths = daily_to_cum(de_deaths_daily)
fl_confirmed = daily_to_cum(fl_confirmed_daily)
fl_deaths = daily_to_cum(fl_deaths_daily)

# %% [markdown]
# - Result: 
#     - DE confirmed: 41 outliers detected.
#     - FL confirmed: 5 outliers detected.
#     - DE deaths: 24 outliers detected.
#     - FL deaths: 3 outliers detected.
# - For daily data, outliers are removed by replaced with average of the previous and next one.
# - Then sum up daily data to get outlier-removed cumulative data.
# %% [markdown]
# # 2a: Time Series Analysis

# %%
def mse(estimator, truth):
    n = len(estimator)
    sum = 0
    for i in range(n):
        sum += (truth[i] - estimator[i]) ** 2
    return sum/n


# %%
def mape(estimator, truth):
    n = len(estimator)
    sum = 0
    for i in range(n):
        if truth[i] == 0:
            n -= 1
        else:
            sum += abs(truth[i] - estimator[i]) / truth[i]
    return (100 * sum) / n

# %% [markdown]
# ## EWMA

# %%
def ewma(input, alpha):
    n = len(input)
    prediction = [input[0]]
    for i in range(1, n):
        last_observation = input[i-1]
        last_prediction = prediction[-1]
        prediction.append(alpha * last_observation + (1-alpha) * last_prediction)
    return prediction


# %%
def ewma_one_week(data, alpha):
    aug = data[date_index('2020-08-01'):date_index('2020-08-29')]
    true = data[date_index('2020-08-22'):date_index('2020-08-29')]
    predict = ewma(aug, alpha)[-7:]

    print('True value =', true)
    print('Predict value =', predict)
    print('MAPE =', mape(predict, true), '%')
    print('MSE =', mse(predict,true))
    # return predict


# %%
# EWMA with alpha=0.5
print('EWMA with alpha=0.5')
print('\nDE confirmed: ')
ewma_one_week(de_confirmed, 0.5)
print('\nFL confirmed: ')
ewma_one_week(fl_confirmed, 0.5)
print('\nDE deaths: ')
ewma_one_week(de_deaths, 0.5)
print('\nFL deaths: ')
ewma_one_week(fl_deaths, 0.5)


# %%
# EWMA with alpha = 0.7
print('EWMA with alpha = 0.7')
print('\nDE confirmed: ')
ewma_one_week(de_confirmed, 0.7)
print('\nFL confirmed: ')
ewma_one_week(fl_confirmed, 0.7)
print('\nDE deaths: ')
ewma_one_week(de_deaths, 0.7)
print('\nFL deaths: ')
ewma_one_week(fl_deaths, 0.7)

# %% [markdown]
# ## AR

# %%
def ar(data, p):
    y = []
    x = []
    for i in range(len(data)-p):
        x.append([1]+data[i:i+p])
        y.append([data[i+p]])
    
    beta = np.dot(np.transpose(x), x)
    beta = np.linalg.inv(beta)
    beta = np.dot(beta, np.transpose(x))
    beta = np.dot(beta, y)

    feature = [1] + data[-p:]
    result = np.dot(feature, beta)[0]

    return result


# %%
def ar_one_week(data,p):
    aug_first3 = data[date_index('2020-08-01'):date_index('2020-08-22')]
    true = data[date_index('2020-08-22'):date_index('2020-08-29')]
    predict = []

    predict.append(ar(aug_first3, p))
    for i in range(6):
        aug_first3 = np.append(aug_first3, [true[i]])
        predict.append(ar(aug_first3, p))

    print('True value =', true)
    print('Predict value =', predict)
    print('MAPE =', mape(predict, true), '%')
    print('MSE =', mse(predict,true))
    # return predict


# %%
# AR(3)
print('AR(3)')
print('\nDE confirmed: ')
ar_one_week(de_confirmed, 3)
print('\nFL confirmed: ')
ar_one_week(fl_confirmed, 3)
print('\nDE deaths: ')
ar_one_week(de_deaths, 3)
print('\nFL deaths: ')
ar_one_week(fl_deaths, 3)


# %%
# AR(5)
print('AR(5)')
print('\nDE confirmed: ')
ar_one_week(de_confirmed, 5)
print('\nFL confirmed: ')
ar_one_week(fl_confirmed, 5)
print('\nDE deaths: ')
ar_one_week(de_deaths, 5)
print('\nFL deaths: ')
ar_one_week(fl_deaths, 5)

# %% [markdown]
# ## Results
# | MAPE         | AR(3) | AR(5) | EWMA $\alpha=0.5$ | EWMA $\alpha=0.8$ |
# |--------------|-------|-------|-------------------|-------------------|
# | DE Confirmed | 0.28% | 0.32% | 0.53%             | 0.37%             |
# | DE Deaths    | 0.28% | 0.35% | 0.43%             | 0.26%             |
# | FL Confirmed | 0.10% | 0.09% | 1.13%             | 0.79%             |
# | FL Deaths    | 0.60% | 0.52% | 2.29%             | 1.61%             |
# 
# 
# 
# | MSE          | AR(3)     | AR(5)     | EWMA $\alpha=0.5$ | EWMA $\alpha=0.8$ |
# |--------------|-----------|-----------|-------------------|-------------------|
# | DE Confirmed | 2628.73   | 3199.43   | 9385.76           | 5060.40           |
# | DE Deaths    | 2.23      | 4.61      | 5.28              | 3.01              |
# | FL Confirmed | 753619.93 | 552181.23 | 46170181.76       | 22572975.90       |
# | FL Deaths    | 4642.40   | 3767.07   | 55311.68          | 28615.72          |
# %% [markdown]
# # 2b: Hypothesis Testing
# %% [markdown]
# ## Data

# %%
fl_confirmed_daily_feb21 = fl_confirmed_daily[date_index('2021-02-01'):date_index('2021-03-01')]
fl_confirmed_daily_mar21 = fl_confirmed_daily[date_index('2021-03-01'):date_index('2021-04-01')]
fl_deaths_daily_feb21 = fl_deaths_daily[date_index('2021-02-01'):date_index('2021-03-01')]
fl_deaths_daily_mar21 = fl_deaths_daily[date_index('2021-03-01'):date_index('2021-04-01')]

de_confirmed_daily_feb21 = de_confirmed_daily[date_index('2021-02-01'):date_index('2021-03-01')]
de_confirmed_daily_mar21 = de_confirmed_daily[date_index('2021-03-01'):date_index('2021-04-01')]
de_deaths_daily_feb21 = de_deaths_daily[date_index('2021-02-01'):date_index('2021-03-01')]
de_deaths_daily_mar21 = de_deaths_daily[date_index('2021-03-01'):date_index('2021-04-01')]

plt.figure()
plt.plot(range(28), fl_confirmed_daily_feb21, label='Feb 21')
plt.plot(range(31), fl_confirmed_daily_mar21, label='Mar 21')
plt.title('FL Confirmed Daily')
plt.xlabel('Day')
plt.ylabel('Number')
plt.legend(loc='upper right')
plt.show()

plt.figure()
plt.plot(range(28), fl_deaths_daily_feb21, label='Feb 21')
plt.plot(range(31), fl_deaths_daily_mar21, label='Mar 21')
plt.title('FL Deaths Daily')
plt.xlabel('Day')
plt.ylabel('Number')
plt.legend(loc='upper right')
plt.show()

plt.figure()
plt.plot(range(28), de_confirmed_daily_feb21, label='Feb 21')
plt.plot(range(31), de_confirmed_daily_mar21, label='Mar 21')
plt.title('DE Confirmed Daily')
plt.xlabel('Day')
plt.ylabel('Number')
plt.legend(loc='upper right')
plt.show()

plt.figure()
plt.plot(range(28), de_deaths_daily_feb21, label='Feb 21')
plt.plot(range(31), de_deaths_daily_mar21, label='Mar 21')
plt.title('DE Deaths Daily')
plt.xlabel('Day')
plt.ylabel('Number')
plt.legend(loc='upper right')
plt.show()


# %%
def get_corrected_std(data):
    n = len(data)
    sum = 0
    mean = np.mean(data)
    for x in data:
        sum += (x - mean) ** 2
    return np.sqrt(sum/(n-1))


# %%
def get_uncorrected_std(data):
    n = len(data)
    sum = 0
    mean = np.mean(data)
    for x in data:
        sum += (x - mean) ** 2
    return np.sqrt(sum/n)

# %% [markdown]
# ## One Sample Wald's Test

# %%
def walds_test_1(data, theta0, threshold):
    mean = np.mean(data)
    n = len(data)
    w = mean - theta0
    w = w / np.sqrt(mean/n)
    w = abs(w)

    if w > threshold:
        # reject H0: theta = theta0
        print('|W| =', w, '>', threshold, ', thus reject H0, theta != theta0')
    else:
        # accept H0: theta != theta0
        print('|W| =', w, '<=', threshold, ', thus accept H0, theta = theta0')


# %%
print('\nFL Confirmed Daily')
walds_test_1(fl_confirmed_daily_mar21, np.mean(fl_confirmed_daily_feb21), 1.96)

print('\nFL Deaths Daily')
walds_test_1(fl_deaths_daily_mar21, np.mean(fl_deaths_daily_feb21), 1.96)

print('\nDE Confirmed Daily')
walds_test_1(de_confirmed_daily_mar21, np.mean(de_confirmed_daily_feb21), 1.96)

print('\nDE Deaths Daily')
walds_test_1(de_deaths_daily_mar21, np.mean(de_deaths_daily_feb21), 1.96)

# %% [markdown]
# Applicability: 
# - $\hat{\theta}$ is AN, since we use MLE for Wald's test as the estimator.
# - Thus the test is applicable.
# %% [markdown]
# ## One Sample Z-test

# %%
def z_test_1(data, u0, sigma, threshold):
    mean = np.mean(data)
    n = len(data)

    z = mean - u0
    z = z / (sigma / np.sqrt(n))
    z = abs(z)

    if z > threshold:
        # reject H0: mu = mu0
        print('|Z| =', z, '>', threshold, ', thus reject H0, mu != mu0')
    else:
        # accept H0: mu != mu0
        print('|Z| =', z, '<=', threshold, ', thus accept H0, mu = mu0')


# %%
print('\nFL Confirmed Daily')
z_test_1(fl_confirmed_daily_mar21, np.mean(fl_confirmed_daily_feb21), get_corrected_std(fl_confirmed_daily), 1.96)

print('\nFL Deaths Daily')
z_test_1(fl_deaths_daily_mar21, np.mean(fl_deaths_daily_feb21), get_corrected_std(fl_deaths_daily), 1.96)

print('\nDE Confirmed Daily')
z_test_1(de_confirmed_daily_mar21, np.mean(de_confirmed_daily_feb21), get_corrected_std(de_confirmed_daily), 1.96)

print('\nDE Deaths Daily')
z_test_1(de_deaths_daily_mar21, np.mean(de_deaths_daily_feb21), get_corrected_std(de_deaths_daily), 1.96)

# %% [markdown]
# Applicability: 
# - Z-test is applicable when n is large. But in this case n=31, we cannot say it large enough.
# - Thus the test is not applicable.
# %% [markdown]
# ## One Sample T-test

# %%
def t_test_1(data, u0, threshold):
    mean = np.mean(data)
    n = len(data)

    t = mean - u0
    t = t / (get_corrected_std(data) / np.sqrt(n))
    t = abs(t)

    if t > threshold:
        # reject H0: mu = mu0
        print('|T| =', t, '>', threshold, ', thus reject H0, mu != mu0')
    else:
        # accept H0: mu != mu0
        print('|T| =', t, '<=', threshold, ', thus accept H0, mu = mu0')


# %%
# t(30, 0.05/2) ~= 2.04

print('\nFL Confirmed Daily')
t_test_1(fl_confirmed_daily_mar21, np.mean(fl_confirmed_daily_feb21), 2.04)

print('\nFL Deaths Daily')
t_test_1(fl_deaths_daily_mar21, np.mean(fl_deaths_daily_feb21), 2.04)

print('\nDE Confirmed Daily')
t_test_1(de_confirmed_daily_mar21, np.mean(de_confirmed_daily_feb21), 2.04)

print('\nDE Deaths Daily')
t_test_1(de_deaths_daily_mar21, np.mean(de_deaths_daily_feb21), 2.04)

# %% [markdown]
# Applicability:
# - T-test is applicable when data are normally distributed. It is not in this case.
# - Thus this test is not applicable.
# %% [markdown]
# ## Two Sample Wald's Test

# %%
def walds_test_2(x, y, threshold):
    n = len(x)
    m = len(y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    w = x_mean - y_mean
    w = w / np.sqrt(x_mean/n + y_mean/m)
    w = abs(w)

    if w > threshold:
        # reject H0: theta0 = theta1
        print('|W| =', w, '>', threshold, ', thus reject H0, theta0 != theta1')
    else:
        # accept H0: theta0 != theta1
        print('|W| =', w, '<=', threshold, ', thus accept H0, theta0 = theta1')


# %%
print('\nFL Confirmed Daily')
walds_test_2(fl_confirmed_daily_mar21, fl_confirmed_daily_feb21, 1.96)

print('\nFL Deaths Daily')
walds_test_2(fl_deaths_daily_mar21, fl_deaths_daily_feb21, 1.96)

print('\nDE Confirmed Daily')
walds_test_2(de_confirmed_daily_mar21, de_confirmed_daily_feb21, 1.96)

print('\nDE Deaths Daily')
walds_test_2(de_deaths_daily_mar21, de_deaths_daily_feb21, 1.96)

# %% [markdown]
# Applicability:
# - Two populated Wald's test is applicable when two datas are independent. In this case we can say it fits.
# - Thus the test is applicable.
# %% [markdown]
# ## Two Sample Unpaired T-test

# %%
def t_test_2(x, y, threshold):
    n = len(x)
    m = len(y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_s2 = get_corrected_std(x) ** 2
    y_s2 = get_corrected_std(y) ** 2

    t = x_mean - y_mean
    t = t / np.sqrt(x_s2/n + y_s2/m)
    t = abs(t)

    if t > threshold:
        # reject H0: mu0 = mu1
        print('|T| =', t, '>', threshold, ', thus reject H0, mu0 != mu1')
    else:
        # accept H0: mu0 != mu1
        print('|T| =', t, '<=', threshold, ', thus accept H0, mu0 = mu1')


# %%
# t(30, 0.05/2) ~= 2.04

print('\nFL Confirmed Daily')
t_test_2(fl_confirmed_daily_mar21, fl_confirmed_daily_feb21, 2.04)

print('\nFL Deaths Daily')
t_test_2(fl_deaths_daily_mar21, fl_deaths_daily_feb21, 2.04)

print('\nDE Confirmed Daily')
t_test_2(de_confirmed_daily_mar21, de_confirmed_daily_feb21, 2.04)

print('\nDE Deaths Daily')
t_test_2(de_deaths_daily_mar21, de_deaths_daily_feb21, 2.04)

# %% [markdown]
# Applicability:
# - Two-sample unpaired T-test is applicable when 2 datas are normally distributed. It is not in this case.
# - Thus the test is not applicable.
# %% [markdown]
# # 2c: Inference of Equality
# %% [markdown]
# ## Data

# %%
fl_confirmed_daily_3month = fl_confirmed_daily[date_index('2020-10-01'): date_index('2021-01-01')]
fl_deaths_daily_3month = fl_deaths_daily[date_index('2020-10-01'): date_index('2021-01-01')]

de_confirmed_daily_3month = de_confirmed_daily[date_index('2020-10-01'): date_index('2021-01-01')]
de_deaths_daily_3month = de_deaths_daily[date_index('2020-10-01'): date_index('2021-01-01')]

plt.figure()
plt.plot(range(len(de_confirmed_daily_3month)), de_confirmed_daily_3month, label='DE')
plt.plot(range(len(fl_confirmed_daily_3month)), fl_confirmed_daily_3month, label='FL')
plt.title('Confirmed Daily')
plt.xlabel('Day')
plt.ylabel('Number')
plt.legend(loc='upper left')
plt.show()

plt.figure()
plt.plot(range(len(de_deaths_daily_3month)), de_deaths_daily_3month, label='DE')
plt.plot(range(len(fl_deaths_daily_3month)), fl_deaths_daily_3month, label='FL')
plt.title('Deaths Daily')
plt.xlabel('Day')
plt.ylabel('Number')
plt.legend(loc='upper left')
plt.show()

# %% [markdown]
# ## 2-Sample K-S Test

# %%
def get_eCDF(sample, x):
    n = len(sample)
    count = 0
    for d in sample:
        if d <= x:
            count += 1
    return count/n


# %%
def ks_test_2(data1, data2, plt_title, threshold):
    x = []
    for i in range(-1, int(np.max(data1+data2)) * 10 + 10):
        x.append(i/10)
    
    y1 = []
    y2 = []
    max_dif = 0
    max_dif_x = 0
    max_dif_y1 = 0
    max_dif_y2 = 0

    for xi in x:
        eCDF_1 = get_eCDF(data1, xi)
        eCDF_2 = get_eCDF(data2, xi)
        dif = abs(eCDF_1 - eCDF_2)
        if (dif > max_dif):
            max_dif = dif
            max_dif_x = xi
            max_dif_y1 = eCDF_1
            max_dif_y2 = eCDF_2
        y1.append(eCDF_1)
        y2.append(eCDF_2)

    plt.figure(figsize=(12,10))
    plt.plot(x, y1, label='DE')
    plt.plot(x, y2, label='FL')
    plt.plot([max_dif_x, max_dif_x], [max_dif_y1, max_dif_y2], label='Max Difference='+str(max_dif)+' at x='+str(max_dif_x))
    plt.xlabel('Day')
    plt.ylabel('eCDF')
    plt.title(plt_title)
    plt.legend()
    plt.show()

    if max_dif > threshold:
        # reject H0: F_x != F_y
        print('d =', max_dif, '>', threshold, ', thus reject H0, F_x != F_y\n')
    else:
        # accept H0: F_x = F_y
        print('d =', max_dif, '<=', threshold, ', thus accept H0, F_x = F_y\n')


# %%
ks_test_2(de_confirmed_daily_3month, fl_confirmed_daily_3month, 'Confirmed Daily', 0.05)
ks_test_2(de_deaths_daily_3month, fl_deaths_daily_3month, 'Deaths Daily', 0.05)

# %% [markdown]
# ## 1-Sample K-S Test: Poisson
# - $\hat \lambda_{MME}  = \bar{X}$

# %%
def ks_test_poisson(data, lam, plt_title, threshold):
    x = []
    for i in range(-1, int(np.max(data)) * 10 + 10):
        x.append(i/10)
    
    y1 = []
    y2 = []
    max_dif = 0
    max_dif_x = 0
    max_dif_y1 = 0
    max_dif_y2 = 0

    for xi in x:
        eCDF_1 = get_eCDF(data, xi)
        eCDF_2 = scipy.stats.poisson.cdf(xi, lam)
        dif = abs(eCDF_1 - eCDF_2)
        if (dif > max_dif):
            max_dif = dif
            max_dif_x = xi
            max_dif_y1 = eCDF_1
            max_dif_y2 = eCDF_2
        y1.append(eCDF_1)
        y2.append(eCDF_2)

    plt.figure(figsize=(12,10))
    plt.plot(x, y1, label='FL')
    plt.plot(x, y2, label='Poisson')
    plt.plot([max_dif_x, max_dif_x], [max_dif_y1, max_dif_y2], label='Max Difference='+str(max_dif)+' at x='+str(max_dif_x))
    plt.xlabel('Day')
    plt.ylabel('eCDF')
    plt.title(plt_title)
    plt.legend()
    plt.show()

    if max_dif > threshold:
        # reject H0: F_x != F_y
        print('d =', max_dif, '>', threshold, ', thus reject H0, F_x != Poisson(' + str(lam) + ')\n')
    else:
        # accept H0: F_x = F_y
        print('d =', max_dif, '<=', threshold, ', thus accept H0, F_x = Poisson(' + str(lam) + ')\n')


# %%
ks_test_poisson(fl_confirmed_daily_3month, np.mean(de_confirmed_daily_3month), 'Confirmed Daily', 0.05)
ks_test_poisson(fl_deaths_daily_3month, np.mean(de_deaths_daily_3month), 'Deaths Daily', 0.05)

# %% [markdown]
# ## 1-Sample K-S Test: Geometric
# - $\hat{p}_{MME} = \frac{1}{\bar{X}}$

# %%
def ks_test_geometric(data, p, plt_title, threshold):
    x = []
    for i in range(-1, int(np.max(data)) * 10 + 10):
        x.append(i/10)
    
    y1 = []
    y2 = []
    max_dif = 0
    max_dif_x = 0
    max_dif_y1 = 0
    max_dif_y2 = 0

    for xi in x:
        eCDF_1 = get_eCDF(data, xi)
        eCDF_2 = scipy.stats.geom.cdf(xi, p)
        dif = abs(eCDF_1 - eCDF_2)
        if (dif > max_dif):
            max_dif = dif
            max_dif_x = xi
            max_dif_y1 = eCDF_1
            max_dif_y2 = eCDF_2
        y1.append(eCDF_1)
        y2.append(eCDF_2)

    plt.figure(figsize=(12,10))
    plt.plot(x, y1, label='FL')
    plt.plot(x, y2, label='Geometric')
    plt.plot([max_dif_x, max_dif_x], [max_dif_y1, max_dif_y2], label='Max Difference='+str(max_dif)+' at x='+str(max_dif_x))
    plt.xlabel('Day')
    plt.ylabel('eCDF')
    plt.title(plt_title)
    plt.legend()
    plt.show()

    if max_dif > threshold:
        # reject H0: F_x != F_y
        print('d =', max_dif, '>', threshold, ', thus reject H0, F_x != Geometric(' + str(p) + ')\n')
    else:
        # accept H0: F_x = F_y
        print('d =', max_dif, '<=', threshold, ', thus accept H0, F_x = Geometric(' + str(p) + ')\n')


# %%
ks_test_geometric(fl_confirmed_daily_3month, 1/np.mean(de_confirmed_daily_3month), 'Confirmed Daily', 0.05)
ks_test_geometric(fl_deaths_daily_3month, 1/np.mean(de_deaths_daily_3month), 'Deaths Daily', 0.05)

# %% [markdown]
# ## 1-sample K-S Test: Binomial
# - $\hat p_{MME} = 1 - \frac{s^2}{\bar{X}}$
# - $\hat n_{MME} = \frac{\bar{X}^2}{\bar{X} - s^2} = \frac{\bar{X}}{p_{MME}}$

# %%
def ks_test_binomial(data, data2, plt_title, threshold):
    x_bar = np.mean(data2)
    s2 = np.var(data2)
    n = (x_bar ** 2) / (x_bar - s2)
    p = 1 - (s2 / x_bar)

    x = []
    for i in range(int(np.max(data)) * 10 + 10):
        x.append(i/10)
    
    y1 = []
    y2 = []
    max_dif = 0
    max_dif_x = 0
    max_dif_y1 = 0
    max_dif_y2 = 0

    for xi in x:
        eCDF_1 = get_eCDF(data, xi)
        eCDF_2 = scipy.stats.binom.cdf(xi, n, p)
        dif = abs(eCDF_1 - eCDF_2)
        if (dif > max_dif):
            max_dif = dif
            max_dif_x = xi
            max_dif_y1 = eCDF_1
            max_dif_y2 = eCDF_2
        y1.append(eCDF_1)
        y2.append(eCDF_2)

    plt.figure(figsize=(12,10))
    plt.plot(x, y1, label='FL')
    plt.plot(x, y2, label='Geometric')
    plt.plot([max_dif_x, max_dif_x], [max_dif_y1, max_dif_y2], label='Max Difference='+str(max_dif)+' at x='+str(max_dif_x))
    plt.xlabel('Day')
    plt.ylabel('eCDF')
    plt.title(plt_title)
    plt.legend()
    plt.show()

    if max_dif > threshold:
        # reject H0: F_x != F_y
        print('d =', max_dif, '>', threshold, ', thus reject H0, F_x != Binomial(' + str(n) + ',' + str(p) + ')\n')
    else:
        # accept H0: F_x = F_y
        print('d =', max_dif, '<=', threshold, ', thus accept H0, F_x = Binomial(' + str(n) + ',' + str(p) + ')\n')


# %%
ks_test_binomial(fl_confirmed_daily_3month, de_confirmed_daily_3month, 'Confirmed Daily', 0.05)
ks_test_binomial(fl_deaths_daily_3month, de_deaths_daily_3month, 'Deaths Daily', 0.05)

# %% [markdown]
# ## Permutation Test

# %%
def get_permutation(data0, data1):
    data = data0 + data1
    data = np.random.permutation(data)
    n = len(data0)
    x = data[:n]
    y = data[n:]
    return (x,y)


# %%
def permutation_test(a1, a2, s, threshold):
    t_obs = abs(np.mean(a1) - np.mean(a2))
    count = 0
    for i in range(s):
        (x, y) = get_permutation(a1, a2)
        t = abs(np.mean(x) - np.mean(y))
        if t > t_obs:
            count += 1
    p_value = count/s
    if p_value <= threshold:
        # reject H0: F_x != F_y
        print('p-value =', p_value, '<=', threshold, ', thus reject H0, F_x != F_y')
    else:
        # accept H0: F_x = F_y
        print('p-value =', p_value, '>', threshold, ', thus accept H0, F_x = F_y')


# %%
print('\nConfirmd Daily')
permutation_test(fl_confirmed_daily_3month, de_confirmed_daily_3month, 1000, 0.05)
print('\nDeaths Daily')
permutation_test(fl_deaths_daily_3month, de_deaths_daily_3month, 1000, 0.05)

# %% [markdown]
# # 2d: Bayesian Inference
# - We can treat $Exp(\lambda)$ as $Gamma(1, \lambda)$
# - The Poisson and Gamma families of distributions are a conjugate pair
# - Thus, posterior of $\lambda \sim Gamma(\alpha^*, \beta^*)$, where $\alpha^* = \alpha + \sum X_i, \beta^* = \frac{\beta}{1+n\beta} = \frac{\bar X}{1+n\bar X}$

# %%
def plot_gamma(alpha, beta, plt_label):
    x = np.linspace(scipy.stats.gamma.ppf(0.01, alpha, scale=beta),scipy.stats.gamma.ppf(0.99, alpha, scale=beta), 100)
    y = scipy.stats.gamma.pdf(x, alpha, scale=beta)
    map = (alpha - 1) * beta
    plt.plot(x,y, label=plt_label+', MAP='+ str(map))


# %%
def beyasian_inference(data0, data1, plt_title):
    alpha = 1
    beta = np.mean(data0)

    plt.figure(figsize=(12,10))
    # plot_gamma(alpha, beta, 'Prior: Week 1-4')
    for i in range(len(data1)):
        data0 = np.append(data0, data1[i])

        sum = np.sum(data0)
        n = len(data0)
        mean = sum / n

        alpha = 1 + sum
        beta = mean / (1 + n*mean)
        plot_gamma(alpha, beta, 'Posterior: Week '+str(5+i))
    plt.xlabel('Number')
    plt.ylabel('PDF of Posterior')
    plt.title(plt_title)
    plt.legend(loc='upper right')
    plt.show()


# %%
confirmed_week14 = np.add(de_confirmed_daily[date_index('2020-06-01'): date_index('2020-06-29')], fl_confirmed_daily[date_index('2020-06-01'): date_index('2020-06-29')])
confirmed_week58 = []
for i in range(4):
    confirmed_week58.append(np.add(de_confirmed_daily[date_index('2020-06-29')+7*i: date_index('2020-07-06')+7*i], fl_confirmed_daily[date_index('2020-06-29')+7*i: date_index('2020-07-06')+7*i]))

deaths_week14 = np.add(de_deaths_daily[date_index('2020-06-01'): date_index('2020-06-29')], fl_deaths_daily[date_index('2020-06-01'): date_index('2020-06-29')])
deaths_week58 = []
for i in range(4):
    deaths_week58.append(np.add(de_deaths_daily[date_index('2020-06-29')+7*i: date_index('2020-07-06')+7*i], fl_deaths_daily[date_index('2020-06-29')+7*i: date_index('2020-07-06')+7*i]))


# %%
beyasian_inference(confirmed_week14, confirmed_week58, 'Confirmed Daily')
beyasian_inference(deaths_week14, deaths_week58, 'Deaths Daily')

# %% [markdown]
# # 3a: Does X-data Impacted

# %%
def date_index_slash(date_string):
    date = date_string.split('/')
    month = int(date[0])
    day = int(date[1])
    year = int(date[2])
    d1 = datetime.datetime(2020, 1, 22)
    d2 = datetime.datetime(year, month, day)
    return (d2 - d1).days

# Since the stock market closed for weekends and vacations, we need to fill those days with latest available price.
def fill_stock(date, price_without_weekends):
    price = []
    latest_date = -1
    for i in range(len(date)):
        d = date[i]
        for j in range(d - latest_date):
            price.append(price_without_weekends[i])
        latest_date = d
    return price


# %%
zoom_price_without_weekends = []
stock_date = []
csv_file=open('Datasets/X-dataset/Zoom Price.csv')
csv_reader_lines = csv.reader(csv_file)
for one_line in csv_reader_lines:
    if one_line[0]!='Date':
        stock_date.append(date_index_slash(one_line[0]))
        zoom_price_without_weekends.append(float(one_line[4]))
        
zoom_price = fill_stock(stock_date, zoom_price_without_weekends)
zoom_price.append(zoom_price[-1])
zoom_price.append(zoom_price[-1])


# %%
plt.figure(figsize=(12,10))
plt.plot(range(len(zoom_price)), zoom_price)
plt.xlabel('Day')
plt.ylabel('Price')
plt.show()


# %%
us_confirmed = [0]*(date_index('2021-04-03') - date_index('2020-01-22') + 1)
csv_file=open('Datasets/US-all/US_confirmed.csv')
csv_reader_lines = csv.reader(csv_file)
for one_line in csv_reader_lines:
    if one_line[0]!='State':
        for i in range(1, len(one_line)):
            us_confirmed[i-1] += int(one_line[i])

us_deaths = [0]*(date_index('2021-04-03') - date_index('2020-01-22') + 1)
csv_file=open('Datasets/US-all/US_deaths.csv')
csv_reader_lines = csv.reader(csv_file)
for one_line in csv_reader_lines:
    if one_line[0]!='State':
        for i in range(1, len(one_line)):
            us_deaths[i-1] += int(one_line[i])

us_confirmed_daily = [0]
us_deaths_daily = [0]
for i in range(1, len(us_confirmed)):
    us_confirmed_daily.append(us_confirmed[i] - us_confirmed[i-1])
    us_deaths_daily.append(us_deaths[i] - us_deaths[i-1])

plt.figure(figsize=(12,10))
plt.plot(range(len(us_confirmed_daily)), us_confirmed_daily)
plt.title('US Confirmed')
plt.xlabel('Day')
plt.ylabel('Confirmed Daily')
plt.show()

plt.figure(figsize=(12,10))
plt.plot(range(len(us_deaths_daily)), us_deaths_daily)
plt.title('US Deaths')
plt.xlabel('Day')
plt.ylabel('Deaths Daily')
plt.show()


# %%
def pearsons_correlation_coefficient(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    sum0 = 0
    sum1 = 0
    sum2 = 0
    for i in range(len(x)):
        sum0 += (x[i] - x_mean) * (y[i] - y_mean)
        sum1 += (x[i] - x_mean) ** 2
        sum2 += (y[i] - y_mean) ** 2
    pho = sum0 / (np.sqrt(sum1 * sum2))
    return pho


# %%
def pearsons_test(x, y, threshold):
    pho = pearsons_correlation_coefficient(x,y)

    if abs(pho) > threshold:
        # reject H0
        if pho > 0.5:
            print('Pho =', pho, '>', threshold, ', thus reject H0, X and Y are positive linear correlated.')
        else:
            print('Pho =', pho, '<', -threshold, ', thus reject H0, X and Y are positive linear correlated.')
    else:
        # accept H1
        print('|Pho| =', pho, '<=', threshold, ', thus accept H0, X and Y are not linear correlated.')


# %%
print('\nUS Confirmed Daily')
pearsons_test(zoom_price, us_confirmed_daily, 0.5)
print('\nUS Deaths Daily')
pearsons_test(zoom_price, us_deaths_daily, 0.5)

# %% [markdown]
# - H0: Zoom stock price and US #confirmed/deaths are not linear correlated.
# - H1: Zoom stock price and US #confirmed/deaths are linear correlated.
# - We use a threshold of 0.5 here.
# - Results: we can say zoom stock price are positive linear correlated with US #confirmed but not linear correlated with US #deaths.
# %% [markdown]
# # 3b. If COVID19 Data Changed after Local Events
# - Here we chose two events:
#     - Lockdown, started since April 2020
#     - Vaccine, started since Feb 2021

# %%
def walds_test_2_split(data, split_date, threshold):
    split = date_index(split_date)
    data0 = data[:split]
    data1 = data[split:]

    walds_test_2(data0, data1, 1.96)


# %%
print('\nUS Confirmed Daily')
walds_test_2_split(us_confirmed_daily, '2020-04-01', 1.96)

print('\nUS Deaths Daily')
walds_test_2_split(us_deaths_daily, '2020-04-01', 1.96)


# %%
print('\nUS Confirmed Daily')
walds_test_2_split(us_confirmed_daily, '2021-02-01', 1.96)

print('\nUS Deaths Daily')
walds_test_2_split(us_deaths_daily, '2021-02-01', 1.96)

# %% [markdown]
# - H0: mean of US #confirmed/deaths before the event = mean of US #confirmed/deaths after the event
# - H1: mean of US #confirmed/deaths before the event != mean of US #confirmed/deaths after the event
# - We chose 2-populated Wald's test.
# - The result is that means are different as lockdown and vaccine started, which means they are useful.

