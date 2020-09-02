import numpy as np
import os
import xgboost as xgb
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import random
import csv


with open('ratings.csv', 'r') as f:
    ratings = pd.read_csv(
        f, dtype={"emp": int, "comp": str, "Date": str, "rating": int})

with open('train.csv', 'r') as f:
    train = pd.read_csv(
        f, dtype={"id": int, "emp": int, "comp": str, "lastratingdate": str, "left": int})

with open('test.csv', 'r') as f:
    test = pd.read_csv(
        f, dtype={"id": int, "emp": int, "comp": str, "lastratingdate": str})

with open('remarks.csv', 'r') as f:
    remarks = pd.read_csv(
        f, dtype={"emp": int, "comp": str, "remarkId": str, "txt": str, "remarkDate": str})

companies = pd.unique(ratings['comp'])

with open('remarks_supp_opp.csv', 'r') as f:
    remarks_supp_opp = pd.read_csv(f, skipfooter=1, dtype={
                                   "emp": int, "comp": str, "support": bool, "oppose": bool, "remarkId": str}, engine='python')

n_ratings = 10
n_remarks = 1
remark_dates_to_use = 0
N_MODELS_IN_BEST_SCORE = 20
BEST_THRESHOLD = 6
global_seed = 1331


def generate_features(generate_train_features=True):
    TRAIN = generate_train_features
    if TRAIN:
        print("Generating features for the training data")
    else:
        print("Generating features for the test data")
    # 42 remarks which do not have data points repeated
    # one of the remarks has nan as a remarkDate
    # It, however, has another datapoint which contains the correspondin remarkDate
    companies = pd.unique(train['comp'])
    company_specific_ratings = np.asanyarray(
        list(zip(ratings['comp'], ratings['rating'])))
    np.shape(company_specific_ratings)
    avg_company_ratings = {}
    for company in companies:
        ratings_ = company_specific_ratings[company_specific_ratings[:, 0]
                                            == company][:, 1]
        avg_ratings = sum([int(i) for i in ratings_])/len(ratings_)
        avg_company_ratings[company] = avg_ratings

    rating_ids = np.asarray(list(zip(ratings['emp'], ratings['comp'])))
    rating_date_ids = list(
        zip(ratings['emp'], ratings['comp'], ratings['Date']))
    remark_ids = np.asarray(list(zip(remarks['emp'], remarks['comp'])))
    remark_supp_opp_inds = np.array(remarks_supp_opp['remarkId'])
    remarks_made = np.asarray(
        list(zip(remarks_supp_opp['emp'], remarks_supp_opp['comp'])))
    companies = pd.unique(train['comp'])
    remark_dates = np.asarray(
        list(zip(remarks['remarkId'], remarks['remarkDate'])))
    n = len(train['id'])
    assert remark_dates_to_use <= n_remarks, "Remark dates to use must be less than the number of remarks"
    print(f"{n_ratings} ratings being used")
    print(f"{n_remarks} remarks being used")
    USE_AVERAGE_REMARK_SUPPORT_OPPOSE = True
    if USE_AVERAGE_REMARK_SUPPORT_OPPOSE:
        d = 1+n_ratings+n_ratings+1+4+(n_remarks*4)
    else:
        d = 1+n_ratings+n_ratings+1+3+(n_remarks*3)
    if USE_AVERAGE_REMARK_SUPPORT_OPPOSE:
        print("Average of remarks supports and opposes being used")
    d = d+remark_dates_to_use
    X_train = np.ones((n, d))*np.nan
    Y_train = np.ones(n)*np.nan

    skipped = 0
    not_rated = 0

    def get_int_from_datetime_object(date):
        return 10000*(date.year)+100*(date.month)+date.day

    def get_last_avail_rating(rating_date_ids, emp, comp, date):
        try:
            ind = rating_date_ids.index((emp, comp, date))
            return ind
        except ValueError:
            inds = [rating_date_ids.index(
                el) for el in rating_date_ids if el[0] == emp and el[1] == comp]
            if len(inds):
                return inds[-1]
            return None

    def get_prev_remarks(all_remark_inds, remarks, num, rating_date, only_id=True):
        '''Get previous num remarks made before the rating date.
        only_id=True returns only the remark Ids'''
        if not only_id:
            remarks = []
        remarkIds = []
        count = 0
        for remark_ind in reversed(list(all_remark_inds)):
            if count == num:
                break
            curr_date = remarks['remarkDate'][remark_ind]
            if curr_date is np.nan:  # some dates are nans
                continue
            curr_date = datetime.strptime(curr_date, '%d-%m-%Y')
            if rating_date is None:
                remarkIds.append(remarks['remarkId'][remark_ind])
                if not only_id:
                    remarks.append(len(remarks['txt'][remark_ind]))
                count += 1
            elif curr_date <= rating_date:
                try:
                    remark = len(remarks['txt'][remark_ind])
                except TypeError:
                    continue
                remarkId = remarks['remarkId'][remark_ind]
                remarkIds.append(remarkId)
                if not only_id:
                    remarks.append(remark)
                count += 1
        if only_id:
            return remarkIds
        return remarks, remarkIds

    dates = []

    if TRAIN:
        num = n
    else:
        num = len(test['id'])  # create test data
        X_test = np.ones((num, d))*np.nan
    for i in tqdm(range(num)):
        if TRAIN:
            _, emp, comp, date, label = [train[key][i] for key in train.keys()]
            Y_train[i] = label
            matrix = X_train
        else:
            _, emp, comp, date = [test[key][i] for key in test.keys()]
            matrix = X_test
        comp_ind = list(companies).index(comp)

        matrix[i, 0] = comp_ind
        avg_rating = avg_company_ratings[comp]
        matrix[i, 2*n_ratings+1] = avg_rating
        remarks_supported_or_opposed_by_employee = remarks_supp_opp['support'][np.sum(
            remarks_made == (emp, comp), axis=1) == 2]
        remarks_supported_or_opposed_by_employee = np.where(
            remarks_supported_or_opposed_by_employee == True, 1, -1)
        # some exployees have not supported or opposed any remarkId
        if not len(remarks_supported_or_opposed_by_employee):
            not_rated += 1
        else:
            total_pos_supports = np.sum(
                remarks_supported_or_opposed_by_employee == 1)
            total_neg_supports = np.sum(
                remarks_supported_or_opposed_by_employee == -1)
            matrix[i, 2*n_ratings +
                   2] = np.sum(remarks_supported_or_opposed_by_employee)
            matrix[i, 2*n_ratings+3] = total_pos_supports
            matrix[i, 2*n_ratings+4] = total_neg_supports
            if USE_AVERAGE_REMARK_SUPPORT_OPPOSE:
                matrix[i, 2*n_ratings +
                       5] = np.mean(remarks_supported_or_opposed_by_employee)
        try:
            # sometimes, there is no rating corresponding to last rating date!!!
            ind = get_last_avail_rating(rating_date_ids, emp, comp, date)
            # get all remarks made by the employee
            all_remark_inds = np.sum(remark_ids == (emp, comp), axis=1) == 2
            all_remark_inds = np.arange(remark_ids.shape[0])[all_remark_inds]
            if ind:
                rating = ratings['rating'][ind]
                matrix[i, 1] = rating
                # both emp and comp should match
                all_inds = np.sum(rating_ids == (emp, comp), axis=1) == 2
                prev_ratings = list(ratings['rating'][all_inds])
                for j in range(1, len(prev_ratings)):
                    matrix[i, j+1] = prev_ratings[-(j+1)]
                    if j+1 == n_ratings:
                        break

                # datetime object allows date comparison with <, >
                rating_date = datetime.strptime(date, '%d-%m-%Y')
                date_feature = get_int_from_datetime_object(rating_date)
                all_rating_dates = [datetime.strptime(
                    date, '%d-%m-%Y') for date in ratings['Date'][all_inds]]
                valid_dates = [
                    date for date in all_rating_dates if date <= rating_date]
                all_date_features = [get_int_from_datetime_object(
                    date) for date in all_rating_dates if date <= rating_date]
                matrix[i, n_ratings+1] = date_feature
                valid_dates = sorted(valid_dates)
                for j in range(1, len(all_date_features)):
                    matrix[i, j+1+n_ratings] = all_date_features[-(j+1)]
                    if j+1 == n_ratings:
                        break

                dates.append(len(valid_dates))

                remarkIds = get_prev_remarks(
                    all_remark_inds, remarks, n_remarks, rating_date)
            else:
                remarkIds = get_prev_remarks(
                    all_remark_inds, remarks, n_remarks, None)

            if not len(remarkIds):
                skipped += 1
                continue
            for j, remarkId in enumerate(remarkIds):
                # get all remark supp-opp ratings corresponding to remarkId
                inds = remark_supp_opp_inds == remarkId
                supps = remarks_supp_opp['support'][inds]
                supps = np.where(supps == True, 1, -1)  # opp-> -1, supp -> +1
                n_pos = np.sum(supps == 1)
                n_neg = np.sum(supps == -1)
                # update matrix
                if USE_AVERAGE_REMARK_SUPPORT_OPPOSE:
                    matrix[i, 2*n_ratings+6+4*j] = np.sum(supps)
                    matrix[i, 2*n_ratings+7+4*j] = n_pos
                    matrix[i, 2*n_ratings+8+4*j] = n_neg
                    # some remarks have no supports or opposes
                    if len(supps):
                        matrix[i, 2*n_ratings+9+4*j] = np.mean(supps)
                    else:
                        matrix[i, 2*n_ratings+9+4*j] = 0
                        # for remarks with no supports and opposes, set mean also to be zero
                    if j >= remark_dates_to_use:
                        continue
                else:
                    matrix[i, 2*n_ratings+5+3*j] = np.sum(supps)
                    matrix[i, 2*n_ratings+6+3*j] = n_pos
                    matrix[i, 2*n_ratings+7+3*j] = n_neg

                if j >= remark_dates_to_use:
                    continue
                # for repeated data points, take the first
                remark_date = remark_dates[remark_dates[:, 0]
                                           == remarkId][:, 1][0]
                remark_date = get_int_from_datetime_object(
                    datetime.strptime(remark_date, '%d-%m-%Y'))
                matrix[i, d-remark_dates_to_use+j] = remark_date
        except ValueError:  # when there is no rating by the employee
            skipped += 1
            continue
        except Exception as e:
            sys.exit(1)

    if TRAIN:
        np.save('X_train_new.npy', X_train)
        np.save('Y_train_new.npy', Y_train)
    else:
        np.save('X_test_new.npy', X_test)

# generate features for the train data
generate_features(generate_train_features=True)
# generate features for the test data
generate_features(generate_train_features=False)


# load the generated data
X_train = np.load('X_train_new.npy')
Y_train = np.load('Y_train_new.npy')
X_test = np.load('X_test_new.npy')
orig_num = X_train.shape[-1]

n_ratings_being_used = n_ratings
n_remarks = n_remarks
n_remark_dates = remark_dates_to_use

all_ratings_train = X_train[:, 1:n_ratings_being_used+1]
avg_ratings_train = np.nanmean(all_ratings_train, axis=1).reshape(-1, 1)
all_ratings_test = X_test[:, 1:n_ratings_being_used+1]
avg_ratings_test = np.nanmean(all_ratings_test, axis=1).reshape(-1, 1)

X_train = np.hstack((X_train, avg_ratings_train))  # add average user ratings
X_test = np.hstack((X_test, avg_ratings_test))

# last rating - avg user rating
X_train_d = np.concatenate(
    [X_train, X_train[:, 1, None]-X_train[:, orig_num, None]], axis=1)
X_test_d = np.concatenate(
    [X_test, X_test[:, 1, None]-X_test[:, orig_num, None]], axis=1)

X_train_d = np.concatenate([X_train_d, X_train_d[:, 2*n_ratings_being_used+1,
                                                 None]-X_train_d[:, orig_num, None]], axis=1)  # avg comp - avg user
X_test_d = np.concatenate(
    [X_test_d, X_test_d[:, 2*n_ratings_being_used+1, None]-X_test_d[:, orig_num, None]], axis=1)
X_train_d = np.concatenate([X_train_d, X_train_d[:, orig_num,
                                                 None]/X_train_d[:, 2*n_ratings_being_used+1, None]], axis=1)
X_test_d = np.concatenate([X_test_d, X_test_d[:, orig_num, None] /
                           X_test_d[:, 2*n_ratings_being_used+1, None]], axis=1)  # avg user/avg comp

X_train = X_train_d.copy()
X_test = X_test_d.copy()

all_X_train = X_train.copy()  # make a copy just in case
all_Y_train = Y_train.copy()
X_comp = X_train[:, 0].astype(int)
comp_stats = np.bincount(X_comp)

# Find the ratio of zeroes and ones
old_zeroes = np.sum(Y_train == 0)
old_ones = np.sum(Y_train == 1)
original_ratio = old_ones/old_zeroes

comp_indices = set()
for x in X_train[:, 0]:
    comp_indices.add(x)
threshold = 20

X_train_ = X_train.copy()
Y_train_ = Y_train.copy()
for index in comp_indices:
    indices = X_train_[:, 0] == index
    number_of_valid_points = np.sum(indices)
    gap_to_threshold = threshold-number_of_valid_points
    if gap_to_threshold > 0:
        if gap_to_threshold > number_of_valid_points:
            replace = True
        else:
            replace = False

        relevant_data_samples = X_train_[indices]
        relevant_labels = Y_train_[indices]
        np.random.seed(global_seed+int(index))
        indices_to_concatenate = np.random.choice(
            relevant_data_samples.shape[0], gap_to_threshold, replace=replace)
        samples_to_concatenate = relevant_data_samples[indices_to_concatenate, :]
        labels_to_concatenate = relevant_labels[indices_to_concatenate]
        X_train_ = np.concatenate([X_train_, samples_to_concatenate], axis=0)
        Y_train_ = np.concatenate([Y_train_, labels_to_concatenate], axis=0)

X_comp_ = X_train_[:, 0].astype(int)
comp_stats = np.bincount(X_comp_)

# Check the new ratio of zeroes and ones
new_zeroes = np.sum(Y_train_ == 0)
new_ones = np.sum(Y_train_ == 1)
new_ratio = new_ones/new_zeroes

ones_to_add = int(original_ratio*new_zeroes-new_ones)
n = new_ones+ones_to_add
n1 = new_zeroes

one_data_points = X_train_[Y_train_ == 1]
# If you have enough 1's in X_train_, sample without replacement, else with replacement
replace = (one_data_points.shape[0] < ones_to_add)
np.random.seed(global_seed)
indices_to_concatenate = np.random.choice(
    one_data_points.shape[0], ones_to_add, replace=replace)
samples_to_concatenate = one_data_points[indices_to_concatenate]
labels_to_concatenate = np.ones(ones_to_add)
X_train_ = np.concatenate([X_train_, samples_to_concatenate], axis=0)
Y_train_ = np.concatenate([Y_train_, labels_to_concatenate], axis=0)

# Check the new ratio of zeroes and ones
new_zeroes = np.sum(Y_train_ == 0)
new_ones = np.sum(Y_train_ == 1)
new_ratio = new_ones/new_zeroes

X_train = X_train_
Y_train = Y_train_
r = min(original_ratio, new_ratio)  # Use the lower of the ratios
division_factor = (1/r)+1

# Use this for experiments
train_comp_ids = X_train[:, 0].astype(np.int)
test_comp_ids = X_test[:, 0].astype(np.int)

X_train_ = X_train[:, 1:]
X_test_ = X_test[:, 1:]

train_bin = LabelBinarizer()
test_bin = LabelBinarizer()
train_bin.fit(range(max(train_comp_ids)+1))
test_bin.fit(range(max(test_comp_ids)+1))
train_one_hot = train_bin.transform(train_comp_ids)
test_one_hot = test_bin.transform(test_comp_ids)

X_train_ = np.hstack((train_one_hot, X_train[:, 1:]))
X_test_ = np.hstack((test_one_hot, X_test[:, 1:]))

X_train = X_train_.copy()
X_test = X_test_.copy()

# increase val set number as repeating data will lead to overfitting
val_size = int(0.2*Y_train.size)
one_inds = Y_train == 1
zero_inds = Y_train == 0
zero_indices = np.arange(len(Y_train))[zero_inds]
one_indices = np.arange(len(Y_train))[one_inds]
np.random.seed(global_seed)
rand_one_inds = np.random.choice(one_indices, size=int(
    val_size/division_factor), replace=False)
np.random.seed(global_seed)
rand_zero_inds = np.random.choice(
    zero_indices, size=val_size - len(rand_one_inds), replace=False)
train_inds = [i for i in np.arange(len(Y_train)) if (
    i not in rand_one_inds and i not in rand_zero_inds)]
X_val = np.concatenate(
    [X_train[rand_one_inds], X_train[rand_zero_inds]], axis=0)
Y_val = np.concatenate(
    [Y_train[rand_one_inds], Y_train[rand_zero_inds]], axis=0)

X_train_mod = X_train.copy()
Y_train_mod = Y_train.copy()

eval_set = [(X_val, Y_val)]


def acc(Y_true, Y_pred):
    return sum(Y_true == Y_pred)/len(Y_pred)


def weighted_acc(Y_true, Y_pred):
    one_inds = Y_true == 1
    one_sum = sum(Y_true[one_inds] == Y_pred[one_inds])
    other_sum = sum(Y_true[Y_true == 0] == Y_pred[Y_true == 0])
    acc = (5*one_sum + other_sum)/(5*sum(one_inds) + sum(Y_true == 0))
    return acc

# custom evaluation metric for xgboost


def weighted_eval(preds, dmatrix):  # preds will be in the form of probabilities
    Y_pred = np.where(preds >= 0.5, 1, 0)
    Y_true = dmatrix.get_label()
    one_inds = Y_true == 1
    one_sum = sum(Y_true[one_inds] == Y_pred[one_inds])
    other_sum = sum(Y_true[Y_true == 0] == Y_pred[Y_true == 0])
    acc = (5*one_sum + other_sum)/(5*sum(one_inds) + sum(Y_true == 0))
    return 'weighted-acc', acc


def weighted_bce_wb(y_pred, dtrain, w=3):
    y_true = dtrain.label
    eps = 1e-6
    grad = -w*np.divide(y_true, y_pred+eps)+np.divide(1-y_true, 1-y_pred+eps)
    hess = w*np.divide(y_true, y_pred**2+eps) - \
        np.divide(1-y_true, (1-y_pred)**2+eps)
    return grad, hess


max_depths = [3, 4, 5, 6]
num_parallel_trees = [63, 127, 255]
learning_rates = [0.1, 0.3, 0.5, 1]
num_estimators = [10, 15, 20, 25]
subsamples = [0.8, 0.9, 1]
colsamples = [0.8, 0.9, 0.6, 1]
fractions = [0.9, 0.99]
n_models_to_use = 20

random.seed(global_seed)
depth_seeds = np.array([0, 5, 1, 1, 0, 0, 0, 1, 2, 0,
                        0, 2, 0, 2, 1, 2, 1, 2, 5, 1])
tree_seeds = np.array([0, 5, 0, 0, 5, 5, 0, 1, 1, 0,
                       5, 0, 1, 0, 1, 5, 1, 1, 0, 5])
lr_seeds = np.array([5, 5, 5, 1, 2, 0, 0, 0, 2, 0,
                     0, 0, 5, 0, 0, 2, 5, 0, 5, 1])
estimator_seeds = np.array(
    [1, 1, 1, 2, 0, 1, 5, 2, 0, 1, 1, 2, 2, 1, 1, 0, 2, 5, 1, 5])
subsamples_seeds = np.array(
    [5, 1, 0, 1, 5, 5, 0, 1, 5, 1, 5, 5, 5, 0, 1, 1, 1, 0, 1, 0])
colsamples_seeds = np.array(
    [5, 1, 5, 0, 2, 0, 0, 5, 2, 1, 2, 5, 5, 2, 5, 1, 1, 5, 0, 2])

models = []
total_ones = np.sum(Y_train_mod == 1)
total_zeroes = np.sum(Y_train_mod == 0)
for i in tqdm(range(n_models_to_use)):
    if i < N_MODELS_IN_BEST_SCORE:
        random.seed(depth_seeds[i])
        max_depth = random.choice(max_depths)
        fraction_of_data_to_use_in_each_ensemble_model = random.choice(
            fractions)

        random.seed(subsamples_seeds[i])
        subsample = random.choice(subsamples)

        random.seed(tree_seeds[i])
        num_parallel_tree = random.choice(num_parallel_trees)

        random.seed(lr_seeds[i])
        learning_rate = random.choice(learning_rates)

        random.seed(colsamples_seeds[i])
        colsample_bynode = random.choice(colsamples)

        random.seed(estimator_seeds[i])
        n_estimator = random.choice(num_estimators)
    else:
        fraction_of_data_to_use_in_each_ensemble_model = random.choice(
            fractions)
        subsample = random.choice(subsamples)
        num_parallel_tree = random.choice(num_parallel_trees)
        learning_rate = random.choice(learning_rates)
        colsample_bynode = random.choice(colsamples)
        n_estimator = random.choice(num_estimators)
        max_depth = random.choice(max_depths)

    assert fraction_of_data_to_use_in_each_ensemble_model < 1, "Fraction of data sample in each ensemble must be strictly less than 1.\nValue used here = {}".format(
        fraction_of_data_to_use_in_each_ensemble_model)
    n_ones = int(fraction_of_data_to_use_in_each_ensemble_model *
                 Y_train_mod.size/division_factor)
    n_zeroes = int(
        fraction_of_data_to_use_in_each_ensemble_model*Y_train_mod.size-n_ones)
    ones_data = X_train_mod[Y_train_mod == 1]
    zeroes_data = X_train_mod[Y_train_mod == 0]
    one_indices_to_concatenate = np.random.choice(
        total_ones, n_ones, replace=False)
    one_samples_to_concatenate = ones_data[one_indices_to_concatenate, :]
    one_labels_to_concatenate = np.ones(n_ones)
    zero_indices_to_concatenate = np.random.choice(
        total_zeroes, n_zeroes, replace=False)
    zero_samples_to_concatenate = zeroes_data[zero_indices_to_concatenate, :]
    zero_labels_to_concatenate = np.zeros(n_zeroes)
    X_train_mod_sample = np.concatenate(
        [one_samples_to_concatenate, zero_samples_to_concatenate], axis=0)
    Y_train_mod_sample = np.concatenate(
        [one_labels_to_concatenate, zero_labels_to_concatenate], axis=0)
    scale_pos_weight = np.sum(Y_train_mod_sample == 0) / \
        np.sum(Y_train_mod_sample == 1)
    xgb_model = xgb.XGBClassifier(subsample=subsample, colsample_bynode=colsample_bynode, scale_pos_weight=scale_pos_weight,
                                  n_estimators=n_estimator, learning_rate=learning_rate, max_depth=max_depth, num_parallel_tree=num_parallel_tree)
    xgb_model.fit(X_train_mod_sample, Y_train_mod_sample,
                  eval_metric=weighted_eval, eval_set=eval_set)
    models.append(xgb_model)

predictions = np.zeros(Y_val.size)
train_preds = np.zeros(Y_train_mod.size)
for model in models:
    prediction = model.predict(X_val)
    predictions += prediction
    train_pred = model.predict(X_train_mod)
    train_preds += train_pred

predictions_ = predictions.copy()
train_preds_ = train_preds.copy()

threshold = BEST_THRESHOLD

Y_preds = np.zeros(X_test.shape[0])
for model in models:
    prediction = model.predict(X_test)
    Y_preds += prediction

filename_to_save = "Overfitters_EE17B031_EE17B032.csv"

ones_in_test = Y_preds > threshold
Y_preds[ones_in_test] = 1
Y_preds[~ones_in_test] = 0

with open(filename_to_save, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'left'])
    for i in range(len(X_test)):
        id1 = test['id'][i]
        writer.writerow([id1, int(Y_preds[i])])


os.remove("X_train_new.npy")
os.remove("X_test_new.npy")
os.remove("Y_train_new.npy")
