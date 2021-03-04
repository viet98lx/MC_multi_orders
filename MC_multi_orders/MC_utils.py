import itertools
import scipy.sparse as sp
import re
import numpy as np
import time

def calculate_transition_matrix_at_order(train_instances, item_dict, prev_dict, prev_freq_dict, prev_reversed_item_dict, w_behavior, mc_order):
  pair_dict = dict()
  NB_ITEMS = len(item_dict)
  NB_ENTRY = len(prev_dict)
  print("number items: ", NB_ENTRY)
  # j = 0
  start = time.time()
  for line in train_instances:
      # print(j)
      # j += 1
      elements = line.split("|")
      user = elements[0]
      # print('User')
      basket_seq = elements[1:]
      st = mc_order
      # if len(basket_seq) < mc_order+1:
      #   st = 1
      for i in range(st,len(basket_seq)):
        prev_baskets = basket_seq[i-st:i]
        cur_basket = basket_seq[i]
        # prev_item_list = re.split('[\\s]+', prev_basket.strip())
        list_prev = [p.split(':')[0] for p in re.split('[\\s]+', prev_baskets[0].strip())]
        for basket in prev_baskets[1:]:
            list_item_in_basket = [p.split(':')[0] for p in re.split('[\\s]+', basket.strip())]
            list_prev = itertools.product(list_prev, list_item_in_basket)
        prev_ib_idx = [prev_dict[ib] for ib in list_prev]
        cur_item_list = [p.split(':')[0] for p in re.split('[\\s]+', cur_basket.strip())]
        cur_item_idx = [item_dict[item] for item in cur_item_list]
        for t in list(itertools.product(prev_ib_idx, cur_item_idx)):
            item_pair = (t[0], t[1])
            if item_pair in pair_dict.keys():
                pair_dict[item_pair] += 1
            else:
                pair_dict[item_pair] = 1
  end = time.time()
  print("Time to run all seq line: ", end-start)
  entries = sorted(list(pair_dict.keys()))
  entry_dict = dict()
  for entry in entries:
      entry_dict[entry] = len(entry_dict)

  reversed_entry_dict = dict(zip(entry_dict.values(), entry_dict.keys()))

  start_1 = time.time()

  row = [p[0] for p in pair_dict]
  col = [p[1] for p in pair_dict]
  data = [pair_dict[p]/prev_freq_dict[prev_reversed_item_dict[p[0]]] for p in pair_dict]
  transition_matrix = sp.csr_matrix((data, (row, col)), shape=(NB_ENTRY, NB_ITEMS), dtype="float32")
  end_1 = time.time()
  print("Time to Create transition matrix: ", end_1-start_1)
  nb_nonzero = len(pair_dict)
  density = nb_nonzero * 1.0 / NB_ENTRY / NB_ITEMS
  print("Density of matrix: {:.6f}".format(density))

  return transition_matrix, pair_dict, entry_dict, reversed_entry_dict

def build_knowledge(training_instances, w_behavior):
    MAX_SEQ_LENGTH = 0
    item_freq_dict = {}
    user_dict = dict()

    for line in training_instances:
        elements = line.split("|")

        if len(elements) - 1 > MAX_SEQ_LENGTH:
            MAX_SEQ_LENGTH = len(elements) - 1

        user = elements[0]
        user_dict[user] = len(user_dict)

        basket_seq = elements[1:]

        for basket in basket_seq:
            ib_pair = [tuple(p.split(':')) for p in re.split('[\\s]+', basket.strip())]
            # print(ib_pair)
            for item_obs in ib_pair:
                if item_obs[0] not in item_freq_dict:
                    # print(item_obs[0])
                    # print(item_obs[1])
                    item_freq_dict[item_obs[0]] = w_behavior[item_obs[1]]
                else:
                    item_freq_dict[item_obs[0]] += w_behavior[item_obs[1]]

    items = sorted(list(item_freq_dict.keys()))
    item_dict = dict()
    item_probs = []
    for item in items:
        item_dict[item] = len(item_dict)
        item_probs.append(item_freq_dict[item])

    item_probs = np.asarray(item_probs, dtype=np.float32)
    item_probs /= np.sum(item_probs)

    reversed_item_dict = dict(zip(item_dict.values(), item_dict.keys()))
    return MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs, item_freq_dict, user_dict

def write_predict(file_name, test_instances, topk, MC_model):
    f = open(file_name, 'w')
    for line in test_instances:
        elements = line.split("|")
        user = elements[0]
        if len(elements[1:]) < MC_model.mc_order+1:
            basket_seq = elements[1:-1]
        else:
            basket_seq = elements[-MC_model.mc_order-1:-1]
        last_basket = basket_seq[-1]
        # prev_basket = basket_seq[-2]
        if MC_model.mc_order == 1:
            prev_item_list = []
            for basket in basket_seq:
                prev_item_list += [p.split(':')[0] for p in re.split('[\\s]+', basket.strip())]
            list_predict_item = MC_model.top_predicted_item(prev_item_list, topk)
        else :
            prev_item_list = []
            for basket in basket_seq:
                prev_item_list.append([p.split(':')[0] for p in re.split('[\\s]+', basket.strip())])
            list_predict_item = MC_model.top_predicted_mc_order(prev_item_list, topk)
        # item_list = re.split('[\\s]+', last_basket.strip())
        cur_item_list = [p.split(':')[0] for p in re.split('[\\s]+', last_basket.strip())]
        f.write(str(user)+'\n')
        f.write('ground_truth:')
        for item in cur_item_list:
            f.write(' '+str(item))
        f.write('\n')
        f.write('predicted:')
        predict_len = len(list_predict_item)
        for i in range(predict_len):
            f.write(' '+str(list_predict_item[predict_len-1-i]))
        f.write('\n')
    f.close()

def read_predict(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    list_ground_truth_basket = []
    list_predict_basket = []
    for i in range(0, len(lines), 3):
        user = lines[i].strip('\n')
        list_ground_truth_basket.append(re.split('[\\s]+',lines[i+1].strip('\n'))[1:])
        list_predict_basket.append(re.split('[\\s]+',lines[i+2].strip('\n'))[1:])

    return list_ground_truth_basket, list_predict_basket

def hit_ratio(list_ground_truth_basket, list_predict_basket, topk):
    hit_count = 0
    for gt, predict in zip(list_ground_truth_basket, list_predict_basket):
        num_correct = len(set(gt).intersection(predict[:topk]))
        if num_correct > 0:
            hit_count += 1
            # user_correct.add(user)
    return hit_count / len(list_ground_truth_basket)

def recall(list_ground_truth_basket, list_predict_basket, topk):
    list_recall = []
    for gt, predict in zip(list_ground_truth_basket, list_predict_basket):
        num_correct = len(set(gt).intersection(predict[:topk]))
        list_recall.append(num_correct / len(gt))
    return np.array(list_recall).mean()

def read_instances_lines_from_file(file_path):
    with open(file_path, "r") as f:
        lines = [line.rstrip('\n') for line in f]
        return lines