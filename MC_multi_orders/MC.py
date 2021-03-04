import numpy as np
import itertools
import re

class MarkovChain():
  def __init__(self, item_dict, list_entry_dict, list_freq_dict, list_reverse_entry_dict, list_transition_matrix, weight_behaivor, mc_order):
    self.list_freq_dict = list_freq_dict
    self.item_dict = item_dict
    self.nb_items = len(self.item_dict)
    # self.sp_matrix_path = sp_matrix_path
    self.mc_order = mc_order
    self.w_behavior = weight_behaivor
    self.list_transition_matrix = list_transition_matrix
    self.list_entry_dict = list_entry_dict
    self.list_reverse_entry_dict = list_reverse_entry_dict

  def top_predicted_item(self, previous_baskets, topk):
    candidate = np.zeros(self.nb_items)
    list_prev = previous_baskets[0]
    for i in range(1, len(previous_baskets)):
        list_prev = itertools.product(list_prev, previous_baskets[i])
    orders_mc_dict = {i:[] for i in range(len(self.list_entry_dict))}
    for i in range(len(list_prev)):
        tup = list_prev[i]
        order = len(self.list_entry_dict)-1
        while tup in self.list_entry_dict[order]:
            order = order-1
            listx = list(tup)
            listx.pop(0)
            tup = tuple(listx)
        if order == -1:
            continue
        orders_mc_dict[i].append(tup)
    for j in range(len(orders_mc_dict)):
        list_prev_idx = [self.list_entry_dict[p] for p in orders_mc_dict[j]]
        candidate_order_score = np.array(self.list_transition_matrix[j][list_prev_idx, :].todense().sum(axis=0))[0]
        candidate += candidate_order_score / len(list_prev_idx)
    topk_idx = np.argpartition(candidate, -topk)[-topk:]
    sorted_topk_idx = topk_idx[np.argsort(candidate[topk_idx])]
    topk_item = [self.list_entry_dict[item] for item in sorted_topk_idx]
    # print("Done")
    return topk_item

