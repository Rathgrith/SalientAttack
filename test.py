from core.utils.utility import *
from torch.utils.data import Dataset
import torchvision
import os

# possibility_lists_dict = json_read_file("possibility_lists_dict.json")
#     # print(possibility_lists_dict)
# possibility_lists = possibility_lists_dict["possibility_lists_dict"]
# possibility_lists_dict_tem = json_read_file("possibility_lists_dict_tem.json")
#     # print(possibility_lists_dict)
# possibility_lists_tem = possibility_lists_dict_tem["possibility_lists_dict"]
# for i in range(len(possibility_lists)):
#     print(np.asarray(possibility_lists_tem[i])-np.asarray(possibility_lists[i]))
get_borda_path