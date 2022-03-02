import torch
import numpy as np
from torch.utils.data import DataLoader
import data_utils
from test_mimic import test_evaluation
import models
device = torch.device("cpu")
fold = '1'
simulation_name = 'mimic'
params_dict = np.load(fr"mimic_fold_idx_{fold}/mimic_params.npy",allow_pickle=True).item()
model = models.NNFOwithBayesianJumps(input_size=params_dict["input_size"],
                                              hidden_size=params_dict["hidden_size"],
                                              p_hidden=params_dict["p_hidden"], prep_hidden=params_dict["prep_hidden"],
                                              logvar=params_dict["logvar"], mixing=params_dict["mixing"],
                                              classification_hidden=params_dict["classification_hidden"],
                                              cov_size=params_dict["cov_size"], cov_hidden=params_dict["cov_hidden"],
                                              dropout_rate=params_dict["dropout_rate"],
                                              full_gru_ode=params_dict["full_gru_ode"], impute=params_dict["impute"])

optimizer = torch.optim.Adam(model.parameters(), lr=params_dict["lr"], weight_decay=params_dict["weight_decay"])
class_criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
model.load_state_dict(torch.load(rf"{simulation_name}_fold_idx_{fold}/{simulation_name}_MAX.pt"))
test_idx = np.load(rf"{simulation_name}_fold_idx_{fold}/test_idx.npy",allow_pickle=True)
print(test_idx)
csv_file_path = params_dict["csv_file_path"]
csv_file_cov = params_dict["csv_file_cov"]
csv_file_tags = params_dict["csv_file_tags"]
if params_dict["lambda"] == 0:
    validation = True
    val_options = {"T_val": params_dict["T_val"], "max_val_samples": params_dict["max_val_samples"]}
else:
    validation = False
    val_options = None


data_test = data_utils.ODE_Dataset(csv_file=csv_file_path,label_file=csv_file_tags,
                                        cov_file= csv_file_cov, idx=test_idx, validation = validation,
                                        val_options = val_options)
dl_test = DataLoader(dataset=data_test, collate_fn=data_utils.custom_collate_fn, shuffle=True, batch_size=len(test_idx))
loss_test, auc_total_test, mse_test = test_evaluation(model, params_dict, class_criterion, device, dl_test)
