import torch
import numpy as np
import data_utils
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
format_name = 'jpg'
def test_evaluation(model, params_dict, class_criterion, device, dl_test):
    with torch.no_grad():
        model.eval()
        total_loss_test = 0
        auc_total_test = 0
        loss_test = 0
        mse_test  = 0
        corr_test = 0
        num_obs = 0
        for i, b in enumerate(dl_test):
            # print(i,b)
            times    = b["times"]
            time_ptr = b["time_ptr"]
            X        = b["X"].to(device)
            M        = b["M"].to(device)
            obs_idx  = b["obs_idx"]
            cov      = b["cov"].to(device)
            labels   = b["y"].to(device)
            batch_size = labels.size(0)

            if b["X_val"] is not None:
                X_val     = b["X_val"].to(device)
                M_val     = b["M_val"].to(device)
                times_val = b["times_val"]
                times_idx = b["index_val"]

            h0 = 0 #torch.zeros(labels.shape[0], params_dict["hidden_size"]).to(device)
            hT, loss, class_pred, t_vec, p_vec, h_vec, _, _ = model(times, time_ptr, X, M, obs_idx, delta_t=params_dict["delta_t"], T=params_dict["T"], cov=cov, return_path=True)
            total_loss = (loss + params_dict["lambda"]*class_criterion(class_pred, labels))/batch_size

            try:
                auc_test = roc_auc_score(labels.cpu(),torch.sigmoid(class_pred).cpu())
            except ValueError:
                if params_dict["verbose"]>=3:
                    print("Only one class. AUC is wrong")
                auc_test = 0
                pass

            if params_dict["lambda"]==0:
                t_vec = np.around(t_vec, str(params_dict["delta_t"])[::-1].find('.')).astype(np.float32) # Round floating points error in the time vector.
                p_val = data_utils.extract_from_path(t_vec, p_vec, times_val, times_idx)
                m, v = torch.chunk(p_val, 2, dim=1)
                last_loss = (data_utils.log_lik_gaussian(X_val,m,v)*M_val).sum()
                mse_loss = (torch.pow(X_val-m,2)*M_val).sum()
                n = X_val.shape[0]
                index = np.linspace(1, n, n)
                for k in range(3):
                # print(X_val.shape)
                    plt.plot(index, m[0:n, k] + 3 * v[0:n, k],'g--')
                    plt.plot(index, m[0:n, k] - 3 * v[0:n, k], 'g--')
                    # plt.plot(m[0:n, k], X_val[0:n, k], '*')
                    plt.title('Actual v Predicted (Testing)')
                    plt.plot(index, m[0:n, k], 'b*')
                    plt.plot(index, X_val[0:n, k],'ro')
                    plt.savefig(f'Item_{k}_testing_result.{format_name}')
                    plt.close()
                corr_test_loss = data_utils.compute_corr(X_val, m, M_val)

                loss_test += last_loss.cpu().numpy()
                num_obs += M_val.sum().cpu().numpy()
                mse_test += mse_loss.cpu().numpy()
                corr_test += corr_test_loss.cpu().numpy()
            else:
                num_obs=1

            total_loss_test += total_loss.cpu().detach().numpy()
            auc_total_test += auc_test

        loss_test /= num_obs
        mse_test /=  num_obs
        auc_total_test /= (i+1)

        return(loss_test, auc_total_test, mse_test)
