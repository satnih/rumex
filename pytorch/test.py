# # # %% Test model------------------------------------------------------------
# model = load_pretrained(model_name, num_classes)
# model.load_state_dict(torch.load(model_name))

# dl_te = RumexDataset(data_dir+'test_uncleanded/', tfms_te).make_data_loader(bs)
# ntest = len(dl_te.dataset)
# nrumex = np.sum(dl_te.dataset.targets)
# nother = ntest - nrumex

# y_te = []
# score_te = []
# loss_te = []
# yhat_te = []
# for batch_idx_te, (x_te_b, y_te_b) in enumerate(dl_te):
#     x_te_b = x_te_b.to(device)
#     y_te_b = y_te_b.to(device)

#     # Forward pass
#     score_te_b = model(x_te_b)  # logits
#     loss_tr_b = loss_fn(score_te_b, y_te_b)
#     _, yhat_te_b = torch.max(score_te_b, 1)

#     # book keeping at batch level
#     y_te.append(y_te_b)
#     score_te.append(score_te_b)
#     yhat_te.append(yhat_te_b)

# # predictions and  metrics
# y_te = torch.cat(y_te).cpu().detach()
# score_te = torch.cat(score_te).cpu().detach()
# loss_te = torch.cat(loss_te).cpu().detach()
# yhat_te = torch.cat(yhat_te).cpu().detach()

# acc_te = accuracy_score(y_te, yhat_te)
# f1_te = f1_score(y_te, yhat_te)
# pre_te = precision_score(y_te, yhat_te)
# recall_te = recall_score(y_te, yhat_te)
# auc_te = roc_auc_score(y_te, score_te[:, 1])

# print(f"{model_name}|#rumex:{nrumex}|#other:{nother}")
# print(f"acc:{acc_te:.5f}|auc:{auc_te:.5f}|f1:{f1_te:.5f}" +
#       f"|pre:{pre_te:.5f}|recall:{recall_te:.5f}")

# # %%
