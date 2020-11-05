-trainset: WENR_ortho_Rumex_10m_2_sw
-validset: WENR_ortho_Rumex_10m_3_ne
-testset: WENR_ortho_Rumex_10m_1_nw, WENR_ortho_Rumex_10m_4_se

-train_c: rumex patch is "center cropped" based on ground truth bbox
-train_w: patches creted by non-overlapping moving window of size 256
-train_wa: data augmentation applied to train_w patches
-test: patches creted by non-overlapping moving window (this is more realistic)
-valid: patches creted by non-overlapping moving window (this is more realistic)

NOTE: non-overlapping moving window patchs are more realisitc so valid and 
train set are always created this way. Train set can be crated for best performance