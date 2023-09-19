#################################################################
##### Part 1: Radiomics signature construction (survivalsvm)#####
#################################################################
# Training dataset for radiomics signature construction
data_rad_GD <- data1 %>%
  filter(center == "GD" & use == 1) %>%
  select(DFS_status, DFS_month, rad_feat_sel[rad_feat_sel$rank <= 5, ]$Var1)
# Testing dataset for radiomics signature validation
data_rad_XY <- data1 %>%
  filter(center == "XY" & use == 1) %>%
  select(DFS_status, DFS_month, rad_feat_sel[rad_feat_sel$rank <= 5, ]$Var1)
data_rad_SX.JM <- data1 %>%
  filter((center == "SX" | center == "JM") & use == 1) %>%
  select(DFS_status, DFS_month, rad_feat_sel[rad_feat_sel$rank <= 5, ]$Var1)
# Model construction
model_GD <- survivalsvm(Surv(DFS_month, DFS_status) ~ .,
  data = data_rad_GD,
  gamma.mu = parameter
)
# Output radiomics signature and evaluate its performance (training dataset)
data_rad_GD <- data_rad_GD %>%
  mutate(rad_sig = as.numeric(predict(model_GD, newdata = data_rad_GD)$predicted))
model_rad_sig_GD <- coxph(Surv(DFS_month, DFS_status) ~ rad_sig, data_rad_GD)
print(paste("GD c-index: "))
print(summary(model_rad_sig_GD)$concordance)
# Output radiomics signature and evaluate its performance (testing dataset 1)
data_rad_XY <- data_rad_XY %>%
  mutate(rad_sig = as.numeric(predict(model_GD, newdata = data_rad_XY)$predicted))
model_rad_sig_XY <- coxph(Surv(DFS_month, DFS_status) ~ rad_sig, data_rad_XY)
print(paste("XY c-index: "))
print(summary(model_rad_sig_XY)$concordance)
# Output radiomics signature and evaluate its performance (testing dataset 2)
data_rad_SX.JM <- data_rad_SX.JM %>%
  mutate(rad_sig = as.numeric(predict(model_GD, newdata = data_rad_SX.JM)$predicted))
model_rad_sig_SX.JM <- coxph(Surv(DFS_month, DFS_status) ~ rad_sig, data_rad_SX.JM)
print(paste("SX.JM c-index: "))
print(summary(model_rad_sig_SX.JM)$concordance)
rm(list = ls(pattern = "model"))



##################################################################
##### Part 2: Pathomics signature construction (survivalsvm)######
##################################################################
# Training dataset for pathomics signature construction
data_pat_GD <- data1 %>%
  filter(center == "GD" & use == 1) %>%
  select(DFS_status, DFS_month, pat_feat_sel[pat_feat_sel$rank <= 4, ]$Var1)
# Testing dataset for pathomics signature validation
data_pat_XY <- data1 %>%
  filter(center == "XY" & use == 1) %>%
  select(DFS_status, DFS_month, pat_feat_sel[pat_feat_sel$rank <= 4, ]$Var1)
data_pat_SX.JM <- data1 %>%
  filter((center == "SX" | center == "JM") & use == 1) %>%
  select(DFS_status, DFS_month, pat_feat_sel[pat_feat_sel$rank <= 4, ]$Var1)
# Model construction
model_GD <- survivalsvm(Surv(DFS_month, DFS_status) ~ .,
  data = data_pat_GD,
  gamma.mu = parameter
)
# Output pathomics signature and evaluate its performance (training dataset)
data_pat_GD <- data_pat_GD %>%
  mutate(pat_sig = as.numeric(predict(model_GD, newdata = data_pat_GD)$predicted))
model_pat_sig_GD <- coxph(Surv(DFS_month, DFS_status) ~ pat_sig, data_pat_GD)
print(paste("GD c-index: "))
print(summary(model_pat_sig_GD)$concordance)
# Output pathomics signature and evaluate its performance (testing dataset 1)
data_pat_XY <- data_pat_XY %>%
  mutate(pat_sig = as.numeric(predict(model_GD, newdata = data_pat_XY)$predicted))
model_pat_sig_XY <- coxph(Surv(DFS_month, DFS_status) ~ pat_sig, data_pat_XY)
print(paste("XY c-index: "))
print(summary(model_pat_sig_XY)$concordance)
# Output pathomics signature and evaluate its performance (testing dataset 2)
data_pat_SX.JM <- data_pat_SX.JM %>%
  mutate(pat_sig = as.numeric(predict(model_GD, newdata = data_pat_SX.JM)$predicted))
model_pat_sig_SX.JM <- coxph(Surv(DFS_month, DFS_status) ~ pat_sig, data_pat_SX.JM)
print(paste("SX.JM c-index: "))
print(summary(model_pat_sig_SX.JM)$concordance)
rm(list = ls(pattern = "model"))



########################################################################
##### Part 3: Radio-pathomics signature construction (survivalsvm)######
########################################################################
# Training dataset for radio-pathomics signature construction
data_rad.pat_GD <- data1 %>%
  filter(center == "GD" & use == 1) %>%
  select(
    DFS_status, DFS_month, rad_feat_sel[rad_feat_sel$rank <= 5, ]$Var1,
    pat_feat_sel[pat_feat_sel$rank <= 4, ]$Var1
  )
# Testing dataset for radio-pathomics signature validation
data_rad.pat_XY <- data1 %>%
  filter(center == "XY" & use == 1) %>%
  select(
    DFS_status, DFS_month, rad_feat_sel[rad_feat_sel$rank <= 5, ]$Var1,
    pat_feat_sel[pat_feat_sel$rank <= 4, ]$Var1
  )
data_rad.pat_SX.JM <- data1 %>%
  filter((center == "SX" | center == "JM") & use == 1) %>%
  select(
    DFS_status, DFS_month, rad_feat_sel[rad_feat_sel$rank <= 5, ]$Var1,
    pat_feat_sel[pat_feat_sel$rank <= 4, ]$Var1
  )
# Model construction
model_GD <- survivalsvm(Surv(DFS_month, DFS_status) ~ .,
  data = data_rad.pat_GD,
  gamma.mu = parameter
)
# Output radio-pathomics signature and evaluate its performance (training dataset)
data_rad.pat_GD <- data_rad.pat_GD %>%
  mutate(rad.pat_sig = as.numeric(predict(model_GD, newdata = data_rad.pat_GD)$predicted))
model_rad.pat_sig_GD <- coxph(Surv(DFS_month, DFS_status) ~ rad.pat_sig, data_rad.pat_GD)
print(paste("GD c-index: "))
print(summary(model_rad.pat_sig_GD)$concordance)
# Output radio-pathomics signature and evaluate its performance (testing dataset 1)
data_rad.pat_XY <- data_rad.pat_XY %>%
  mutate(rad.pat_sig = as.numeric(predict(model_GD, newdata = data_rad.pat_XY)$predicted))
model_rad.pat_sig_XY <- coxph(Surv(DFS_month, DFS_status) ~ rad.pat_sig, data_rad.pat_XY)
print(paste("XY c-index: "))
print(summary(model_rad.pat_sig_XY)$concordance)
# Output radio-pathomics signature and evaluate its performance (testing dataset 2)
data_rad.pat_SX.JM <- data_rad.pat_SX.JM %>%
  mutate(rad.pat_sig = as.numeric(predict(model_GD, newdata = data_rad.pat_SX.JM)$predicted))
model_rad.pat_sig_SX.JM <- coxph(Surv(DFS_month, DFS_status) ~ rad.pat_sig, data_rad.pat_SX.JM)
print(paste("SX.JM c-index: "))
print(summary(model_rad.pat_sig_SX.JM)$concordance)
rm(list = ls(pattern = "model"))