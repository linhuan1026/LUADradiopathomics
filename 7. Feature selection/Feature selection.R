###############################################
##### Part 1: Radiomics feature selection #####
###############################################
# Filter the dataset for radiomics feature selection
data_lasso_rad <- data1 %>%
  filter(center == "GD" & use == 1) %>%
  select(c(1:36, 4284), cor_rad.pat_sel[grep(pattern = "RaTumor_", cor_rad.pat_sel$feature), ]$feature, 
         cor_rad.pat_sel[grep(pattern = "RaPeritumoral_", cor_rad.pat_sel$feature), ]$feature)
# LASSO Cox regression with 10 fold cross validation
lasso_rad_coef <- data.frame(feature = character(), coef = numeric())
set.seed(0510)
i <- 1
x_cv <- as.matrix(data_lasso_rad[, -c(1:37)])
y_cv <- Surv(data_lasso_rad$DFS_month, data_lasso_rad$DFS_status)
while (i <= 1000) {
  fit_cv <- cv.glmnet(x_cv, y_cv, family = "cox", alpha = 1, nfolds = 10)
  fit_coef <- coef(fit_cv, s = "lambda.1se")
  lasso_rad_res <- as.data.frame(which(fit_coef != 0, arr.ind = T))
  lasso_rad_coef <- rbind(lasso_rad_coef, data.frame(feature = rownames(lasso_rad_res), coef = fit_coef[which(fit_coef != 0, arr.ind = T)]))
  print(glue("Current progress：{i} / 1000, Done：{round((i/1000)*100, digits = 1)}%"))
  i <- i + 1
}
# Calculate the number of votes for radiomics features
rad_feat_sel <- data.frame(table(lasso_rad_coef$feature)) %>%
  arrange(desc(Freq)) %>%
  mutate(rank = rank((1000 - Freq), ties.method = "average"))
rm(x_cv, y_cv, fit_cv, fit_coef, lasso_rad_res, lasso_rad_coef)
# Export the calculation result
write.xlsx(rad_feat_sel, "rad_feat_sel.xlsx", colNames = T)



###############################################
##### Part 2: Pathomics feature selection #####
###############################################
# Filter the dataset for pathomics feature selection
data_lasso_pat <- data1 %>%
  filter(center == "GD" & use == 1) %>%
  select(c(1:36, 4284), cor_rad.pat_sel[grep(pattern = "PaTu_", cor_rad.pat_sel$feature), ]$feature, 
         cor_rad.pat_sel[grep(pattern = "PaEp_", cor_rad.pat_sel$feature), ]$feature, 
         cor_rad.pat_sel[grep(pattern = "PaSt_", cor_rad.pat_sel$feature), ]$feature,
         cor_rad.pat_sel[grep(pattern = "PaNu_", cor_rad.pat_sel$feature), ]$feature)
# LASSO Cox regression with 10 fold cross validation
lasso_pat_coef <- data.frame(feature = character(), coef = numeric())
set.seed(0510)
i <- 1
x_cv <- as.matrix(data_lasso_pat[, -c(1:37)])
y_cv <- Surv(data_lasso_pat$DFS_month, data_lasso_pat$DFS_status)
while (i <= 1000) {
  fit_cv <- cv.glmnet(x_cv, y_cv, family = "cox", alpha = 1, nfolds = 10)
  fit_coef <- coef(fit_cv, s = "lambda.1se")
  lasso_pat_res <- as.data.frame(which(fit_coef != 0, arr.ind = T))
  lasso_pat_coef <- rbind(lasso_pat_coef, data.frame(feature = rownames(lasso_pat_res), coef = fit_coef[which(fit_coef != 0, arr.ind = T)]))
  print(glue("Current progress：{i} / 1000, Done：{round((i/1000)*100, digits = 1)}%"))
  i <- i + 1
}
# Calculate the number of votes for pathomics features
pat_feat_sel <- data.frame(table(lasso_pat_coef$feature)) %>%
  arrange(desc(Freq)) %>%
  mutate(rank = rank((1000 - Freq), ties.method = "average"))
rm(x_cv, y_cv, fit_cv, fit_coef, lasso_pat_res, lasso_pat_coef)
# Export the calculation result
write.xlsx(pat_feat_sel, "pat_feat_sel.xls", colNames = T)