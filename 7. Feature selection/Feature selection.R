##################################
##### 影像特征筛选 #####
##################################
# 筛选lasso分析集：筛选省医303例数据，筛选候选影像特征
data_lasso_rad <- data1 %>%
  filter(center == "GD" & use == 1) %>%
  select(c(1:36, 4284), cor_rad.pat_sel[grep(pattern = "RaTumor_", cor_rad.pat_sel$feature), ]$feature, 
         cor_rad.pat_sel[grep(pattern = "RaPeritumoral_", cor_rad.pat_sel$feature), ]$feature)
# lasso折交叉验证筛选影像特征
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
  print(glue("当前进度：{i} / 1000, 已完成：{round((i/1000)*100, digits = 1)}%"))
  i <- i + 1
}
# 提取影像特征被选中的次数
rad_feat_sel <- data.frame(table(lasso_rad_coef$feature)) %>%
  arrange(desc(Freq)) %>%
  mutate(rank = rank((1000 - Freq), ties.method = "average"))
# 清除当前环境过程性运行结果
rm(x_cv, y_cv, fit_cv, fit_coef, lasso_rad_res, lasso_rad_coef)
# 导出影像特征筛选结果
write.xlsx(rad_feat_sel, "rad_feat_sel.xlsx", colNames = T)
# 查看所选影像特征的相关性
cor_rad_res <- as.data.frame(cor((data1 %>%
                                    select(rad_feat_sel$Var1)), use = "complete.obs"))
# 导出影像特征相关性结果
write.xlsx(cor_rad_res, "cor_rad_res.xlsx", colNames = T)



##################################
##### 病理特征筛选 #####
##################################
# 筛选lasso分析集：筛选省医303例数据
data_lasso_pat <- data1 %>%
  filter(center == "GD" & use == 1) %>%
  select(c(1:36, 4284), cor_rad.pat_sel[grep(pattern = "PaTu_", cor_rad.pat_sel$feature), ]$feature, 
         cor_rad.pat_sel[grep(pattern = "PaEp_", cor_rad.pat_sel$feature), ]$feature, 
         cor_rad.pat_sel[grep(pattern = "PaSt_", cor_rad.pat_sel$feature), ]$feature,
         cor_rad.pat_sel[grep(pattern = "PaNu_", cor_rad.pat_sel$feature), ]$feature)
# lasso折交叉验证筛选病理特征
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
  print(glue("当前进度：{i} / 1000, 已完成：{round((i/1000)*100, digits = 1)}%"))
  i <- i + 1
}
# 提取病理特征被选中的次数
pat_feat_sel <- data.frame(table(lasso_pat_coef$feature)) %>%
  arrange(desc(Freq)) %>%
  mutate(rank = rank((1000 - Freq), ties.method = "average"))
# 清除当前环境过程性运行结果
rm(x_cv, y_cv, fit_cv, fit_coef, lasso_pat_res, lasso_pat_coef)
# 导出病理特征筛选结果
write.xlsx(pat_feat_sel, "pat_feat_sel.xls", colNames = T)
# 查看所选病理特征的相关性
cor_pat_res <- as.data.frame(cor((data1 %>%
                                    select(pat_feat_sel$Var1)), use = "complete.obs"))
# 导出病理特征相关性结果
write.xlsx(cor_pat_res, "cor_pat_res.xlsx", colNames = T)