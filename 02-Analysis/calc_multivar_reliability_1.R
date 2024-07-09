# ---
# jupyter:
#   jupytext:
#     formats: ipynb,Rmd,R:light
#     text_representation:
#       extension: .R
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# #find edges that are consistently higher than 0

library('I2C2') 
library('tidyverse') 


icc_calc <- function(df) { 

    df <- df[-'pos']

    wide_df <- pivot_wider(data=df, names_from='X', values_from='values')

    y <- wide_df[-1:-2]
    id <- wide_df$sub
    visit <- wide_df$ses

    lasso_i2c2 <- I2C2(y, id = id, visit = visit, demean = TRUE) 
    lasso_ci <- I2C2.mcCI(y, id = id, visit = visit, demean = TRUE)$CI

    return()
}

df = read.csv('../results/lasso_icc_df.csv') 

# +
df <- df[-5]

wide_df <- pivot_wider(data=df, names_from='X', values_from='values')
y <- wide_df[-1:-2]
id <- wide_df$sub
visit <- wide_df$ses

lasso_i2c2 <- I2C2(y, id = id, visit = visit, demean = TRUE) 
lasso_ci <- I2C2.mcCI(y, id = id, visit = visit, demean = TRUE)$CI
print("lasso")
print(lasso_ci)
# -

print(lasso_i2c2$lambda)

df = read.csv('../uoi_icc_df.csv')
dim(df) 

# +
df = read.csv('../uoi_icc_df.csv')
df <- df[-5]
wide_df <- pivot_wider(data=df, names_from='X', values_from='values')
y <- wide_df[-1:-2]
id <- wide_df$sub
visit <- wide_df$ses

uoi_i2c2 <- I2C2(y, id = id, visit = visit, demean = TRUE) 
uoi_ci <- I2C2.mcCI(y, id = id, visit = visit, demean = TRUE)$CI
print(uoi_ci)
print(uoi_i2c2$lambda)
# -

df = read.csv('../results/pearson_icc_df.csv')


dim(df) 

# +
df = read.csv('../results/pearson_icc_df.csv')
df <- df[-5]
wide_df <- pivot_wider(data=df, names_from='X', values_from='values')
y <- wide_df[-1:-2]
id <- wide_df$sub
visit <- wide_df$ses

pearson_i2c2 <- I2C2(y, id = id, visit = visit, demean = TRUE) 
pearson_ci <- I2C2.mcCI(y, id = id, visit = visit, demean = TRUE)$CI
print(pearson_ci)
print(pearson_i2c2$lambda)
