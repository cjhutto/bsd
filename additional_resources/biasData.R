### set the working directory
setwd("C:\\Users\\ch279\\PycharmProjects\\BSD\\bsd\\additional_resources\\")
#getwd() # print the current working directory
file = "collapsedData.csv"
csv = read.csv(file, header=TRUE, sep=",")

myData = data.frame(tID=csv$tID, 
                    avgBias=csv$avgBias,
                    subj_weak_rto=csv$subj_weak_rto,
                    liwc_discr_rto=csv$liwc_discr_rto,
                    factive_rto=csv$factive_rto,
                    subjectivity=csv$subjectivity,
                    liwc_work_rto=csv$liwc_work_rto,
                    modality=csv$modality,
                    liwc_aux_rto=csv$liwc_aux_rto,
                    mood=as.factor(csv$mood), 
                    hedge_rto=csv$hedge_rto,
                    liwc_prep_rto=csv$liwc_prep_rto,
                    liwc_3pp_rto=csv$liwc_3pp_rto,
                    fk_gl=csv$fk_gl,
                    dm_rto=csv$dm_rto,
                    vader_sentiment=csv$vader_sentiment,
                    liwc_tent_rto=csv$liwc_tent_rto,
                    assertive_rto=csv$assertive_rto,
                    opinion_rto=csv$opinion_rto,
                    implicative_rto=csv$implicative_rto,
                    liwc_achiev_rto=csv$liwc_achiev_rto,
                    cm_rto=csv$cm_rto,
                    liwc_cert_rto=csv$liwc_cert_rto,
                    bias_rto=csv$bias_rto,
                    subj_strong_rto=csv$subj_strong_rto,
                    liwc_causn_rto=csv$liwc_causn_rto,
                    liwc_adv_rto=csv$liwc_adv_rto,
                    liwc_conj_rto=csv$liwc_conj_rto)

### data analysis
summary(myData)


### Multiple (Linear) Regression model http://www.statmethods.net/stats/regression.html
library(MASS)
#initial model
fit_initial <- lm(avgBias ~ vader_sentiment + opinion_rto + dm_rto + cm_rto + 
            modality + subjectivity + #mood + 
            fk_gl + liwc_3pp_rto + liwc_causn_rto + 
            liwc_cert_rto + liwc_tent_rto + liwc_achiev_rto + liwc_discr_rto + 
            liwc_adv_rto + liwc_prep_rto + liwc_conj_rto +
            bias_rto + liwc_work_rto + liwc_aux_rto + liwc_prep_rto + 
            factive_rto + hedge_rto + assertive_rto + implicative_rto + 
            subj_strong_rto + subj_weak_rto, data=myData)
summary(fit_initial) # help interpret: http://blog.yhathq.com/posts/r-lm-summary.html
### Stepwise Regression for variable selection to the improved model
step <- stepAIC(fit_initial, direction="both")
step$anova # display results 

#improved model
fit <- lm(avgBias ~ vader_sentiment + opinion_rto + modality + liwc_3pp_rto + 
             liwc_tent_rto + liwc_achiev_rto + liwc_discr_rto + bias_rto + 
             liwc_work_rto + factive_rto + hedge_rto + assertive_rto + 
             subj_strong_rto + subj_weak_rto, data=myData)
summary(fit)
#layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page 
plot(fit) # diagnostics plots
plot(fitted(fit), myData$avgBias, xlab="Predicted Bias", ylab="Observed (Measured) Bias")

# Other useful functions
coefficients(fit) # model coefficients --> *unstandardized*, same as in summary(fit)
library(QuantPsyc)
lm.beta(fit) # **standardized** (beta) coefficients
confint(fit, level=0.95) # CIs for model parameters
fitted(fit) # predicted values
residuals(fit) # residuals --> diff. between observed and predicted: [=Obs-Pred]
anova(fit) # anova table
vcov(fit) # covariance matrix for model parameters
influence(fit) # regression diagnostics

# K-fold cross-validation
library(DAAG)
cv.lm(df=myData, fit, m=10, plotit = c("Observed","Residual")) # 10fold cross-validation

### Calculate Relative Importance for Each Predictor 
library(relaimpo)
#help(calc.relimp) 
crlm = calc.relimp(fit,type=c("last","first", "betasq"), diff=TRUE, rank=TRUE, 
                   rela=TRUE)
crlm
par(las=2)
plot(crlm)
cor(myData)

######## IRR
library(irr)
irrfile = "interrater.csv"
irrcsv = read.csv(irrfile, header=FALSE, sep=",")
icc(irrcsv, model="twoway", type="consistency", unit="single")
finn(irrcsv, s.levels=4, model="twoway")
