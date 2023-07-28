import os
os.environ["R_HOME"] = #REDACTED PATH
import numpy as np
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
import pandas as pd
import statsmodels.api as sm
import bioinfokit.analys
from mne.stats import fdr_correction
import researchpy as rp
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri, numpy2ri
numpy2ri.activate()
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro
import itertools
import statsmodels
from utilities import relabel
from data_processor import data_processor
from scipy.stats import sem
from scipy import stats

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

class Stats():

    def __init__(self, path, statsPath,dpOHC=None,dpnonOHC=None,dpOHCnonBin=None,dpnonOHCnonBin=None):
        self.path = path
        self.statsPath = statsPath
        self.ARL = dict()
        self.mediation = dict()
        self.mendelian = dict()
        self.linearRegression = dict()
        self.logisticRegression = dict()
        self.oneANOVA = dict()
        self.twoANOVA = dict()
        self.AssocAnalysis = dict()
        self.dpOHC=dpOHC
        self.dpnonOHC=dpnonOHC
        self.dpOHCnonBin=dpOHCnonBin
        self.dpnonOHCnonBin=dpnonOHCnonBin
    

    def Association_Rule_Learning(self, rhs, min_support = 0.00045, min_confidence = 0.02, min_items = 2, max_items = 5, min_lift = 2, protective = False):
        ''' 
        Do Association Rules mining for the items within the passed dataframe. Write all the found 
        association rules that meet the specified conditions and save the produced graphs
        in the passed parameters to AssociationRules.txt in Apriori Folder in the passed path.
        
        Args:
            data: DataFrame containing the items of interest
            min_support: Minimum value of support for the rule to be considered
            min_confidence: Minimum value of confidence for the rule to be considered
            min_items: Minimum number of item in the rules including both sides
            max_items: Maximum number of item in the rules including both sides
            rhs: Item to be on the right hand side -- outcome variable
            min_lift: Minimum value for lift for the rule to be considered
            protective: If True, the rhs values will be flipped to find protective features
        Returns:
            A dataframe of all the association rules found
        '''

        data = self.dpOHC.df.copy(deep = True)
        data.columns = data.columns.str.replace(' ', '_')
        columns=data.columns
        for feature in columns:
            if '.' in feature:
                data.drop([feature],axis=1,inplace=True)
        data.columns = data.columns.str.replace('.', '_')
        types=np.array(self.dpOHC.types)
        features=np.array(data.columns)
        continuous=list(features[np.where(types!='c')[0]])
        data.drop(labels=continuous,axis=1,inplace=True)

        for col in data.columns:
            if len(data[col].unique()) > 2:
                data.drop([col], axis = 1, inplace = True)
        
        print(data)
        if not os.path.exists(self.statsPath+'Apriori'):
            os.makedirs(self.statsPath+'Apriori')
        aprioriPath = self.statsPath + 'Apriori/'
        
        
        if protective:
            data = data.copy(deep = True)
            data[rhs] = np.absolute(np.array(data[rhs].values)-1)
            
            if not os.path.exists(aprioriPath+'Protective/'):
                os.makedirs(aprioriPath+'Protective/')
            currPath = aprioriPath+'Protective/'
        
        else:
            if not os.path.exists(aprioriPath+'Risk/'):
                os.makedirs(aprioriPath+'Risk/')
            currPath = aprioriPath+'Risk/'

        data.to_csv(currPath + 'AprioriData.csv')
        args = currPath + ' ' + str(min_support) + ' '  + str(min_confidence) + ' ' + str(max_items) + ' ' + str(min_items) + ' ' + str(rhs) + ' ' + str(min_lift)
        os.system('Rscript ' + self.path + 'Association_Rules.R ' + args)
        os.remove(currPath + 'AprioriData.csv') 
        if os.path.exists(currPath + 'apriori.csv'):
            ARLRules = pd.read_csv(currPath + 'apriori.csv')
            pvals = ARLRules['pValue']
            pvals = fdr_correction(pvals, alpha=0.05, method='indep')
            ARLRules['adj pVals'] = pvals[1]
        else:
            print('No rules meeting minimum requirements were found')
            print('Process Terminated')
            return
        os.remove(currPath + 'apriori.csv') 

        vars = ARLRules['LHS'].tolist()
        features, newF, rows, pvals = list(), list(), list(), list()
        oddsRatios = pd.DataFrame(columns=['LHS-RHS', 'Odds Ratio', 'Confidence Interval', 'pValue', 'adjusted pVal'])
        for var in vars:
            newF.append(var)
            features.append(var.replace('{', '').replace('}', '').split(','))
        for i in range(len(features)):
            cols = features[i]
            newFeature = newF[i]
            dataC = data.drop([x for x in data.columns if x not in cols], axis = 1)
            dataC[newFeature] = dataC[dataC.columns[:]].apply(lambda x: ','.join(x.astype(str)),axis=1)
            dataC = dataC[[newFeature]]
            dataC[rhs] = data[rhs]
            toDrop = list()
            for index, r in dataC.iterrows():                
                fValue = set(r[newFeature].split(','))
                if (len(fValue) > 1):
                    toDrop.append(index)
            dataC.drop(toDrop, inplace = True)
            dataTrue = dataC[dataC[rhs] == 1].drop([rhs], axis =1).value_counts().tolist()
            dataFalse = dataC[dataC[rhs] == 0].drop([rhs], axis = 1).value_counts().tolist()
            if len(dataTrue) == 1:
                dataTrue.append(0)
            if len(dataFalse) == 1:
                dataFalse.append(0)
            dataTrue.reverse(); dataFalse.reverse()
            
            table = np.array([dataTrue, dataFalse])
            print(table)
            res = statsmodels.stats.contingency_tables.Table2x2(table, shift_zeros = True)

            rows.append([str(newFeature)+'-'+str(rhs), 
            res.oddsratio, res.oddsratio_confint(), res.oddsratio_pvalue()])
            pvals.append(res.oddsratio_pvalue())
            
        pvals = fdr_correction(pvals, alpha=0.05, method='indep')
        for i in range(len(pvals[1])):
            rows[i].append(pvals[1][i])
            oddsRatios.loc[len(oddsRatios.index)] = rows[i]
            
            

        Association = open(currPath + 'AssociationRules.txt', 'w')
        Association.write(ARLRules.to_string(index=False))
        Association.write('\n\n----------------------------------------------------------------------------------------\n\n')
        Association.write('\n\nOdds Ratio analysis for Association Rule Learning: \n----------------------------------------------------------------------------------------\n\n')
        for i in range(len(oddsRatios)):
            two_var = oddsRatios.iloc[i, :]
            two_var = two_var.to_frame()
            variables = str(two_var.iloc[0,0]).split('-')
            two_var = two_var.iloc[1: , :]
            Association.write('The odds ratio, p-Value, and confidence interval between ' +variables[0]+' and ' + variables[1] + ' are: \n\n')
            toWrite = two_var.to_string(header = False, index = True)
            Association.write(toWrite+'\n')
            Association.write('----------------------------------------------------------------------------------------\n\n')

        Association.close()
        if protective:
            self.ARL['ARLProtect'] = ARLRules
        else:
            self.ARL['ARLRisk'] = ARLRules

        
        
        return ARLRules


    def Mediation_Analysis(self, dep, mediator, indep, sims=1000):
        '''
        Do Mediation Analysis between the passed dependent & indepent variables 
        and the mediation variable(s) passed in the passed data frame. 
        Write the results to Mediation_Analysis.txt in the passed path
        
        Args:
            data: DataFrame containing the items of interest
            dep: The dependent varaible in the analysis
            mediator: The mediation variable in the analysis
            indep: The independent variable(s) in the analysis - can be a list
            continuous: list containing continuous variables
        '''
        data = self.dpOHCnonBin.df.copy(deep = True)
        data.columns = data.columns.str.replace(' ', '_')
        data.columns = data.columns.str.replace('.', '_')
        types=np.array(self.dpOHCnonBin.types)
        features=np.array(data.columns)
        continuous=list(features[np.where(types!='c')[0]])


        if not os.path.exists(self.statsPath + 'MediationAnalysis/'):
            os.makedirs(self.statsPath +'MediationAnalysis/')
        currPath = self.statsPath + 'MediationAnalysis/'

        if type(indep) == str:
            t = list(); t.append(indep)
            indep = t

        for var in indep:
            filePath = currPath+'MedAnalysis-'+str(var) + '-' + str(mediator) + '-' + str(dep) + '.txt'

            l1 = importr('mediation')
            formulaMake = r['as.formula']
            mediate, lm, glm, summary, capture = r['mediate'], r['lm'], r['glm'], r['summary'], r['capture.output']

            MediationFormula = formulaMake(mediator + ' ~ ' + var)
            OutcomeFormula = formulaMake(dep + ' ~ ' + var + ' + ' + mediator)

            with localconverter(ro.default_converter + pandas2ri.converter):
                data = ro.conversion.py2rpy(data)

            if mediator in continuous:
                modelM = lm(MediationFormula, data) 
            else:
                modelM = glm(MediationFormula, data = data, family = "binomial")
            
            if dep in continuous:
                modelY = lm(OutcomeFormula, data)
            else:
                modelY = glm(OutcomeFormula, data = data, family = "binomial")
            
            results = mediate(modelM, modelY, treat=var, mediator=mediator,sims=sims)
            dfR = summary(results)
            self.mediation['results'] = dfR
            capture(dfR, file = filePath)



    def Mendelian_Ranomization(self, dep, indep, inst, control = list()):
        '''
        Conduct Mendelian Ranomization analysis using 2-Stage-Least-Square between 
        the dependent and independent variables passed based on the instrumental variables 
        sepcified. Write the results to Mendelian_Ranomization.txt in the specified path
        
        Args:
            data: DataFrame containing the items of interest
            dep: the dependent varaible in the analysis
            indep: the independent variable in the analysis
            control: the control variable(s) in the analysis -- can be a list
            inst: the instrumental variables to be used in the analysis
            path: Folder path to which the data will be saved

        Returns:
            an IV2SLS object containing the 2SLS analysis results
        '''
        if not os.path.exists(self.statsPath + 'MendelianRandomization'):
            os.makedirs(self.statsPath +'MendelianRandomization')
        currPath = self.statsPath + 'MendelianRandomization/'
        
        MendAnalysis = open(currPath + 'Mendelian_Ranomization.txt', 'w')
        MendAnalysis.write('Mendelian Randomization Analysis Results: \n----------------------------------------------------------------------------------------\n\n')
        data = self.dpOHCnonBin.df.copy(deep = True)
        print(data.columns)
        data.columns = data.columns.str.replace(' ', '_')
        data.columns = data.columns.str.replace('.', '_')

        if type(control) == str:
            t = list()
            t.append(control)
            control = t

        if type(inst) == str:
            t = list()
            t.append(inst)
            inst = t

        formula = dep + ' ~'
        for var in control:
            formula += ' + C(' + str(var) +')'
        formula += ' + '
        for var in inst:
            formula += var + ' + '
        formula = formula[:-2]

        #Checking the first assumption -- is the instrument affecting the modifiable behavior?

        first_stage = smf.ols(formula, data=data).fit()
        for var in inst:
            MendAnalysis.write( var + " parameter estimate:, " + str(first_stage.params[var])+ '\n') 
            MendAnalysis.write(var + " p-value:, " + str(first_stage.pvalues[var]) + '\n')
            MendAnalysis.write('----------------------------------------------------------------------------------------\n')

        def parse(model, exog):
            param = model.params[exog]
            se = model.std_errors[exog]
            p_val = model.pvalues[exog]
            MendAnalysis.write(f"Parameter: {param}\n")
            MendAnalysis.write(f"SE: {se}\n")
            MendAnalysis.write(f"95 CI: {(-1.96*se,1.96*se) + param}\n")
            MendAnalysis.write(f"P-value: {p_val}\n")
            MendAnalysis.write('----------------------------------------------------------------------------------------\n')

        #Conducting Mendelian_Ranomization using 2 stage least square

        formula = dep + ' ~'
        for var in control:
            formula += ' + C(' + str(var) +')'
        formula += ' + [' + str(indep) + ' ~ '
        for var in inst:
            formula += var + '+'
        formula = formula[:-1] + ']'


        iv2sls = IV2SLS.from_formula(formula, data).fit()
        parse(iv2sls, exog = indep)
        MendAnalysis.close()

        self.mendelian['results'] = iv2sls
        return iv2sls


    def Multivariate_Reg(self, target, correct = False, cutoff = 10, stepwise = True, VIF = True):
        
        '''
        Conduct Multivariate Regression analysis between the target and independent variables
        passed. Write the results to Maltivariate_Reg.txt in statisticalAnalysis
        
        Args:
            data: DataFrame containing the items of interest
            target: the outcome varaible of interest in the analysis
            correct: boolean variable. If True, variables with high VIF value would be dropped
            cutoff: cutoff value to drop variables based on VIF.
            stepwise: if True, conduct stepwise ajdustment. Otherwise, the function won't.
            VIF: if True, conduct VIF check for multicolinearity over continuous variables
                if continuous varaibels are fewer than two, the check won't be conducted
            continuous: list containing continuous variables
            categorical: list of categorical variables

        Returns:
            The regression model before and after stepwise if stepwise is True
        '''
        data = self.dpOHCnonBin.df.copy(deep = True)
        data.columns = data.columns.str.replace(' ', '_')
        data.columns = data.columns.str.replace('.', '_')
        types=np.array(self.dpOHCnonBin.types)
        features=np.array(data.columns)
        continuous=list(features[np.where(types!='c')[0]])
        categorical=list(np.setdiff1d(features,continuous))

        indep = list(data)
        indep.remove(target)
        
        if not os.path.exists(self.statsPath+'MultiVarRegression'):
            os.makedirs(self.statsPath+'MultiVarRegression')
        currPath = self.statsPath + 'MultiVarRegression/'

        MultiReg = open(currPath + 'MultivariateRegression.txt', 'w')


        #Check for linear relationship between features and target

        contPath = currPath+'ContinousVariablesCheck/'
        if not os.path.exists(contPath):
            os.makedirs(contPath)

        for var in continuous:
            xvalues, yvalues = list(), list()
            for j in range(0,100,1):
                i=j/100
                newdf=data.iloc[np.intersect1d(np.where(data[var]>i)[0],np.where(data[var]<i+0.01)[0]),:]
                xvalues.append(i+0.005)
                yvalues.append(np.mean(newdf[target]))

            plt.bar(xvalues,height=yvalues,width=0.005, linewidth = 1, edgecolor = '#000000')
            plt.xlabel(var)
            plt.ylabel(target)
            plt.savefig(contPath+var+'_barPlot.png')
            plt.close()

            sns.lmplot(x=var, y=target, data=data.sample(1000, random_state = 42), order=1)
            plt.ylabel(target)
            plt.xlabel(var)
            plt.savefig(contPath+var+'_scatterPlot.png')
            plt.close()

        catPath = currPath+'CategoricalVariablesCheck/'
        if not os.path.exists(catPath):
            os.makedirs(catPath)

        for var in categorical:
            xvalues=[]
            yvalues=[]
            stds=[]
            for i in range(data[var].nunique()):
                newdf=data.iloc[np.where(data[var]==i)[0],:]
                xvalues.append(i)
                yvalues.append(np.mean(newdf[target]))
                stds.append(2*np.std(newdf[target])/np.sqrt(len(newdf)))
            plt.bar(xvalues,height=yvalues,width=0.5)
            plt.errorbar(xvalues,yvalues,yerr=stds,fmt='o',markersize=1,capsize=8,color='r')
            plt.xlabel(var)
            plt.ylabel(target)
            plt.savefig(catPath+var+'_barPlot.png')
            plt.close()

            sns.lmplot(x=var, y=target, data=data.sample(1000, random_state = 42), order=1)
            plt.ylabel(target)
            plt.xlabel(var)
            plt.savefig(catPath+var+'_scatterPlot.png')
            plt.close()
            

        #Check for Multicolinearity
        if VIF and len(continuous) > 1:
            vif_data = pd.DataFrame()
            VIF_values = continuous

            vif_data["feature"] = VIF_values
            vif_data["VIF"] = [variance_inflation_factor(data[VIF_values].values, i)
                                for i in range(len(VIF_values))]

            self.linearRegression['VIF'] = vif_data
            toDrop = list()        
            if correct:
                for i in range(len(vif_data)):
                    if vif_data.iloc[i, 1] > cutoff:
                        toDrop.append(vif_data.iloc[i,0])

                data.drop(toDrop, axis = 1, inplace = True)

            vifString = vif_data.to_string(header=True, index=False)
            
            MultiReg.write('VIF values without correction to check multicollinearity: \n\n')
            if not VIF_values:
                MultiReg.write('VIF check for Multicolinearity was not conducted'+'\n')
            else:
                MultiReg.write(vifString+'\n')
            MultiReg.write('----------------------------------------------------------------------------------------\n\n')

            if correct:
                for i in toDrop:
                    vif_data = vif_data[vif_data.feature != i]
                    indep.remove(i)
                    data.drop([i], axis = 1, inplace = True)
                vifString = vif_data.to_string(header=True, index=False)
                MultiReg.write('VIF values after correction to check multicollinearity: \n\n')
                MultiReg.write(vifString+'\n')
                MultiReg.write('----------------------------------------------------------------------------------------\n')
            

        #Conduct regression analysis before stepwise adjustment
        X = sm.add_constant(data.drop([target], axis = 1)) 
        SMF_model = sm.OLS(endog= data[target], exog = X).fit()
        self.linearRegression['linearBeforeStepwise'] = SMF_model
        pvalues = SMF_model.pvalues
        corrected_p = fdr_correction(pvalues, alpha=0.05, method='indep')
        self.linearRegression['adjustedP_BeforeStepwise'] = corrected_p
        
        MultiReg.write('Multivariate Regression results before stepwise adjustment: \n\n')
        MultiReg.write(SMF_model.summary().as_text() + '\n\n')
        MultiReg.write('----------------------------------------------------------------------------------------\n')
        cp = [str(a) for a in corrected_p[1]]
        indep.insert(0, 'intercept')
        adjustedP = pd.DataFrame(cp, indep)
        MultiReg.write('adjusted p-values for Regression before stepwise adjustment: \n\n')
        MultiReg.write(adjustedP.to_string(header=False, index=True) + '\n\n')
        MultiReg.write('----------------------------------------------------------------------------------------\n')


        #Histogram & QQplot to test for normality before stepwise

        histo = sns.histplot(SMF_model.resid)
        fig = histo.get_figure()
        fig.savefig(currPath+"MultiRegHistogram-noStepwise.png")
        plt.close()
        fig, ax = plt.subplots(1, 1)
        sm.ProbPlot(SMF_model.resid).qqplot(line='s', color='#1f77b4', ax=ax)
        ax.title.set_text('QQ Plot')
        plt.savefig(currPath+"MultiRegQQPlot-noStepwise.png")
        plt.close()

        #Check for outliers before stepwise adjustment

        np.set_printoptions(suppress=True)
        influence = SMF_model.get_influence()
        cooks = influence.cooks_distance

        plt.scatter(np.arange(start=0, stop=len(cooks[0]), step=1), cooks[0])
        plt.xlabel('participant')
        plt.ylabel('Cooks Distance')
        plt.savefig(currPath+'MultiRegCooksDist-noStepwise.png')
        plt.close()
        

        #Check for Homoscedasticity using scale-location plot before stepwise adjustment
        y_predict = SMF_model.predict(X)

        fig, ax = plt.subplots(1, 1)
        
        sns.residplot(data[target],y_predict, lowess = True, scatter_kws={'alpha':0.5},
        line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
        ax.title.set_text('Scale Location')
        ax.set(xlabel='Fitted', ylabel='Standardized Residuals')
        plt.savefig(currPath+"MultiRegScaleLocation-noStepwise.png")
        plt.close()

        if stepwise:


            data.to_csv(currPath+'StepWiseRegData.csv')
            args = currPath + ' ' + str('linear') + ' ' + target
            os.system('Rscript ' + self.path + 'StepWiseRegression.R ' + args)
            os.remove(currPath + 'StepWiseRegData.csv')
            file = open(currPath+ 'stepWiseVars.txt')
            newVars = file.read().split()
            file.close()
            os.remove(currPath + 'stepWiseVars.txt')

            if '(Intercept)' in newVars:
                newVars.remove('(Intercept)')
            
            data = data[newVars + [target]]
        
            #Conduct regression analysis after stepwise adjustment

            X2 = sm.add_constant(data.drop([target], axis = 1)) 
            SMF_model2 = sm.OLS(endog= data[target], exog = X2).fit()
            pvalues2 = SMF_model2.pvalues
            corrected_p2 = fdr_correction(pvalues2, alpha=0.05, method='indep')
            self.linearRegression['linearAfterStepwise'] = SMF_model2
            self.linearRegression['adjustedP_AfterStepwise'] = corrected_p2

            MultiReg.write('Multivariate Regression resluts after stepwise adjustment: \n\n')
            MultiReg.write(SMF_model2.summary().as_text() + '\n\n')
            MultiReg.write('----------------------------------------------------------------------------------------\n')
            cp2 = [str(a) for a in corrected_p2[1]]
            newVars.insert(0, 'intercept')
            adjustedP2 = pd.DataFrame(cp2, newVars)
            MultiReg.write('adjusted p-values for Regression after stepwise adjustment: \n\n')
            MultiReg.write(adjustedP2.to_string(header=False, index=True) + '\n\n')
            MultiReg.write('----------------------------------------------------------------------------------------\n')


            #Histogram to test for normality after stepwise
            
            histo = sns.histplot(SMF_model2.resid)
            fig = histo.get_figure()
            fig.savefig(currPath+"MultiRegHistogram-Stepwise.png")
            plt.close()
            fig, ax = plt.subplots(1, 1)
            sm.ProbPlot(SMF_model2.resid).qqplot(line='s', color='#1f77b4', ax=ax)
            ax.title.set_text('QQ Plot')
            plt.savefig(currPath+"MultiRegQQPlot-Stepwise.png")
            plt.close()

            #Check for Homoscedasticity using scale-location plot after stepwise adjustment

            y_predict2 = SMF_model2.predict(X2)
            fig, ax = plt.subplots(1, 1)
            sns.residplot(data[target], y_predict2, lowess=True, scatter_kws={'alpha':0.5},
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
            ax.title.set_text('Scale Location')
            ax.set(xlabel='Fitted', ylabel='Standardized Residuals')
            plt.savefig(currPath+"MultiRegScaleLocation-Stepwise.png")
            plt.close()

            #Check for outliers after stepwise adjustment

            np.set_printoptions(suppress=True)
            influence = SMF_model2.get_influence()
            cooks = influence.cooks_distance

            plt.scatter(np.arange(start=0, stop=len(cooks[0]), step=1), cooks[0])
            plt.xlabel('participant')
            plt.ylabel('Cooks Distance')
            plt.savefig(currPath+'MultiRegCooksDist-Stepwise.png')
            plt.close()

            MultiReg.close()
            return SMF_model, SMF_model2
        
        MultiReg.close()
        return SMF_model

    def UniVariateLogisitc(self, X, y, uniPath, continuous, categorical):

        indep = X.name

        currPath = uniPath
        univariate = open(currPath +'UnivariateLogisticRegression.txt', 'a')

        X = sm.add_constant(X)
        log_reg = GLM(y, X, family=families.Binomial()).fit()
        model_odds = pd.DataFrame(np.exp(log_reg.params), columns= ['Odds Ratio'])
        model_odds['p-value']= log_reg.pvalues
        model_odds[['2.5%', '97.5%']] = np.exp(log_reg.conf_int())

        univariate.write('Odds Ratios for univariate regression between ' + indep + ' and ' + y.name + ': \n\n')
        toWrite = model_odds.to_string(header = True, index = True)
        univariate.write(toWrite+'\n')
        univariate.write('----------------------------------------------------------------------------------------\n\n')
        univariate.close()

        log_reg.pvalues

        if indep in continuous or indep in categorical:
            predict = log_reg.predict(X)
            predict = np.log(predict)
            plt.scatter(X[indep], predict)
            plt.xlabel(indep)
            plt.ylabel('log odds of ' + str(y.name))
            plt.savefig(currPath+str(indep)+'-logOdds.png')
            plt.close()
        
        return log_reg.pvalues[1]


    def Odds_Ratios(self, target, correct = False, cutoff = 10, stepwise = True, VIF = True):

        '''
        Calculates the odds ratio for the signifcant variables and write the results to Odds_Ratios.txt
        in statisticalAnalysis. 
        
        Args:
            data: DataFrame containing the items of interest
            target: the dependent varaible in the analysis
            correct: boolean variable. If True, variables with high VIF value would be dropped
            cutoff: Integer value for the cutoff value for VIF
            stepwise: if True, conduct stepwise ajdustment. Otherwise, the function won't.
            VIF: if True, conduct VIF check for multicolinearity over continuous variables
                if continuous varaibels are fewer than two, the check won't be conducted
            continuous: list containing continuous variables

        Returns:
            two np arrays containing the odds ratio as well as teh confidence intervals before
            and after doing stepwise adjustement if stepwise was True.
        '''
        data = self.dpOHC.df.copy(deep = True)
        data.columns = data.columns.str.replace(' ', '_')
        data.columns = data.columns.str.replace('.', '_')
        types=np.array(self.dpOHC.types)
        features=np.array(data.columns)
        continuous=list(features[np.where(types!='c')[0]])
        categorical=list(np.setdiff1d(features,continuous))

        indep = list(data)
        indep.remove(target)


        if not os.path.exists(self.statsPath+'OddsRatios'):
            os.makedirs(self.statsPath+'OddsRatios')
        currPath = self.statsPath + 'OddsRatios/'

        oddsRatio = open(currPath + 'Odds_Ratios.txt', 'w')

        #Checking for Multicolinearity
        
        if VIF and len(continuous) > 1:
            vif_data = pd.DataFrame()
            VIF_values = continuous

            vif_data["feature"] = VIF_values
            vif_data["VIF"] = [variance_inflation_factor(data[VIF_values].values, i)
                                for i in range(len(VIF_values))]
            
            self.logisticRegression['VIF'] = vif_data
            toDrop = list()        
            if correct:
                for i in range(len(vif_data)):
                    if vif_data.iloc[i, 1] > cutoff:
                        toDrop.append(vif_data.iloc[i,0])

                data.drop(toDrop, axis = 1, inplace = True)

            vifString = vif_data.to_string(header=True, index=False)
            
            oddsRatio.write('VIF values without correction to check multicollinearity: \n\n')
            oddsRatio.write(vifString+'\n')
            oddsRatio.write('----------------------------------------------------------------------------------------\n')

            if correct:
                for i in toDrop:
                    vif_data = vif_data[vif_data.feature != i]
                    indep.remove(i)
                    data.drop([i], axis = 1, inplace = True)
                vifString = vif_data.to_string(header=True, index=False)
                oddsRatio.write('VIF values after correction to check multicollinearity: \n\n')
                oddsRatio.write(vifString+'\n')
                oddsRatio.write('----------------------------------------------------------------------------------------\n')
            
            
        #Find odds ratios before stepwise adjustment
        X = sm.add_constant(data.drop([target], axis = 1))
        log_reg = GLM(data[target], X, family=families.Binomial()).fit()
        model_odds = pd.DataFrame(np.exp(log_reg.params), columns= ['Odds Ratio'])
        model_odds['p-value']= log_reg.pvalues
        model_odds[['2.5%', '97.5%']] = np.exp(log_reg.conf_int())
        self.logisticRegression['OddsBeforeStepwise'] = model_odds


        oddsRatio.write('Odds Ratios before stepwise adjustment: \n\n')
        toWrite = model_odds.to_string(header = True, index = True)
        oddsRatio.write(toWrite+'\n')
        oddsRatio.write('----------------------------------------------------------------------------------------\n')
        pvalues = log_reg.pvalues
        corrected_p = fdr_correction(pvalues, alpha=0.05, method='indep')
        cp = [str(a) for a in corrected_p[1]]
        indep.insert(0, 'intercept')
        adjustedP = pd.DataFrame(cp, indep)
        self.logisticRegression['adjustedP_BeforeStepwise'] = adjustedP
        oddsRatio.write('adjusted p-values for Regression before stepwise adjustment: \n\n')
        oddsRatio.write(adjustedP.to_string(header=False, index=True) + '\n\n')
        oddsRatio.write('----------------------------------------------------------------------------------------\n')    
        indep.remove('intercept')
        #Checking for outliers before stepwise adjustement

        infl = log_reg.get_influence(observed=False)

        np.set_printoptions(suppress=True)
        cooks = infl.cooks_distance
        plt.scatter(np.arange(start=0, stop=len(cooks[0]), step=1), cooks[0])
        plt.xlabel('participant')
        plt.ylabel('Cooks Distance')
        plt.savefig(currPath+'LogitRegCooksDistance-noStepwise.png')
        plt.close()

        uniPath = currPath+ 'UnivariateLogisticRegression/'
        if not os.path.exists(uniPath):
            os.makedirs(uniPath)
        open(uniPath +'UnivariateLogisticRegression.txt', 'w').close()

        if stepwise: 
            data.to_csv(currPath+'StepWiseRegData.csv')
            args = currPath + ' ' + str('logit') + ' ' + target
            os.system('Rscript ' + self.path + 'StepWiseRegression.R ' + args)
            os.remove(currPath+'StepWiseRegData.csv')
            file = open(currPath+'stepWiseVars.txt')
            newVars = file.read().split()
            file.close()
            os.remove(currPath+'stepWiseVars.txt')
            if '(Intercept)' in newVars:
                newVars.remove('(Intercept)')
            
            data = data[newVars + [target]]

            #Find odds ratios after stepwise adjustment

            X2 = sm.add_constant(data.drop([target], axis = 1))
            log_reg2 = GLM(data[target], X2, family=families.Binomial()).fit()

            model_odds2 = pd.DataFrame(np.exp(log_reg2.params), columns= ['Odds Ratio'])
            model_odds2['p-value']= log_reg2.pvalues
            model_odds2[['2.5%', '97.5%']] = np.exp(log_reg2.conf_int())
            self.logisticRegression['OddsAfterStepwise'] = model_odds2

            oddsRatio.write('Odds Ratios after stepwise adjustment: \n\n')
            toWrite = model_odds2.to_string(header = True, index = True)
            oddsRatio.write(toWrite+'\n')
            oddsRatio.write('----------------------------------------------------------------------------------------\n')
            pvalues2 = log_reg2.pvalues
            corrected_p2 = fdr_correction(pvalues2, alpha=0.05, method='indep')
            cp2 = [str(a) for a in corrected_p2[1]]
            newVars.insert(0, 'intercept')
            adjustedP2 = pd.DataFrame(cp2, newVars)
            self.logisticRegression['adjustedP_AfterStepwise'] = adjustedP2
            oddsRatio.write('adjusted p-values for Regression after stepwise adjustment: \n\n')
            oddsRatio.write(adjustedP2.to_string(header=False, index=True) + '\n\n')
            oddsRatio.write('----------------------------------------------------------------------------------------\n')    


            #Checking for outliers after stepwise adjustement

            infl2 = log_reg2.get_influence(observed=False)

            cooks2 = infl2.cooks_distance
            plt.scatter(np.arange(start=0, stop=len(cooks[0]), step=1), cooks2[0])
            plt.xlabel('participant')
            plt.ylabel('Cooks Distance')
            plt.savefig(currPath+'LogitRegCooksDistance-Stepwise.png')
            plt.close()

            oddsRatio.close()
            return model_odds, model_odds2
        
        
        pVal = list()
        for var in indep:
            pVal.append(self.UniVariateLogisitc(data[var], data[target], uniPath, continuous, categorical))
        
        corrected_p = fdr_correction(pVal, alpha=0.05, method='indep')
        cp = [str(a) for a in corrected_p[1]]

        adjustedPuni = pd.DataFrame(cp, indep)
        univariate = open(uniPath +'UnivariateLogisticRegression.txt', 'a')
        univariate.write('P-Values after adjustement for univariate regression: \n\n')
        toWrite = adjustedPuni.to_string(header = True, index = True)
        univariate.write(toWrite+'\n')


        oddsRatio.close()
        return model_odds


    def oneWay_ANOVA(self, data, dep, indep, alpha, between, followUp):
        results = dict()
        if not os.path.exists(self.statsPath+'oneWayANOVA'):
            os.makedirs(self.statsPath+'oneWayANOVA')

        print(indep)
        #new_indep=self.relabel(indep[:indep.find('_oneVsRest')+1]+indep[indep.find('oneVsRest')+len('oneVsRest')+1:])
        new_indep=relabel(indep)
        print(new_indep)
        #new_indep=new_indep[:new_indep.rfind('_')+1]+'oneVsRest'+new_indep[new_indep.rfind('_'):]
        data2=data.rename(columns={indep:new_indep},inplace=False)
        indep=new_indep

        if not os.path.exists(self.statsPath+'oneWayANOVA/'+indep):
            os.makedirs(self.statsPath+'oneWayANOVA/'+indep)
        currPath = self.statsPath+'oneWayANOVA/'+indep+'/'

        formula = dep + ' ~ C(' + indep + ')'
        oneWayANOVA = open(currPath+'oneWayANOVA_summary.txt', 'w')
        oneWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
        
        #Create a box plot for outliers detection
        
        data = data2[[indep, dep]]
        colors = ['#808080']
        box = sns.boxplot(x=indep, y=dep, data=data, palette=colors)
        fig = box.get_figure()
        fig.savefig(currPath+"oneWayANOVA_boxPlot.png")
        plt.close()
        

        #Create a bar plot

        sns.set(rc = {'figure.figsize':(15,10)})
        sns.set(font_scale = 1.5)
        sns.set_style('whitegrid')
        fig, bar = plt.subplots()
        
        colors = ['#808080']
        
        if data2[indep[0]].isin(['other']).any():
            for i in range(len(data2)):
                if data2[indep[0]].iat[i]!='other':
                    data2[indep[0]].iat[i]=indep[0][indep[0].rfind('_')+1:]

        sns.barplot(x=indep, ax = bar, y=dep, data=data2, palette = colors, capsize=.1)
        width = 0.3

        num_var2 = len(data[indep].unique())
        hatches = itertools.cycle(['+', 'x', '-', '/', '//', '\\', '*', 'o', 'O', '.'])

        '''
        for i, patch in enumerate(bar.patches):
            # Set a different hatch for each bar
            if i % num_var2 == 0:
                hatch = next(hatches)
            patch.set_hatch(hatch)
        '''

        for patch in bar.patches:
            current_width = patch.get_width()
            diff = current_width - width
            patch.set_width(width)
            patch.set_x(patch.get_x() + diff * .5)
            patch.set_edgecolor('#000000')
        fig = bar.get_figure()
        fig.savefig(currPath+"oneWayANOVA_barPlot.png")
        plt.close()

        #Conducting the ANOVA test
        oneWayANOVA.write('Results for one way ANOVA between ' + indep + ' and ' + dep + ' are: \n\n')
        res = bioinfokit.analys.stat()
        res.anova_stat(df=data, res_var=dep, anova_model=formula)
        asummary = res.anova_summary.to_string(header=True, index=True)
        oneWayANOVA.write(asummary + '\n')
        oneWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
        results['ANOVA_Results'] = res.anova_summary

        #Follow-up Test TukeyTest
        if (res.anova_summary.iloc[0,4] > alpha) and (not followUp):
            oneWayANOVA.write('The p-value is higher than alpha; hence, no follow-up test was conducted\n')
        else:
            oneWayANOVA.write('Results for follow-up Tukey test between ' + indep + ' and ' + dep + ' are: \n\n')
            if len(data[indep].value_counts()) <= 2:
                oneWayANOVA.write('Only two groups. No follow up test was conducted\n')
            else:
                followUp = bioinfokit.analys.stat()
                followUp.tukey_hsd(df=data, res_var=dep, xfac_var=indep, anova_model=formula)
                fSummary = followUp.tukey_summary.to_string(header=True, index=True)
                oneWayANOVA.write(fSummary + '\n')
                results['Tukey_Results'] = followUp.tukey_summary
        oneWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
            

        #histograms and QQ-plot for Normality detection
        sm.qqplot(res.anova_std_residuals, line='45')
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Standardized Residuals")
        plt.savefig(currPath+'oneWayANOVA_qqPlot.png')
        plt.close()

        plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k') 
        plt.xlabel("Residuals")
        plt.ylabel('Frequency')
        plt.savefig(currPath+'oneWayANOVA_histogram.png')
        plt.close()

        #Shapiro-Wilk Test for Normality
        w, pvalue = stats.shapiro(res.anova_model_out.resid)
        oneWayANOVA.write('Results for Shapiro-Wilk test to check for normality are: \n\n')
        oneWayANOVA.write('w is: ' + str(w) + '/ p-value is: ' + str(pvalue) + '\n')
        oneWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
        results['Shaprio-Wilk_Results'] = (w, pvalue)

        #Check for equality of varianve using Levene's test
        oneWayANOVA.write("Results for Levene's test to check for equality of variance are: \n\n")
        eqOfVar = bioinfokit.analys.stat()
        eqOfVar.levene(df=data, res_var=dep, xfac_var=indep)
        eqOfVarSummary = eqOfVar.levene_summary.to_string(header=True, index=False)
        oneWayANOVA.write(eqOfVarSummary + '\n')
        oneWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
        results['Levene_Results'] = eqOfVar.levene_summary

        #The Kruskal-Wallis H test
        groups = list()
        vals = data[indep].unique()
        for val in vals:
            g = data.loc[data[indep] == val]
            g = g.loc[:, [dep]].squeeze().tolist()
            groups.append(g)

        Kruskal = stats.kruskal(*groups)
        oneWayANOVA.write('Results for the Kruskal-Wallis Test -- to be used if ANOVA assumptions are violated: \n\n')
        oneWayANOVA.write('statistic: ' + str(Kruskal[0]) + '/ p-value is: ' + str(Kruskal[1]) + '\n')
        oneWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
        results['Kruskal-Wallis_Results'] = Kruskal

        
        #The dunn's test -- follow up
        if (Kruskal[1] > alpha) and (not followUp):
            oneWayANOVA.write('The p-value is higher than alpha; hence, no follow-up test was conducted for Kruskal test\n')    
        else:
            oneWayANOVA.write("Results for follow-up Dunn's test between " + indep + " and " + dep + " are: \n\n")
            if len(data[indep].value_counts()) <= 2:
                oneWayANOVA.write('Only two groups. No follow up test was conducted\n')
            else:
                FSA = importr('FSA')
                dunnTest, formulaMaker, names = r['dunnTest'], r['as.formula'], r['names']
                with localconverter(ro.default_converter + pandas2ri.converter):
                    rDf = ro.conversion.py2rpy(data)

                formula = formulaMaker(dep + ' ~ ' + indep)
                dunnTwoWay = dunnTest(formula, data=rDf, method="bonferroni")

                asData, doCall, rbind = r['as.data.frame'], r['do.call'], r['rbind']
                dunnTwoWay = asData(doCall(rbind, dunnTwoWay))

                with localconverter(ro.default_converter + pandas2ri.converter):
                    dunnTwoWay = ro.conversion.rpy2py(dunnTwoWay)

                dunnTwoWay.drop(['method', 'dtres'], inplace = True)

                for col in ['Z', 'P.unadj', 'P.adj']:
                    dunnTwoWay[col] = pd.to_numeric(dunnTwoWay[col])
                    dunnTwoWay[col] = np.round(dunnTwoWay[col], decimals = 5)
                
                dunnSummary = dunnTwoWay.to_string(header=True, index=False)
                oneWayANOVA.write(dunnSummary + '\n\n')
        
        oneWayANOVA.write('----------------------------------------------------------------------------------------\n\n')

        self.oneANOVA = results
        return results


    def twoWay_ANOVA(self, data, dep, indep, alpha, between, followUp, OGdata):
            
        results = dict()
        if len(data[indep[0]].value_counts()) < len(data[indep[1]].value_counts()):
            temp = indep.pop(0)
            indep.append(temp)
        elif len(data[indep[0]].value_counts()) == len(data[indep[1]].value_counts()):
            temp1 = indep.pop(0)
            temp2 = indep.pop(0)
            if len(temp1) <= len(temp2):
                indep.append(temp2); indep.append(temp1)
            else:
                indep.append(temp1); indep.append(temp2)
        
        print(indep)
        new_indep=relabel(indep[0])
        data=data.rename(columns={indep[0]:new_indep},inplace=False)
        OGdata=OGdata.rename(columns={indep[0][:indep[0].find('_')]:relabel(indep[0][:indep[0].find('_')]),indep[0][indep[0].find('_')+1:indep[0].rfind('_')]:relabel(indep[0][indep[0].find('_')+1:indep[0].rfind('_')])},inplace=False)
        indep=[new_indep,indep[1]]
        print(indep)

        if not os.path.exists(self.statsPath+'twoWayANOVA'):
            os.makedirs(self.statsPath+'twoWayANOVA')
        fname = indep[0] + '_' + indep[1]
        if not os.path.exists(self.statsPath+'twoWayANOVA/'+fname):
            os.makedirs(self.statsPath+'twoWayANOVA/'+fname)
        currPath = self.statsPath+'twoWayANOVA/'+fname+'/'

        formula = dep + ' ~ C(' + indep[0] + ') + C(' + indep[1] + ') + C(' + indep[0] + '):C(' + indep[1] + ')'
        twoWayANOVA = open(currPath+'twoWayANOVA_summary.txt', 'w')
        twoWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
            
        #Create a box plot
        data = data[indep + [dep]]
        colors = ['#808080', '#FFFFFF', '#C0C0C0']
        box = sns.boxplot(x=indep[0], y=dep, hue=indep[1], data=data, palette = colors, width = 0.6)
        fig = box.get_figure()
        fig.savefig(currPath+"twoWayANOVA_boxPlot.png")
        plt.close()

        #Create a bar plot
        sns.set(rc = {'figure.figsize':(15,10)})
        sns.set(font_scale = 1.5)
        sns.set_style('whitegrid')

        if data[indep[0]].isin(['other']).any():
            for i in range(len(data)):
                if data[indep[0]].iat[i]!='other':
                    data[indep[0]].iat[i]=indep[0][indep[0].rfind('_')+1:]

        if 'sex' in indep:
            data['sex']=data['sex'].astype(str)
            for i in range(len(data)):
                if data[indep[1]].iat[i]=='0.0':
                    data[indep[1]].iat[i]='female'
                else:
                    data[indep[1]].iat[i]='male'


        self.plotComboBox(indep[0],dep,data,OGdata,currPath)
        fig, bar = plt.subplots()
        
        colors = ['#808080', '#FFFFFF', '#C0C0C0']

        sns.barplot(x=indep[0], ax = bar, y=dep, hue=indep[1], data=data, 
        palette = colors, capsize=.1)
        width = 0.3

        num_var2 = len(data[indep[0]].unique())
        hatches = itertools.cycle(['+', 'x', '-', '/', '//', '\\', '*', 'o', 'O', '.'])

        '''
        for i, patch in enumerate(bar.patches):
            # Set a different hatch for each bar
            if i % num_var2 == 0:
                hatch = next(hatches)
            patch.set_hatch(hatch)
        '''

        for patch in bar.patches:
            current_width = patch.get_width()
            diff = current_width - width
            patch.set_width(width)
            patch.set_x(patch.get_x() + diff * .5)
            patch.set_edgecolor('#000000')

        bar.legend(frameon = 1, title = indep[1], fontsize = 15, title_fontsize = 20,
        bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)     
        fig = bar.get_figure()
        fig.savefig(currPath+"twoWayANOVA_barPlot.png")
        plt.close()
            
        #Conducting the ANOVA test
        twoWayANOVA.write('Results for two way ANOVA between ' + indep[0] + '&' + indep[1] + ' and ' + dep + ' are: \n\n')
        res = bioinfokit.analys.stat()
        res.anova_stat(df=data, res_var=dep, anova_model=formula, ss_typ=3)
        asummary = res.anova_summary.iloc[1:, :].to_string(header=True, index=True)
        twoWayANOVA.write(asummary + '\n')
        twoWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
        results['ANOVA_Results'] = res.anova_summary

        
        #Follow-up Test TukeyTest
        if (all(x > alpha  for x in res.anova_summary.iloc[1:4, 4].tolist())) and (not followUp):
            twoWayANOVA.write('All the p-values is higher than alpha; hence, no follow-up test was conducted\n\n')
        else:
            tukey = list()
            message = list()
            message.append('Main effect for ' + indep[0] + ':\n')
            message.append('Main effect for ' + indep[1] + ':\n')
            message.append('Interaction effect between ' + indep[0] + ' and ' + indep[1] + ':\n')                
            twoWayANOVA.write('Results for follow-up Tukey test between ' + indep[0] + ' & ' + indep[1] + ' and ' + dep + ' are: \n\n')
            followUp = bioinfokit.analys.stat()
            if len(data[indep[0]].value_counts()) <= 2:
                tukey.append('Only two groups. No follow up test was conducted\n')
            else:
                followUp.tukey_hsd(df=data, res_var=dep, xfac_var=indep[0], anova_model=formula)
                tukey.append(followUp.tukey_summary.to_string(header=True, index=False))
            
            if len(data[indep[1]].value_counts()) <= 2:
                tukey.append('Only two groups. No follow up test was conducted\n')
            else:
                followUp.tukey_hsd(df=data, res_var=dep, xfac_var=indep[1], anova_model=formula)
                tukey.append(followUp.tukey_summary.to_string(header=True, index=False))

            if len(data[indep[0]].value_counts())*len(data[indep[1]].value_counts()) <= 2:
                tukey.append('Only two groups. No follow up test was conducted\n')
            elif res.anova_summary.iloc[1:4, 4].tolist()[2] > alpha:
                tukey.append('Interaction effect not significant. No follow up test was conducted\n')
            else:
                followUp.tukey_hsd(df=data, res_var=dep, xfac_var=indep, anova_model=formula)
                tukey.append(followUp.tukey_summary.to_string(header=True, index=False))

            for i in range(len(tukey)):
                twoWayANOVA.write(message[i] + tukey[i] + '\n\n')
                twoWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
            results['Tukey_Results'] = tukey
        twoWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
            
            
        #histograms and QQ-plot
        sm.qqplot(res.anova_std_residuals, line='45')
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Standardized Residuals")
        plt.savefig(currPath+'twoWayANOVA_qqPlot.png')
        plt.close()

        plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k') 
        plt.xlabel("Residuals")
        plt.ylabel('Frequency')
        plt.savefig(currPath+'twoWayANOVA_histogram.png')
        plt.close()
            
            
        #Shapiro-Wilk Test for Normality
        w, pvalue = stats.shapiro(res.anova_model_out.resid)
        twoWayANOVA.write('Results for Shapiro-Wilk test to check for normality are: \n\n')
        twoWayANOVA.write('w is: ' + str(w) + '/ p-value is: ' + str(pvalue) + '\n')
        twoWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
        results['Shaprio-Wilk_Results'] = (w, pvalue)
            
        #Check for equality of varianve using Levene's test
        twoWayANOVA.write("Results for Levene's test to check for equality of variance are: \n\n")
        eqOfVar = bioinfokit.analys.stat()
        eqOfVar.levene(df=data, res_var=dep, xfac_var=indep)
        eqOfVarSummary = eqOfVar.levene_summary.to_string(header=True, index=False)
        twoWayANOVA.write(eqOfVarSummary + '\n')
        twoWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
        results['Levene_Results'] = eqOfVar.levene_summary

        ###New SRH stuff
        file=open("SRH.txt","w")
        file.close()
        os.system("Rscript SHR.R "+dep+" "+indep[0]+" "+indep[1])


        #The scheirer Ray Hare test
        '''
        rcompanion = importr('rcompanion')
        formulaMaker = r['as.formula']
        scheirerRayHare = r['scheirerRayHare']

        formula = formulaMaker(dep + ' ~ '  + indep[0] + ' + ' + indep[1])
        print(data.dtypes)
        with localconverter(ro.default_converter + pandas2ri.converter):
                rDf = ro.conversion.py2rpy(data)

        scheirerANOVA = scheirerRayHare(formula, data = rDf)

        with localconverter(ro.default_converter + pandas2ri.converter):
            scheirerANOVA = ro.conversion.rpy2py(scheirerANOVA)


        twoWayANOVA.write('Results for the scheirer Ray Hare Test -- to be used if ANOVA assumptions are violated: \n\n')
        scheirerSummary = scheirerANOVA.to_string(header=True, index=True)
        twoWayANOVA.write(scheirerSummary + '\n')
        twoWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
        results['scheirerRayHare'] = scheirerANOVA
        
        #The dunn's test -- follow up
        if (all(x > alpha  for x in scheirerANOVA['p.value'].tolist())) and (not followUp):
            twoWayANOVA.write('All the p-values is higher than alpha; hence, no follow-up test was conducted for ScheirerRayHare test\n\n')
        
        else:
            FSA = importr('FSA')
            dunnTest = r['dunnTest']
            data['interaction'] = data[indep[0]].astype(str) + '_' + data[indep[1]].astype(str)
            with localconverter(ro.default_converter + pandas2ri.converter):
                rDf = ro.conversion.py2rpy(data)
            indep.append('interaction')            
            for var in indep:
                formula = formulaMaker(dep + ' ~ ' + var)
                dunnTwoWay = dunnTest(formula, data=rDf, method="bonferroni")

                asData, doCall, rbind = r['as.data.frame'], r['do.call'], r['rbind']
                dunnTwoWay = asData(doCall(rbind, dunnTwoWay))

                with localconverter(ro.default_converter + pandas2ri.converter):
                    dunnTwoWay = ro.conversion.rpy2py(dunnTwoWay)

                dunnTwoWay.drop(['method', 'dtres'], inplace = True)
                for col in ['Z', 'P.unadj', 'P.adj']:
                    dunnTwoWay[col] = pd.to_numeric(dunnTwoWay[col])
                    dunnTwoWay[col] = np.round(dunnTwoWay[col], decimals = 5)
                if var == 'interaction':
                    twoWayANOVA.write("Results for follow-up Dunn's test between " + indep[0] + ' & ' + indep[1] + " and " + dep + " are: \n\n")
                else:
                    twoWayANOVA.write("Results for follow-up Dunn's test between " + var + " and " + dep + " are: \n\n")
                
                if len(data[var].value_counts()) <= 2:
                    twoWayANOVA.write('Only two groups. No follow up test was conducted\n')
                elif var == 'interaction' and scheirerANOVA['p.value'].tolist()[2] > alpha:
                    twoWayANOVA.write('Interaction effect not significant. No follow up test was conducted\n')
                else:
                    dunnSummary = dunnTwoWay.to_string(header=True, index=False)
                    twoWayANOVA.write(dunnSummary + '\n\n')
        
                twoWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
        
        self.twoANOVA = results
        return results
        '''



    def ANOVA(self, dep, indep, alpha = 0.05, oneWay = True, followUp = False, oneVsOther = dict(), oneVsAnother =dict()):
        '''
        Conduct an ANOVA analysis -- either one or two way -- between the dependent and independent variables
        passed. If there is signifcant effect found, conduct a follow up test. The function checks for the ANOVA
        assumption and provide alternative tests such as Kruskal-Wallis H. Results will be stored at ###
        
        Args:
            data: DataFrame containing the items of interest
            dep: the dependent varaible in the analysis
            indep: column names in the data frame containing the groups -- can be 
                a string or a list of two strings
            alpha: minimum value for the p-value for the effect to be signifcant
                conduct repeated measures ANOVA -- to be implemented.
            oneWay: if True, conduct one way ANOVA. if False, conduct two way ANOVA.
            followUp: if True, a follow up test would be conducted regardless of the ANOVA p-value

        Returns:
            a dictionary mapping each test conducted to its results
        '''
        data = self.dpnonOHCnonBin.df.copy(deep = True)
        data.columns = data.columns.str.replace(' ', '_')
        data.columns = data.columns.str.replace('.', '_')
        types=np.array(self.dpnonOHCnonBin.types)
        features=np.array(data.columns)
        continuous=list(features[np.where(types!='c')[0]])
        if dep in continuous:
            continuous.remove(dep)
        data.drop(labels=continuous,axis=1,inplace=True)
        OGdata = data.copy(deep = True)

        for var in oneVsOther.keys():
            data.loc[data[var] != oneVsOther[var], var] = 'other'
            #newName = var+'_oneVsRest_'+str(oneVsOther[var])
            newName = var+'_'+str(oneVsOther[var])
            data.rename(columns = {var:newName}, inplace = True)
            if indep == var:
                indep = newName
            elif var in indep:
                indep.remove(var)
                indep.append(newName)

        for var in oneVsAnother.keys():

            d1 = data[data[var] == oneVsAnother[var][0]]
            d2 = data[data[var] == oneVsAnother[var][1]]
            data = pd.concat([d1,d2])
            newName = var+'_'+str(oneVsAnother[var][0])+'Vs'+str(oneVsAnother[var][1])
            data.rename(columns = {var:newName}, inplace = True)
            if indep == var:
                indep = newName
            elif var in indep:
                indep.remove(var)
                indep.append(newName)

        if oneWay:
            if (type(indep) == list):
                indep = indep[0]
            if not oneVsOther:
                data = data.astype({indep: 'int64'})
            else:
                data = data.astype({indep: 'str'})
            try:    
                return self.oneWay_ANOVA(data, dep, indep, alpha, False, followUp)
            except ValueError:
                return
        else:
            if not oneVsOther:
                data = data.astype({indep[0]:'int64'})
                data = data.astype({indep[1]:'int64'})
            else:
                data = data.astype({indep[0]:'str'})
                data = data.astype({indep[1]:'str'})
            return self.twoWay_ANOVA(data, dep, indep, alpha, False, followUp, OGdata)


    def Association_Analysis(self, vars=None, oneTarget = False, target = '', chi = False, fisher = False):
            
        ''' 
        Do Chi Square test, Fisher exact test, and g-test of independence between the passed variables.
        Write the findings to to Association_Analysis.txt in statisticalAnalysis.
            
        Args:
            data: DataFrame containing the items of interest
            vars: list of variables to apply the tests to. Have to contain at least two
            oneTarget: if True, the analysis would be done against the specified variable.
                If True, the parameter target must be passed
            target: the variable of interest to test other vraiales against
            chi: if True, conduct ChiSquare test
            Fisher: if True, conduct Fisher exact test -- can result in an error

        Returns:
            a data frame containing the results for chi-square, fisher, and g tests
        '''
        data = self.dpnonOHCnonBin.df.copy(deep = True)
        data.columns = data.columns.str.replace(' ', '_')
        data.columns = data.columns.str.replace('.', '_')
        types=np.array(self.dpnonOHCnonBin.types)
        features=np.array(data.columns)
        continuous=list(features[np.where(types!='c')[0]])
        if target in continuous:
            continuous.remove(target)
        data.drop(labels=continuous,axis=1,inplace=True)
        if vars is None:
            vars=list(data.columns.copy(deep=True))

        
        if not os.path.exists(self.statsPath+'AssociationAnalysis'):
            os.makedirs(self.statsPath+'AssociationAnalysis')
        currPath = self.statsPath + 'AssociationAnalysis/'
        
        #results = pd.DataFrame(columns=['var1-var2','Pearson-Chi-square', 'Chi-p-value', "Chi-Cramer's-phi",
        #    'Fisher-Odds-ratio', 'Fisher-2-sided-p-value', "Fisher-Cramer's-phi", 'G-Test-Log-likelihood-ratio', 
        #    'G-Test-p-value', "G-Test-Cramer's-phi"])
        
        resultsG = pd.DataFrame(columns=['var1-var2', 'G-value', 'G-df', 'G-pValue', 'expectedAbove5'])
        resultsFisher = pd.DataFrame(columns=['var1-var2', 'Fisher-pValue', 'Fisher-oddsRatio'])
        resultsChi = pd.DataFrame(columns=['var1-var2', 'Chi-value', 'Chi-df', 'Chi-pValue'])
        belowThreshhold = list()

        rows = list()
        tempList = list(vars)
        for var1 in vars:
            print(var1)
            if oneTarget and var1!=target:
                #fisher = FisherExactAnalysis(data, var1, target)
                g, exp = self.G_TestAnalysis(data, var1, target)

                rows.append([str(var1)+'-'+str(target), 
                g[0][0], g[1][0], g[2][0], exp])
                if exp < 0.8:
                    belowThreshhold.append((var1, target))

            elif not oneTarget:
                tempList.remove(var1)
                for var2 in tempList:
                    print(var1, var2)
                    g, exp = self.G_TestAnalysis(data, var1, var2)
                    rows.append([str(var1)+'-'+str(var2), 
                    g[0][0], g[1][0], g[2][0], exp])
                    if exp < 0.8:
                        belowThreshhold.append((var1, var2))
            
            
            
        for i in range(len(rows)):
            resultsG.loc[len(resultsG.index)] = rows[i]
        
        AssociationAnalysis = open(currPath+'AssociationAnalysisGTest.txt', 'w')
        AssociationAnalysis.write('Association Analysis Results of G Test: \n---------------------------------------------------\n\n')
        for i in range(len(resultsG)):
            two_var = resultsG.iloc[i, :]
            two_var = two_var.to_frame()
            variables = str(two_var.iloc[0,0]).split('-')
            two_var = two_var.iloc[1: , :]
            AssociationAnalysis.write('The Association Analysis Results for G Test between ' +variables[0]+' and ' + variables[1] + ' are: \n\n')
            toWrite = two_var.to_string(header = False, index = True)
            AssociationAnalysis.write(toWrite+'\n')
            AssociationAnalysis.write('----------------------------------------------------------------------------------------\n\n')
            
        AssociationAnalysis.write('Variables that do not fullfil the GTest and Chi Square assumptions: \n\n')
        for var in belowThreshhold:
            AssociationAnalysis.write(str(var)+'\n')
        AssociationAnalysis.write('----------------------------------------------------------------------------------------\n\n')

        
        AssociationAnalysis.close()
        results = resultsG.copy(deep = True)

        if fisher:
            rows = list()
            tempList = list(vars)
            for var1 in vars:
                if oneTarget and var1!=target:
                    fisher, _ = self.FisherExactAnalysis(data, var1, target)
                    rows.append([str(var1)+'-'+str(target), 
                    fisher[0][0], fisher[2][0]])

                elif not oneTarget:
                    tempList.remove(var1)
                    for var2 in tempList:
                        fisher, _ = self.FisherExactAnalysis(data, var1, var2)
                        rows.append([str(var1)+'-'+str(var2), 
                        fisher[0][0], fisher[2][0]])
                
            for i in range(len(rows)):
                resultsFisher.loc[len(resultsFisher.index)] = rows[i]
            
            AssociationAnalysis = open(currPath+'AssociationAnalysisFisherTest.txt', 'w')
            AssociationAnalysis.write('Association Analysis Results of Fisher Test: \n---------------------------------------------------\n\n')
            for i in range(len(resultsFisher)):
                two_var = resultsFisher.iloc[i, :]
                two_var = two_var.to_frame()
                variables = str(two_var.iloc[0,0]).split('-')
                two_var = two_var.iloc[1: , :]
                AssociationAnalysis.write('The Association Analysis Results for Fisher Test between ' +variables[0]+' and ' + variables[1] + ' are: \n\n')
                toWrite = two_var.to_string(header = False, index = True)
                AssociationAnalysis.write(toWrite+'\n')
                AssociationAnalysis.write('----------------------------------------------------------------------------------------\n\n')
                
            
            AssociationAnalysis.close()
            results = pd.concat([results, resultsFisher], axis = 1)

        if chi:
            rows = list()
            tempList = list(vars)
            for var1 in vars:
                if oneTarget and var1!=target:
                    chiTest, _ = self.Chi_SquareAnalysis(data, var1, target)
                    rows.append([str(var1)+'-'+str(target), 
                    chiTest[0][0], chiTest[1][0], chiTest[2][0]])

                elif not oneTarget:
                    tempList.remove(var1)
                    for var2 in tempList:
                        chiTest, _ = self.Chi_SquareAnalysis(data, var1, var2)
                        rows.append([str(var1)+'-'+str(var2), 
                        chiTest[0][0], chiTest[1][0], chiTest[2][0]])
                
            for i in range(len(rows)):
                resultsChi.loc[len(resultsChi.index)] = rows[i]
            
            AssociationAnalysis = open(currPath+'AssociationAnalysisChiSquareTest.txt', 'w')
            AssociationAnalysis.write('Association Analysis Results of Chi Square Test: \n---------------------------------------------------\n\n')
            for i in range(len(resultsG)):
                two_var = resultsChi.iloc[i, :]
                two_var = two_var.to_frame()
                variables = str(two_var.iloc[0,0]).split('-')
                two_var = two_var.iloc[1: , :]
                AssociationAnalysis.write('The Association Analysis Results for Chi Square Test between ' +variables[0]+' and ' + variables[1] + ' are: \n\n')
                toWrite = two_var.to_string(header = False, index = True)
                AssociationAnalysis.write(toWrite+'\n')
                AssociationAnalysis.write('----------------------------------------------------------------------------------------\n\n')
                
            
            AssociationAnalysis.close()
            results = pd.concat([results, resultsChi], axis = 1)

        self.AssocAnalysis['results'] = results
        return results


    def Chi_SquareAnalysis(self, data, var1, var2):
        table = np.array(pd.crosstab(data[var1], data[var2], margins = False))
        chi = r['chisq.test']
        res = chi(table)
        return res, None


    def FisherExactAnalysis(self, data, var1, var2):
        table = np.array(pd.crosstab(data[var1], data[var2], margins = False))
        stats = importr('stats')
        res = stats.fisher_test(table)
        return res, None

        #_, test_results_Fisher, _ = rp.crosstab(data[var1], data[var2],test= "fisher", 
        #expected_freqs= True, prop= "cell", correction = True)
        #return test_results_Fisher

    def G_TestAnalysis(self, data, var1, var2):

        _, _, exp = rp.crosstab(data[var1], data[var2],test= "g-test", 
        expected_freqs= True, prop= "cell", correction = True)

        exp = np.array(exp)

        total = exp.shape[0]*exp.shape[1]
        overFive = 0
        for i in range(len(exp)):
            for j in range(len(exp[0])):
                if exp[i][j] >= 5:
                    overFive +=1
            
        table = np.array(pd.crosstab(data[var1], data[var2], margins = False))
        desk = importr('DescTools')
        if (table.shape == (2,2)):
            res = desk.GTest(table, correct = 'yates')
        else:
            res = desk.GTest(table, correct = 'williams')

        return res, overFive/total

    def plotCombo(self,indep,dep,data,OGdata,currPath):

        
        unmod=pd.read_csv('Data/unmodified_GAD7.csv',index_col=0)
        data=data.copy(deep=True)
        data['GAD7']=unmod.loc[data.index,'GAD7']
        OGdata['GAD7']=unmod.loc[data.index,'GAD7']

        mapping={'PER3A':['C','G'],'PER3C':['T','G'],'PER2':['G','A'],'CLOCK3111':['A','G'],'PER3B':['G','A'],'ZBTB20':['C','T'],'CRY2':['A','G'],'CRY1':['C','G']}
        grid=plt.GridSpec(1,9,wspace=0,hspace=0.3)
        plt.suptitle(indep[:indep.find('and')]+'-'+indep[indep.find('and')+3:],fontsize=18)


        X = ['Females','Males']
        X_axis = np.arange(len(X))
        ax1=plt.subplot(grid[0, 0:4])
        Females=data[data['sex']=='female']
        Males=data[data['sex']=='male']
        CTEs_target=[np.mean(Females[Females[indep]!='other'][dep]),np.mean(Males[Males[indep]!='other'][dep])]
        CTEs_other=[np.mean(Females[Females[indep]=='other'][dep]),np.mean(Males[Males[indep]=='other'][dep])]
        error_target=[sem(Females[Females[indep]!='other'][dep]),sem(Males[Males[indep]!='other'][dep])]
        error_other=[sem(Females[Females[indep]=='other'][dep]),sem(Males[Males[indep]=='other'][dep])]
        ax1.bar(X_axis,CTEs_target,color='black',yerr=error_target,capsize=5,width=0.2,label=indep[indep.find('and')-2:][:indep[indep.find('and')-2:].find('and')]+'-'+indep[indep.find('and')-2:][indep[indep.find('and')-2:].find('and')+3:])
        ax1.bar(X_axis+0.2,CTEs_other,color='gray',yerr=error_other,capsize=5,width=0.2,label='Others')
        ax1.set_xlabel('Sex',fontsize=18)
        ax1.set_xticks(X_axis)
        ax1.set_xticklabels(X)
        ax1.tick_params(axis='y', which='major', labelsize=11)
        ax1.tick_params(axis='x', which='major', labelsize=18)
        ax1.set_ylabel('GAD7 Score',fontsize=18)
        leg = ax1.legend(prop={"size":15}, loc='upper right')
        ax1.grid(False)
        


        ax0=plt.subplot(grid[0, 4])
        ax0.axis("off")

        ax2=plt.subplot(grid[0, 5:7])
        ax3=plt.subplot(grid[0, 7:9])

        
        xGene=indep[:indep.find('_')]
        yGene=indep[indep.find('_')+1:indep.rfind('_')]
        print(xGene)
        print(yGene)
        X = [mapping[xGene][0]+mapping[xGene][0],mapping[xGene][0]+mapping[xGene][1],mapping[xGene][1]+mapping[xGene][1]]
        Y = [mapping[yGene][0]+mapping[yGene][0],mapping[yGene][0]+mapping[yGene][1],mapping[yGene][1]+mapping[yGene][1]]
        X_axis = np.arange(len(X))
        ax2=plt.subplot(grid[0, 5:7])
        ax3=plt.subplot(grid[0, 7:9])
        OGFemales=OGdata[OGdata['sex']==0.0]
        OGMales=OGdata[OGdata['sex']==1.0]

        temp=OGFemales[(OGFemales[xGene]==0) & (OGFemales[yGene]==0)]
        print("dep: "+dep)
        print(temp)
        print(temp[dep])

        CTEs_Females_y0=[np.mean(OGFemales[(OGFemales[xGene]==0) & (OGFemales[yGene]==0)][dep]),np.mean(OGFemales[(OGFemales[xGene]==1) & (OGFemales[yGene]==0)][dep]),np.mean(OGFemales[(OGFemales[xGene]==2) & (OGFemales[yGene]==0)][dep])]
        CTEs_Females_y1=[np.mean(OGFemales[(OGFemales[xGene]==0) & (OGFemales[yGene]==1)][dep]),np.mean(OGFemales[(OGFemales[xGene]==1) & (OGFemales[yGene]==1)][dep]),np.mean(OGFemales[(OGFemales[xGene]==2) & (OGFemales[yGene]==1)][dep])]
        CTEs_Females_y2=[np.mean(OGFemales[(OGFemales[xGene]==0) & (OGFemales[yGene]==2)][dep]),np.mean(OGFemales[(OGFemales[xGene]==1) & (OGFemales[yGene]==2)][dep]),np.mean(OGFemales[(OGFemales[xGene]==2) & (OGFemales[yGene]==2)][dep])]
        error_Females_y0=[sem(OGFemales[(OGFemales[xGene]==0) & (OGFemales[yGene]==0)][dep]),sem(OGFemales[(OGFemales[xGene]==1) & (OGFemales[yGene]==0)][dep]),sem(OGFemales[(OGFemales[xGene]==2) & (OGFemales[yGene]==0)][dep])]
        error_Females_y1=[sem(OGFemales[(OGFemales[xGene]==0) & (OGFemales[yGene]==1)][dep]),sem(OGFemales[(OGFemales[xGene]==1) & (OGFemales[yGene]==1)][dep]),sem(OGFemales[(OGFemales[xGene]==2) & (OGFemales[yGene]==1)][dep])]
        error_Females_y2=[sem(OGFemales[(OGFemales[xGene]==0) & (OGFemales[yGene]==2)][dep]),sem(OGFemales[(OGFemales[xGene]==1) & (OGFemales[yGene]==2)][dep]),sem(OGFemales[(OGFemales[xGene]==2) & (OGFemales[yGene]==2)][dep])]

        ax2.set_title('Female',fontsize=18)
        ax2.set_xticks(X_axis)
        ax2.set_xticklabels(X)
        ax2.bar(X_axis-0.2,CTEs_Females_y0,color='black',yerr=error_Females_y0,capsize=5,width=0.2,label=Y[0])
        ax2.bar(X_axis,CTEs_Females_y1,color='gray',yerr=error_Females_y1,capsize=5,width=0.2,label=Y[1])
        ax2.bar(X_axis+0.2,CTEs_Females_y2,color='lightgray',yerr=error_Females_y2,capsize=5,width=0.2,label=Y[2])
        ax2.set_xlabel(xGene,fontsize=18)
        ax2.tick_params(axis='both', which='major', labelsize=11)
        ax2.set_ylabel('GAD7 Score',fontsize=18)
        ax2.grid(False)

        CTEs_Males_y0=[np.mean(OGMales[(OGMales[xGene]==0) & (OGMales[yGene]==0)][dep]),np.mean(OGMales[(OGMales[xGene]==1) & (OGMales[yGene]==0)][dep]),np.mean(OGMales[(OGMales[xGene]==2) & (OGMales[yGene]==0)][dep])]
        CTEs_Males_y1=[np.mean(OGMales[(OGMales[xGene]==0) & (OGMales[yGene]==1)][dep]),np.mean(OGMales[(OGMales[xGene]==1) & (OGMales[yGene]==1)][dep]),np.mean(OGMales[(OGMales[xGene]==2) & (OGMales[yGene]==1)][dep])]
        CTEs_Males_y2=[np.mean(OGMales[(OGMales[xGene]==0) & (OGMales[yGene]==2)][dep]),np.mean(OGMales[(OGMales[xGene]==1) & (OGMales[yGene]==2)][dep]),np.mean(OGMales[(OGMales[xGene]==2) & (OGMales[yGene]==2)][dep])]
        error_Males_y0=[sem(OGMales[(OGMales[xGene]==0) & (OGMales[yGene]==0)][dep]),sem(OGMales[(OGMales[xGene]==1) & (OGMales[yGene]==0)][dep]),sem(OGMales[(OGMales[xGene]==2) & (OGMales[yGene]==0)][dep])]
        error_Males_y1=[sem(OGMales[(OGMales[xGene]==0) & (OGMales[yGene]==1)][dep]),sem(OGMales[(OGMales[xGene]==1) & (OGMales[yGene]==1)][dep]),sem(OGMales[(OGMales[xGene]==2) & (OGMales[yGene]==1)][dep])]
        error_Males_y2=[sem(OGMales[(OGMales[xGene]==0) & (OGMales[yGene]==2)][dep]),sem(OGMales[(OGMales[xGene]==1) & (OGMales[yGene]==2)][dep]),sem(OGMales[(OGMales[xGene]==2) & (OGMales[yGene]==2)][dep])]

        ax3.set_title('Male',fontsize=18)
        ax3.set_xticks(X_axis)
        ax3.set_xticklabels(X)
        ax3.bar(X_axis-0.2,CTEs_Males_y0,color='black',yerr=error_Males_y0,capsize=5,width=0.2,label=Y[0])
        ax3.bar(X_axis,CTEs_Males_y1,color='gray',yerr=error_Males_y1,capsize=5,width=0.2,label=Y[1])
        ax3.bar(X_axis+0.2,CTEs_Males_y2,color='lightgray',yerr=error_Males_y2,capsize=5,width=0.2,label=Y[2])
        ax3.set_xlabel(xGene,fontsize=18)
        ax3.tick_params(axis='x', which='major', labelsize=11)
        ax3.set_yticks([])
        leg = ax3.legend(prop={"size":15}, loc='upper right')
        ax3.grid(False)
        
        
        plt.tight_layout()
        plt.savefig(currPath+"Combo_barPlots.png")
        plt.close()


    def plotComboBox(self,indep,dep,data,OGdata,currPath):
        
        unmod=pd.read_csv('Data/unmodified_GAD7.csv',index_col=0)
        data=data.copy(deep=True)
        data['GAD7']=unmod.loc[data.index,'GAD7']
        OGdata['GAD7']=unmod.loc[data.index,'GAD7']

        mapping={'PER3A':['C','G'],'PER3C':['T','G'],'PER2':['G','A'],'CLOCK3111':['A','G'],'PER3B':['G','A'],'ZBTB20':['C','T'],'CRY2':['A','G'],'CRY1':['C','G']}
        grid=plt.GridSpec(1,9,wspace=0,hspace=0.3)
        plt.suptitle(indep[:indep.find('and')]+'-'+indep[indep.find('and')+3:],fontsize=18)


        X = ['Females','Males']
        X_axis = np.arange(len(X))
        ax1=plt.subplot(grid[0, 0:4])
        Females=data[data['sex']=='female']
        Males=data[data['sex']=='male']
        CTEs_target=[Females[Females[indep]!='other'][dep],Males[Males[indep]!='other'][dep]]
        CTEs_other=[Females[Females[indep]=='other'][dep],Males[Males[indep]=='other'][dep]]
        
        label=indep[indep.find('and')-2:][:indep[indep.find('and')-2:].find('and')]+'-'+indep[indep.find('and')-2:][indep[indep.find('and')-2:].find('and')+3:]
        ax1.boxplot(X_axis,CTEs_target)
        ax1.boxplot(X_axis+0.2,CTEs_other,width=0.2,label='Others')
        
        ax1.set_xlabel('Sex',fontsize=18)
        ax1.set_xticks(X_axis)
        ax1.set_xticklabels(X)
        ax1.tick_params(axis='y', which='major', labelsize=11)
        ax1.tick_params(axis='x', which='major', labelsize=18)
        ax1.set_ylabel('GAD7 Score',fontsize=18)
        leg = ax1.legend(prop={"size":15}, loc='upper right')
        ax1.grid(False)
        


        ax0=plt.subplot(grid[0, 4])
        ax0.axis("off")

        ax2=plt.subplot(grid[0, 5:7])
        ax3=plt.subplot(grid[0, 7:9])

        
        xGene=indep[:indep.find('_')]
        yGene=indep[indep.find('_')+1:indep.rfind('_')]
        print(xGene)
        print(yGene)
        X = [mapping[xGene][0]+mapping[xGene][0],mapping[xGene][0]+mapping[xGene][1],mapping[xGene][1]+mapping[xGene][1]]
        Y = [mapping[yGene][0]+mapping[yGene][0],mapping[yGene][0]+mapping[yGene][1],mapping[yGene][1]+mapping[yGene][1]]
        X_axis = np.arange(len(X))
        ax2=plt.subplot(grid[0, 5:7])
        ax3=plt.subplot(grid[0, 7:9])
        OGFemales=OGdata[OGdata['sex']==0.0]
        OGMales=OGdata[OGdata['sex']==1.0]

        temp=OGFemales[(OGFemales[xGene]==0) & (OGFemales[yGene]==0)]
        print("dep: "+dep)
        print(temp)
        print(temp[dep])

        CTEs_Females_y0=[OGFemales[(OGFemales[xGene]==0) & (OGFemales[yGene]==0)][dep],OGFemales[(OGFemales[xGene]==1) & (OGFemales[yGene]==0)][dep],OGFemales[(OGFemales[xGene]==2) & (OGFemales[yGene]==0)][dep]]
        CTEs_Females_y1=[OGFemales[(OGFemales[xGene]==0) & (OGFemales[yGene]==1)][dep],OGFemales[(OGFemales[xGene]==1) & (OGFemales[yGene]==1)][dep],OGFemales[(OGFemales[xGene]==2) & (OGFemales[yGene]==1)][dep]]
        CTEs_Females_y2=[OGFemales[(OGFemales[xGene]==0) & (OGFemales[yGene]==2)][dep],OGFemales[(OGFemales[xGene]==1) & (OGFemales[yGene]==2)][dep],OGFemales[(OGFemales[xGene]==2) & (OGFemales[yGene]==2)][dep]]
        

        ax2.set_title('Female',fontsize=18)
        ax2.set_xticks(X_axis)
        ax2.set_xticklabels(X)
        ax2.boxplot(X_axis-0.2,CTEs_Females_y0,width=0.2,label=Y[0])
        ax2.boxplot(X_axis,CTEs_Females_y1,width=0.2,label=Y[1])
        ax2.boxplot(X_axis+0.2,CTEs_Females_y2,width=0.2,label=Y[2])
        ax2.set_xlabel(xGene,fontsize=18)
        ax2.tick_params(axis='both', which='major', labelsize=11)
        ax2.set_ylabel('GAD7 Score',fontsize=18)
        ax2.grid(False)

        CTEs_Males_y0=[OGMales[(OGMales[xGene]==0) & (OGMales[yGene]==0)][dep],OGMales[(OGMales[xGene]==1) & (OGMales[yGene]==0)][dep],OGMales[(OGMales[xGene]==2) & (OGMales[yGene]==0)][dep]]
        CTEs_Males_y1=[OGMales[(OGMales[xGene]==0) & (OGMales[yGene]==1)][dep],OGMales[(OGMales[xGene]==1) & (OGMales[yGene]==1)][dep],OGMales[(OGMales[xGene]==2) & (OGMales[yGene]==1)][dep]]
        CTEs_Males_y2=[OGMales[(OGMales[xGene]==0) & (OGMales[yGene]==2)][dep],OGMales[(OGMales[xGene]==1) & (OGMales[yGene]==2)][dep],OGMales[(OGMales[xGene]==2) & (OGMales[yGene]==2)][dep]]
        

        ax3.set_title('Male',fontsize=18)
        ax3.set_xticks(X_axis)
        ax3.set_xticklabels(X)
        ax3.boxplot(X_axis-0.2,CTEs_Males_y0,width=0.2,label=Y[0])
        ax3.boxplot(X_axis,CTEs_Males_y1,width=0.2,label=Y[1])
        ax3.boxplot(X_axis+0.2,CTEs_Males_y2,width=0.2,label=Y[2])
        ax3.set_xlabel(xGene,fontsize=18)
        ax3.tick_params(axis='x', which='major', labelsize=11)
        ax3.set_yticks([])
        leg = ax3.legend(prop={"size":15}, loc='upper right')
        ax3.grid(False)
        
        
        plt.tight_layout()
        plt.savefig(currPath+"Combo_barPlots.png")
        plt.close()
