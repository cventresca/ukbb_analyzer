import pandas as pd
import numpy as np
import os
#import kFoldValidationRuns as runs
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
#from StatisticalAnalysis import Stats
import multiprocessing as mp
from data_processor import data_processor
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
import sys
from StepwisePrecomputation import PrecomputeMuL,PrecomputeOD
import shutil
from pathlib import Path
from utilities import relabel



def distill(features):
    new_features=[]
    for i in range(len(features)):
        feature=features[i]
        if '.' in feature:
            working=feature.split('.')
            new_feature='.'.join([working[0].split('_')[0],working[1].split('_')[0]])
        elif '_' in feature:
            new_feature=feature.split('_')[0]
        else:
            new_feature=feature

        if new_feature not in new_features:
            new_features.append(new_feature)
    return new_features

def distillTotal(dir,features_target,features):
    file=open('NormalDataAnalysis/'+dir+'/results/features/robustFinal.txt','r')
    line=[int(i) for i in file.readline().split() if i!='None']
    index=np.where(features==features_target)[0][0]
    line=np.array([i+1 if i>=index else i for i in line])
    #print(index)
    #print(line)
    line=features[line]
    line=np.append(line,features_target)
    #print(len(line))
    #print(line)
    new_features=distill(line)
    #print(len(new_features))
    #print(np.array(new_features))
    return np.array(new_features)
    
    
def importantFeatures(dir,data,target):
    file=open('NormalDataAnalysis/'+dir+'/results/features/robustFinal.txt','r')
    line=[int(i) for i in file.readline().split() if i!='None']
    index=np.where(data.columns==target)[0][0]
    line=np.array([i+1 if i>=index else i for i in line])
    line=np.array(data.columns[line])
    line=np.append(line,target)
    return line
    #to_drop=np.array(data.columns[np.setdiff1d(np.array([i for i in range(len(data.columns))]),line)])
    #print(np.setdiff1d(data.columns,to_drop))
    #to_drop=np.delete(to_drop,np.where(np.array(to_drop)==target)[0])
    #data.drop(labels=to_drop,axis=1,inplace=True)

def strip_underscores(feature): #only works when number of categories does not exceed 10 (otherwise treat as discrete/continuous)
    while '_' in feature:
        feature=feature[:feature.find('_')]+feature[feature.find('_')+2:]
    return feature

def applyFeatureSelection(dpOHC,dpnonOHC,dpOHCnonBin,dpnonOHCnonBin,sex=-1):
    
    dpOHC.loadFile('run1OHC.csv')
    dpnonOHC.loadFile('run1nonOHC.csv')
    dpOHCnonBin.loadFile('run1OHCnonBin.csv')
    dpnonOHCnonBin.loadFile('run1nonOHCnonBin.csv')
    '''
    dpOHC.loadFile('sendToWaelOHC.csv')
    dpnonOHC.loadFile('sendToWaelnonOHC.csv')
    dpOHCnonBin.loadFile('sendToWaelOHCnonBin.csv')
    dpnonOHCnonBin.loadFile('sendToWaelnonOHCnonBin.csv')
    '''
    if sex!=-1:
        dpOHC.filter('sex_1',[sex])
        dpOHCnonBin.filter('sex_1',[sex])
        dpnonOHC.filter('sex',[sex])
        dpnonOHCnonBin.filter('sex',[sex])

    features_datafile='run1OHC'
    features = np.array(pd.read_csv("Data/"+features_datafile+".csv", index_col=0).columns)
    colsnonOHC=distillTotal('run1OHC_GAD7_1','GAD7_1',features)
    colsOHCnonBin=importantFeatures('run1OHC_GAD7_1',dpOHCnonBin.df,'GAD7')
    colsOHC=importantFeatures('run1OHC_GAD7_1',dpOHC.df,'GAD7_1')

    OcolsOHC=dpOHC.df.columns.copy()
    OcolsOHCnonBin=dpOHCnonBin.df.columns.copy()
    OcolsnonOHC=dpnonOHC.df.columns.copy()
    OcolsnonOHCnonBin=dpnonOHCnonBin.df.columns.copy()


    for i in range(len(dpOHC.df.columns)):
        if (OcolsOHC[i] not in colsOHC) and (strip_underscores(OcolsOHC[i]) not in dpOHC.field_titles):
            dpOHC.remove(OcolsOHC[i])
        if (OcolsOHCnonBin[i] not in colsOHCnonBin) and (strip_underscores(OcolsOHCnonBin[i]) not in dpOHCnonBin.field_titles):
            dpOHCnonBin.remove(OcolsOHCnonBin[i])

    for i in range(len(dpnonOHC.df.columns)):
        if (OcolsnonOHC[i] not in colsnonOHC) and (strip_underscores(OcolsnonOHC[i]) not in dpnonOHC.field_titles):
            dpnonOHC.remove(OcolsnonOHC[i])
        if (OcolsnonOHCnonBin[i] not in colsnonOHC) and (strip_underscores(OcolsnonOHCnonBin[i]) not in dpnonOHCnonBin.field_titles):
            dpnonOHCnonBin.remove(OcolsnonOHCnonBin[i])

    if sex!=-1:
        if 'sex_1' in dpOHC.df.columns:
            dpOHC.remove('sex_1')
        if 'sex_1' in dpOHCnonBin.df.columns:
            dpOHCnonBin.remove('sex_1')
        if 'sex' in dpnonOHC.df.columns:
            dpnonOHC.remove('sex')
        if 'sex' in dpnonOHCnonBin.df.columns:
            dpnonOHCnonBin.remove('sex')

def applyNewFeatureSelection(dpOHC,dpnonOHC,dpOHCnonBin,dpnonOHCnonBin,sex=-1):
    
    dpOHC.loadFile('run1OHC.csv')
    dpnonOHC.loadFile('run1nonOHC.csv')
    dpOHCnonBin.loadFile('run1OHCnonBin.csv')
    dpnonOHCnonBin.loadFile('run1nonOHCnonBin.csv')

    if sex!=-1:
        dpOHC.filter('sex_1',[sex])
        dpOHCnonBin.filter('sex_1',[sex])
        dpnonOHC.filter('sex',[sex])
        dpnonOHCnonBin.filter('sex',[sex])

    features_file=open("newFeatures.txt","r")
    features = []
    for feature in features_file.readlines():
        features.append(feature.strip())
    features.append('GAD7')
    features=np.array(features)
    features_file.close()

    OcolsOHC=dpOHC.df.columns.copy()
    OcolsOHCnonBin=dpOHCnonBin.df.columns.copy()
    OcolsnonOHC=dpnonOHC.df.columns.copy()
    OcolsnonOHCnonBin=dpnonOHCnonBin.df.columns.copy()


    for i in range(len(dpOHC.df.columns)):
        if (strip_underscores(OcolsOHC[i]) not in features) and (strip_underscores(OcolsOHC[i]) not in dpOHC.field_titles):
            dpOHC.remove(OcolsOHC[i])
        if (strip_underscores(OcolsOHCnonBin[i]) not in features) and (strip_underscores(OcolsOHCnonBin[i]) not in dpOHCnonBin.field_titles):
            dpOHCnonBin.remove(OcolsOHCnonBin[i])

    for i in range(len(dpnonOHC.df.columns)):
        if (OcolsnonOHC[i] not in features) and (strip_underscores(OcolsnonOHC[i]) not in dpnonOHC.field_titles):
            dpnonOHC.remove(OcolsnonOHC[i])
        if (OcolsnonOHCnonBin[i] not in features) and (strip_underscores(OcolsnonOHCnonBin[i]) not in dpnonOHCnonBin.field_titles):
            dpnonOHCnonBin.remove(OcolsnonOHCnonBin[i])

    if sex!=-1:
        if 'sex_1' in dpOHC.df.columns:
            dpOHC.remove('sex_1')
        if 'sex_1' in dpOHCnonBin.df.columns:
            dpOHCnonBin.remove('sex_1')
        if 'sex' in dpnonOHC.df.columns:
            dpnonOHC.remove('sex')
        if 'sex' in dpnonOHCnonBin.df.columns:
            dpnonOHCnonBin.remove('sex')

def main():

    #----------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------

    #Data reading and processing

    directory_path = os.path.dirname(os.path.realpath(__file__))+'/' #Directory containing the code and the data
    statsPath = directory_path+'StatisticalAnalysis/' #path for the StatisticalAnalysis Results
    if not os.path.exists(statsPath):
        os.makedirs(statsPath)

    #datafile = "run1OHC" #name of the data file in the data folder
    #target = "GAD7_1" #name of the binarized dependent variable 


    #Specify which data file type youa are using
    
    #data = pd.read_csv(directory_path+"Data/"+datafile+".csv", index_col=0)

    #----------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------
    # 
    # #Classification

    #might have to change back to 'fork'
    #mp.set_start_method('spawn')

    #Keep the classification and feature selection methods that you want
    classifiers=['svm']#, 'LDA', 'rdforest', 'logreg']#, 'svm']
    #replace the # with the number of features you want
    fselect=['AllFeatures', 'reliefF_50', 'jmi_50', 'mrmr_50']     #'infogain_50','chisquare_50', 'fisher_50', 'fcbf', 'cfs'] 
    #fselect = ['AllFeatures', 'mrmr_50', 'chisquare_50']
    #Note that cfs and fcbf find all the significant features so they don't need a number

    n_seed = 5 #Number of validations
    splits = 10 #Number of folds or splits in each validation run
    

    #datafile = "sample"+str(2)+'_OHC'
    #target = "GAD7_1"
    #datafile = "run1OHC"
    #data = pd.read_csv(directory_path+"Data/"+datafile+".csv", index_col=0)
    #runs.NormalRun(directory_path, datafile, target, classifiers, fselect, n_seed, splits, classificationData=data,fselectData=data, doC=False,doF=True,cluster=False,fselectRepeat=50,cutoff=0.7,robustFeatures=25)
    
    #----------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------
    
    #Statistical Analysis

    dpOHC=data_processor()
    dpnonOHC=data_processor()
    dpOHCnonBin=data_processor()
    dpnonOHCnonBin=data_processor()
    
    
    #No Feature Selection Application
    
    dpOHC.loadFile('run1OHC.csv')
    dpnonOHC.loadFile('run1nonOHC.csv')
    dpOHCnonBin.loadFile('run1OHCnonBin.csv')
    dpnonOHCnonBin.loadFile('run1nonOHCnonBin.csv')
    '''
    dpOHC.loadFile('sendToWaelOHC.csv')
    dpnonOHC.loadFile('sendToWaelnonOHC.csv')
    dpOHCnonBin.loadFile('sendToWaelOHCnonBin.csv')
    dpnonOHCnonBin.loadFile('sendToWaelnonOHCnonBin.csv')
    '''

    #For Single-Sex Runs without feature selection
    '''
    sex=0
    dpOHC.filter('sex_1',[sex])
    dpOHCnonBin.filter('sex_1',[sex])
    dpnonOHC.filter('sex',[sex])
    dpnonOHCnonBin.filter('sex',[sex])
    dpOHC.remove('sex_1')
    dpOHCnonBin.remove('sex_1')
    dpnonOHC.remove('sex')
    dpnonOHCnonBin.remove('sex')
    '''

    #Feature Selection Application
    #applyFeatureSelection(dpOHC,dpnonOHC,dpOHCnonBin,dpnonOHCnonBin,sex=0)
    #applyFeatureSelection(dpOHC,dpnonOHC,dpOHCnonBin,dpnonOHCnonBin)

    #Uncomment the staistical test desired and pass the suitable parameters
    #sa=Stats(directory_path,statsPath,dpOHC=dpOHC,dpnonOHC=dpnonOHC,dpOHCnonBin=dpOHCnonBin,dpnonOHCnonBin=dpnonOHCnonBin)
    binary_target = "GAD7_1"
    nonbinary_target = "GAD7"

    
    columns=dpnonOHCnonBin.df.columns.copy()
    columns=columns[np.where(np.array(dpnonOHCnonBin.types)=='c')[0]]
    columns = columns.str.replace(' ', '_')
    columns = columns.str.replace('.', '_')
    
    '''
    #ANOVA_TODO={'rs228697':[0,1,2],'rs10462023':[0,1,2],'rs139459337':[2],'rs17031614':[0,2],'rs10838524':[1],'rs1801260':[0]}
    ANOVA_TODO={'rs228697_rs10462023':[0,5,7]}
    varlengths=dpnonOHCnonBin.levels
    for var in ANOVA_TODO:
        print(var)
        for j in ANOVA_TODO[var]:
            print(j)
            sa.ANOVA(dep = nonbinary_target, indep=[var], oneWay=True, followUp=True, oneVsOther={var:j})
    '''
    
    #sa=Stats(directory_path,statsPath,dpOHC=dpOHC,dpnonOHC=dpnonOHC,dpOHCnonBin=dpOHCnonBin,dpnonOHCnonBin=dpnonOHCnonBin)
    ANOVA_TODO={'rs228697_rs10462023':[6],'rs10462023_rs139459337':[7],'rs17031614_rs139459337':[4],'rs139459337_rs10838524':[8],'rs10462023_rs17031614':[4],'rs1801260_rs139459337':[4]}
    varlengths=dpnonOHCnonBin.levels
    tempnum=0
    for var in ANOVA_TODO:
        #print(var)
        for j in ANOVA_TODO[var]:
            #print(j)
            #ANOVA(dpnonOHCnonBin,dep = nonbinary_target, indep=[var,'sex'], oneWay=False, followUp=True, oneVsOther={var:j},num=tempnum)
            path1=#REDACTED PATH
            path2=#REDACTED PATH
            shutil.move(path1,path2)  
            tempnum+=1
    '''
    for var in columns:
        if var != 'GAD7' and var!='sex':
            sa.ANOVA(dep = nonbinary_target, indep=[var,'sex'], oneWay=False, followUp=True)
    '''



    #MuL Precalculations Data Prep
    '''
    for sex in [-1,0,1]:
        dpOHCnonBin.loadFile('run1OHCnonBin.csv')
        if sex==-1:
            applyFeatureSelection(dpOHC,dpnonOHC,dpOHCnonBin,dpnonOHCnonBin)
        else:
            applyFeatureSelection(dpOHC,dpnonOHC,dpOHCnonBin,dpnonOHCnonBin,sex=sex)

        PrecomputeMuL(dpOHCnonBin,target=nonbinary_target)

        if sex==-1:
            s='MF'
        elif sex==0:
            s='F'
        else:
            s='M'
        os.system("cp StepWiseRegData.csv Precomputed_StepWiseVarsData/MuLData_"+s+".csv")
        os.system("rm StepWiseRegData.csv")
    

    #OD Precalculations Data Prep
    
    for cutoff in [8,12,16]:
        for sex in [-1,0,1]:
            dpOHC.loadFile('run1OHC.csv')
            if sex==-1:
                applyFeatureSelection(dpOHC,dpnonOHC,dpOHCnonBin,dpnonOHCnonBin)
            else:
                applyFeatureSelection(dpOHC,dpnonOHC,dpOHCnonBin,dpnonOHCnonBin,sex=sex)
            #print(len(dpOHC.df))
            unmod=pd.read_csv('Data/unmodified_GAD7.csv',index_col=0)
            dpOHC.df['GAD7_1']=unmod.loc[dpOHC.df.index,'GAD7']
            dpOHC.df['GAD7_1']=(dpOHC.df['GAD7_1']>=cutoff).astype(int)

            PrecomputeOD(dpOHC,target=binary_target)

            if sex==-1:
                s='MF'
            elif sex==0:
                s='F'
            else:
                s='M'
            os.system("cp StepWiseRegData.csv Precomputed_StepWiseVarsData/ODData_"+s+"_"+str(cutoff)+".csv")
            os.system("rm StepWiseRegData.csv")
    '''

    #MuL calcualtions
    '''
    for sex in [-1,0,1]:
        dpOHCnonBin.loadFile('run1OHCnonBin.csv')
        if sex==-1:
            applyFeatureSelection(dpOHC,dpnonOHC,dpOHCnonBin,dpnonOHCnonBin)
        else:
            applyFeatureSelection(dpOHC,dpnonOHC,dpOHCnonBin,dpnonOHCnonBin,sex=sex)
        sa=Stats(directory_path,statsPath,dpOHC=dpOHC,dpnonOHC=dpnonOHC,dpOHCnonBin=dpOHCnonBin,dpnonOHCnonBin=dpnonOHCnonBin)
        MultiReg = sa.Multivariate_Reg(target=nonbinary_target, stepwise = True)
        if sex==-1:
            s='MF'
        elif sex==0:
            s='F'
        else:
            s='M'
        os.system("cp -r StatisticalAnalysis/MultiVarRegression cluster_results/MuL_results/MultiVarRegression_"+s)
        os.system("rm -r StatisticalAnalysis/MultiVarRegression")
        
        
    #Odds Ratio Calculations
    
    for cutoff in [8,12,16]:
        for sex in [-1,0,1]:
            dpOHC.loadFile('run1OHC.csv')
            if sex==-1:
                applyFeatureSelection(dpOHC,dpnonOHC,dpOHCnonBin,dpnonOHCnonBin)
            else:
                applyFeatureSelection(dpOHC,dpnonOHC,dpOHCnonBin,dpnonOHCnonBin,sex=sex)
            print(len(dpOHC.df))
            unmod=pd.read_csv('Data/unmodified_GAD7.csv',index_col=0)
            dpOHC.df['GAD7_1']=unmod.loc[dpOHC.df.index,'GAD7']
            dpOHC.df['GAD7_1']=(dpOHC.df['GAD7_1']>=cutoff).astype(int)
            #print(np.sum(dpOHC.df['GAD7_1']))
            sa=Stats(directory_path,statsPath,dpOHC=dpOHC,dpnonOHC=dpnonOHC,dpOHCnonBin=dpOHCnonBin,dpnonOHCnonBin=dpnonOHCnonBin)
            oddsRatios = sa.Odds_Ratios(target=binary_target, stepwise=True)
            if sex==-1:
                s='MF'
            elif sex==0:
                s='F'
            else:
                s='M'
            os.system("cp -r StatisticalAnalysis/OddsRatios cluster_results/OD_results/OddsRatios_"+s+"_"+str(cutoff))
            os.system("rm -r StatisticalAnalysis/OddsRatios")
    '''

    #Univariate Odd Ratio Calculations
    '''
    indeps=['rs228697_2.rs10462023_2','rs10462023_1.rs139459337_2','rs17031614_0.rs139459337_2','rs139459337_2.rs10838524_1','rs10462023_0.rs17031614_2','rs1801260_0.rs139459337_2']
    
    for cutoff in [8,12,16]:
        for sex in [-1,0,1]:
            if sex==-1:
                s='MF'
            elif sex==0:
                s='F'
            else:
                s='M'
            filename='OD_results/Uni_OddsRatios_'+s+'_'+str(cutoff)+'.txt'
            sys.stdout=open(filename,'w')

            dpOHC.loadFile('run1OHC.csv')
            if sex!=-1:
                dpOHC.filter('sex_1',[sex])
                dpOHC.remove('sex_1')
            unmod=pd.read_csv('Data/unmodified_GAD7.csv',index_col=0)
            dpOHC.df['GAD7_1']=unmod.loc[dpOHC.df.index,'GAD7']
            dpOHC.df['GAD7_1']=(dpOHC.df['GAD7_1']>=cutoff).astype(int)
            for indep in indeps:
                try:
                    y=dpOHC.df['GAD7_1'].copy(deep=True)
                    X=dpOHC.df[indep].copy(deep=True)
                    X=sm.add_constant(X)
                    log_reg = GLM(y, X, family=families.Binomial()).fit()
                    model_odds = pd.DataFrame(np.exp(log_reg.params), columns= ['Odds Ratio'])
                    model_odds['p-value']= log_reg.pvalues
                    model_odds[['2.5%', '97.5%']] = np.exp(log_reg.conf_int())
                    print(model_odds)
                    print('----------------------------------------------------------------------------------------\n\n')
                except ValueError:
                    continue
            sys.stdout=sys.__stdout__
    '''

    #assAnalyis = sa.Association_Analysis(oneTarget=True, target=nonbinary_target)

    #ARL
    '''
    for sex in range(2):
        for i in range(10):
            dpOHC.loadFile('run1OHC.csv')
            dpOHC.filter('sex_1',[sex])
            dpOHC.remove('sex_1')
            dpOHC.categoryUndersample('GAD7_1',n=10000)
            tempColumns=dpOHC.df.columns.copy()
            num=dpOHC.num_fields
            for q in range(num):
                feature=tempColumns[q]
                if feature not in ['GAD7_1']:#,'Chronotype_1','Chronotype_2','Chronotype_3']:
                    dpOHC.remove(feature)
            #dpOHC.df=pd.read_csv('ARL_Results_1/ARL_Info/ARL_'+str(i)+'.csv',index_col=0)
            dpOHC.df.to_csv('ARL_Results_'+str(sex)+'/ARL_Info/ARL_'+str(i)+'.csv')
            sa=Stats(directory_path,statsPath,dpOHC=dpOHC,dpnonOHC=dpnonOHC,dpOHCnonBin=dpOHCnonBin,dpnonOHCnonBin=dpnonOHCnonBin)
            assRuleLearning = sa.Association_Rule_Learning(rhs = 'GAD7_1',max_items=4,min_confidence=0.001,min_support=0.00001, protective=False, min_lift=1.05)
            os.system("cp StatisticalAnalysis/Apriori/Risk/AssociationRules.txt ARL_Results_"+str(sex)+"/ARL_"+str(i)+".txt")
            os.system("rm -r StatisticalAnalysis/Apriori")
    '''    
    

    
    #indeps=['rs139459337_2_rs10838524_1'] #female
    #indeps=['rs17031614_0_rs139459337_2'] #males
    #columns= dpOHCnonBin.df.columns.copy()
    #columns = columns.str.replace(' ', '_')
    #columns = columns.str.replace('.', '_')
    #print(columns)

    #indeps=['rs228697_rs10462023','rs10462023_rs139459337','rs17031614_rs139459337','rs139459337_rs10838524','rs10462023_rs17031614','rs1801260_rs139459337']
    
    #Mediation Analysis
    
    #indeps=['rs228697_2_rs10462023_2','rs10462023_1_rs139459337_2','rs17031614_0_rs139459337_2','rs139459337_2_rs10838524_1','rs10462023_0_rs17031614_2','rs1801260_0_rs139459337_2']
    '''
    overall_indeps={"F_8":["rs228697_1.rs139459337_2","rs139459337_2.rs10838524_1"],"F_12":["rs228697_0.rs139459337_2","rs228697_1.rs139459337_2","rs139459337_2.rs10838524_2","rs139459337_2.rs10838524_1"],"F_16":["rs228697_2.rs2287161_2"],"M_8":["rs17031614_0.rs139459337_2"],"M_12":[],"M_16":["rs228697_2.rs2287161_0","rs139459337_2.rs10838524_1"]}
    
    for cutoff in [8,12,16]:
        for sex in [0,1]:

            if sex==0:
                s='F'
            else:
                s='M'

            indeps=overall_indeps[s+"_"+str(cutoff)]
            if len(indeps)==0:
                continue

            dpOHC.loadFile('run1OHC.csv')
            dpOHC.filter('sex_1',[sex])
            dpOHC.remove('sex_1')

            unmod=pd.read_csv('Data/unmodified_GAD7.csv',index_col=0)
            dpOHC.df['GAD7_1']=unmod.loc[dpOHC.df.index,'GAD7']
            dpOHC.df['GAD7_1']=(dpOHC.df['GAD7_1']>=cutoff).astype(int)


            chronotype=np.array([1 for i in range(len(dpOHC.df))])
            indices=dpOHC.df.index
            for i in range(len(indices)):
                for j in range(1,4):
                    if dpOHC.df.at[indices[i],'Chronotype_'+str(j)]==1:
                        chronotype[i]=0
            dpOHC.df['Chronotype_0']=chronotype
            dpOHC.df.to_csv("Data/mediationData.csv")


            for mediator in ['Chronotype_0','Chronotype_3']:
                for indep in indeps:
                    #sa.Mediation_Analysis(dep = 'GAD7_1', mediator=mediator, indep = indep, sims=100)
                    file=open("mediation.txt","w")
                    file.close()
                    os.system("Rscript mediationAnalysis.R "+indep+" "+mediator+" GAD7_1 "+"Ever.addicted.to.any.substance.or.behaviour_1+Age.at.recruitment+Average.total.household.income.before.tax_1+Average.total.household.income.before.tax_2+Average.total.household.income.before.tax_3+Average.total.household.income.before.tax_4+Townsend.deprivation.index.at.recruitment")

                    #os.system("cp mediation.txt mediationAnalysisResults/"+indep+"+"+mediator+"~GAD7_1: "+s+"-"+str(cutoff)+".txt")
                    #os.system("rm mediation.txt")

                    path1=#REDACTED PATH
                    path2=#REDACTED PATH
                    shutil.move(path1,path2)
    '''

    #continuous Chronotype Mediation Analysis
    '''
    for cutoff in [8,12,16]:
        for sex in [0,1]:

            if sex==0:
                s='F'
            else:
                s='M'

            indeps=overall_indeps[s+"_"+str(cutoff)]
            if len(indeps)==0:
                continue

            dpOHC.loadFile('run1OHC.csv')
            dpOHC.filter('sex_1',[sex])
            dpOHC.remove('sex_1')

            dpnonOHC.loadFile('run1nonOHC.csv')
            dpnonOHC.filter('sex',[sex])

            unmod=pd.read_csv('Data/unmodified_GAD7.csv',index_col=0)
            dpOHC.df['GAD7_1']=unmod.loc[dpOHC.df.index,'GAD7']
            dpOHC.df['GAD7_1']=(dpOHC.df['GAD7_1']>=cutoff).astype(int)

            dpOHC.df.drop(columns=['Chronotype_1','Chronotype_2','Chronotype_3'],inplace=True)
            dpOHC.df['Chronotype']= dpnonOHC.df['Chronotype'].copy(deep=True)

            dpOHC.df.to_csv("Data/mediationData.csv")
            
            mediator="Chronotype"
            for indep in indeps:
                file=open("mediation.txt","w")
                file.close()
                os.system("Rscript mediationAnalysis.R "+indep+" "+mediator+" GAD7_1 "+"Ever.addicted.to.any.substance.or.behaviour_1+Age.at.recruitment+Average.total.household.income.before.tax_1+Average.total.household.income.before.tax_2+Average.total.household.income.before.tax_3+Average.total.household.income.before.tax_4+Townsend.deprivation.index.at.recruitment")

                path1=#REDACTED PATH
                path2=#REDACTED PATH
                shutil.move(path1,path2)   
    '''

    #sa.Mendelian_Ranomization(dep = 'GAD7', indep = 'Townsend_deprivation_index_at_recruitment', inst='rs10462020_2_rs17031614_1')
    


#TEMP functions
def twoWay_ANOVA(data, dep, indep, alpha, between, followUp, OGdata,num=0):
        
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
    

    new_indep=relabel(indep[0])
    data=data.rename(columns={indep[0]:new_indep},inplace=False)
    OGdata=OGdata.rename(columns={indep[0][:indep[0].find('_')]:relabel(indep[0][:indep[0].find('_')]),indep[0][indep[0].find('_')+1:indep[0].rfind('_')]:relabel(indep[0][indep[0].find('_')+1:indep[0].rfind('_')])},inplace=False)
    indep=[new_indep,indep[1]]
    print(indep)

    '''
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


    unmod=pd.read_csv('Data/unmodified_GAD7.csv',index_col=0)
    data['GAD7']=unmod.loc[data.index,'GAD7']
    data.to_csv("Data/SRHData"+str(num)+".csv")
    
    ###New SRH stuff
    file=open("SRH.txt","w")
    file.close()
    print(indep)
    os.system("Rscript SHR.R "+dep+" "+indep[0]+" "+indep[1])
    '''




def ANOVA(dpnonOHCnonBin,dep, indep, alpha = 0.05, oneWay = True, followUp = False, oneVsOther = dict(), oneVsAnother =dict(),num=0):
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
    data = dpnonOHCnonBin.df.copy(deep = True)
    data.columns = data.columns.str.replace(' ', '_')
    data.columns = data.columns.str.replace('.', '_')
    types=np.array(dpnonOHCnonBin.types)
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


    if not oneVsOther:
        data = data.astype({indep[0]:'int64'})
        data = data.astype({indep[1]:'int64'})
    else:
        data = data.astype({indep[0]:'str'})
        data = data.astype({indep[1]:'str'})
    return twoWay_ANOVA(data, dep, indep, alpha, False, followUp, OGdata,num=num)

main()



#print(indep)
'''
if not os.path.exists(self.statsPath+'twoWayANOVA'):
    os.makedirs(self.statsPath+'twoWayANOVA')
fname = indep[0] + '_' + indep[1]
if not os.path.exists(self.statsPath+'twoWayANOVA/'+fname):
    os.makedirs(self.statsPath+'twoWayANOVA/'+fname)
currPath = self.statsPath+'twoWayANOVA/'+fname+'/'

formula = dep + ' ~ C(' + indep[0] + ') + C(' + indep[1] + ') + C(' + indep[0] + '):C(' + indep[1] + ')'
twoWayANOVA = open(currPath+'twoWayANOVA_summary.txt', 'w')
twoWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
'''
