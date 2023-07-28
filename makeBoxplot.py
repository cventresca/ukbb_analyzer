import matplotlib.pyplot as plt
import numpy as np
from utilities import relabel
import pandas as pd
from data_processor import data_processor
import seaborn as sns
import os
import shutil
from pathlib import Path


def plotComboBox(indep,dep,data,OGdata,one_v_rest=2):
    unmod=pd.read_csv('Data/unmodified_GAD7.csv',index_col=0)
    data=data.copy(deep=True)
    data[dep]=unmod.loc[data.index,dep]
    OGdata[dep]=unmod.loc[data.index,dep]
    #print(data.columns)
    grid=plt.GridSpec(1,39,wspace=0,hspace=0.3)
    ax1 = plt.subplot(grid[0, 0:14])
    ax2=plt.subplot(grid[0, 19:29],sharey=ax1)
    ax3=plt.subplot(grid[0, 29:39],sharey=ax1)


    data['sex']=data['sex'].astype(str)
    for i in range(len(data)):
        if data['sex'].iat[i]=='0.0':
            data['sex'].iat[i]='Female'
        else:
            data['sex'].iat[i]='Male'


    mapping={'PER3A':['C','G'],'PER3C':['T','G'],'PER2':['G','A'],'CLOCK3111':['A','G'],'PER3B':['G','A'],'ZBTB20':['C','T'],'CRY2':['A','G'],'CRY1':['C','G']}
    SNPs=['00','01','10','11','02','20','22','12','21']
    
    #print("\n\n"+str(one_v_rest))
    label_maker=SNPs[one_v_rest]
    all=[indep[:indep.find('_')],indep[indep.find('_')+1:]]
    all_old=all.copy()
    all.sort()
    if all!=all_old:
        label_maker=label_maker[1]+label_maker[0]

    first=all[0]
    second=all[1]
    #print(label_maker)
    if int(label_maker[0])==0:
        first_label=mapping[first][0]+mapping[first][0]
    elif int(label_maker[0])==1:
        first_label=mapping[first][0]+mapping[first][1]
    else:
        first_label=mapping[first][1]+mapping[first][1]
    
    if int(label_maker[1])==0:
        second_label=mapping[second][0]+mapping[second][0]
    elif int(label_maker[1])==1:
        second_label=mapping[second][0]+mapping[second][1]
    else:
        second_label=mapping[second][1]+mapping[second][1]
        
    label=first_label+'-'+second_label
    


    type_label=first+'-'+second
    data[type_label]=((data[indep]==one_v_rest)*1)
    data[type_label]=data[type_label].astype(str)
    for i in range(len(data)):
        if data[type_label].iat[i]=='0':
            data[type_label].iat[i]='Other'
        else:
            data[type_label].iat[i]=label

    hue_order=[label,'Other']
    sns.barplot(x='sex',y="GAD7",data=data,hue=type_label,hue_order=hue_order,palette=["gray", "lightgray"],ax=ax1,errwidth=1.5,capsize=0.1)#,sym="")
    ax1.set_xlabel('Sex',fontsize=13)
    ax1.set_ylabel("GAD7",fontsize=13)



    data[first]=data[first].astype(str)
    for i in range(len(data)):
        if data[first].iat[i]=='0':
            data[first].iat[i]=mapping[first][0]+mapping[first][0]
        elif data[first].iat[i]=='1':
            data[first].iat[i]=mapping[first][0]+mapping[first][1]
        else:
            data[first].iat[i]=mapping[first][1]+mapping[first][1]


    data[second]=data[second].astype(str)
    for i in range(len(data)):
        if data[second].iat[i]=='0':
            data[second].iat[i]=mapping[second][0]+mapping[second][0]
        elif data[second].iat[i]=='1':
            data[second].iat[i]=mapping[second][0]+mapping[second][1]
        else:
            data[second].iat[i]=mapping[second][1]+mapping[second][1]

    

    Females=data[data['sex']=='Female']
    Males=data[data['sex']=='Male']

    order=[mapping[first][0]+mapping[first][0],mapping[first][0]+mapping[first][1],mapping[first][1]+mapping[first][1]]
    hue_order=[mapping[second][0]+mapping[second][0],mapping[second][0]+mapping[second][1],mapping[second][1]+mapping[second][1]]
    
    #Females
    sns.barplot(x=first,y="GAD7",data=Females,order=order,hue=second,hue_order=hue_order,palette=["darkgray", "gray", "lightgray"],ax=ax2,errwidth=1.5,capsize=0.1)#,sym="")
    ax2.legend([],[],frameon=False)
    ax2.set_title('Female')
    ax2.set_xlabel(first,fontsize=13)
    ax2.xaxis.set_label_coords(1,-0.08)
    ax2.set_ylabel("GAD7",fontsize=13)

    #Males
    sns.barplot(x=first,y="GAD7",data=Males,order=order,hue=second,hue_order=hue_order,palette=["darkgray", "gray", "lightgray"],ax=ax3,errwidth=1.5,capsize=0.1)#,sym="")
    #ax3.set_yticks([])
    ax3.set_title('Male')
    ax3.tick_params(labelleft=False)
    ax3.set(ylabel=None,xlabel=None)

    #setting ticks for all
    '''
    grouped=data.groupby([first,second])['GAD7']
    means=np.array(grouped.mean())
    stds=np.array(grouped.std())
    max=np.max(means+stds)
    increment=round(0.9*max/4)
    ax1.set_yticks([0,increment*1,increment*2,increment*3,increment*4])
    '''
    yticks1=np.array(ax1.get_yticks())
    max=np.max(yticks1)
    increment=max/2
    ax1.set_yticks([round(0,1),round(increment*1,1),round(increment*2,1)])

    plt.savefig("testBoxPlot1.png")
    


def ANOVA(dpnonOHCnonBin, dep, indep, one_v_rest):
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


    new_indep=relabel(indep[0])
    data=data.rename(columns={indep[0]:new_indep,indep[0][:indep[0].find('_')]:new_indep[:new_indep.find('_')],indep[0][indep[0].find('_')+1:]:new_indep[new_indep.find('_')+1:]},inplace=False)
    OGdata=OGdata.rename(columns={indep[0][:indep[0].find('_')]:relabel(indep[0][:indep[0].find('_')]),indep[0][indep[0].find('_')+1:indep[0].rfind('_')]:relabel(indep[0][indep[0].find('_')+1:indep[0].rfind('_')])},inplace=False)
    indep=new_indep
    plotComboBox(indep,dep,data,OGdata,one_v_rest=one_v_rest)





nonbinary_target = "GAD7"

dpnonOHCnonBin=data_processor()
dpnonOHCnonBin.loadFile('run1nonOHCnonBin.csv')


SNPs=['00','01','10','11','02','20','22','12','21']
ANOVA_TODO={'rs139459337_rs10838524':[6,8],'rs228697_rs139459337':[4,7],'rs228697_rs2287161':[5,6],'rs17031614_rs139459337':[4],'rs10462023_rs139459337':[7],'rs1801260_rs139459337':[4]}
for var in ANOVA_TODO:
    #print(var)
    for j in ANOVA_TODO[var]:
        ANOVA(dpnonOHCnonBin=dpnonOHCnonBin, dep = nonbinary_target, indep=[var], one_v_rest=j)
        path1=#REDACTED PATH
        path2=#REDACTED PATH
        shutil.move(path1,path2)



















'''
tips = sns.load_dataset("tips")
#axs2 = plt.subplot(grid[0, 5:6])
sns.boxplot(x="day", y="total_bill",
        hue="smoker", palette=["k", "gray"],
        data=tips)#,ax=axs2)
#print(tips)
print(tips)
plt.show()
'''
#plt.suptitle(indep[:indep.find('and')]+'-'+indep[indep.find('and')+3:],fontsize=18)
#dd=pd.melt(data,id_vars=['sex'],value_vars=["type"],var_name='types')
#print(dd)
