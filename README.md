# ukbb_analyzer
Notes: 
- (DNC) indicates that a function should not be called directly
- (MOD) indicates that a function can/should be directly modified but not called
- (SAME) indicates that function must be called in the same session as its inverse function
- (REV) indicates that an action is reversable
## Data Processor
__init__(self)
- Parameters: no parameters are necessary in the instantiation of a data processor object
### Purpose
| Gather data from uk biobank manipulate it in order to prepare for analysis
### Attributes
 - types: list of data types for each clinical field in order ('c'=categorical,'g'=gaussian/continuous)
 - levels: list of category counts for each clinical field in order (1 for all continuous variables)
 - OHC_aplied_to_fields: boolean concerning whether or not one-hot-encoding has been applied to the clinical fields
 - OHC_aplied_to_genes: boolean concerning whether or not one-hot-encoding has been applied to the SNPs
 - codes: ID for encoding file to change categorical variables to different assignments (None for all other variables)
 - num_fields: the number of clinical variables (including OHC-originating variables)
 - num_SNPs: the number of genetic variables (including OHC-originating variables, excluding two-way interaction variables)
 - df_purged: boolean concerning whether or not the data has undergone the NaN purging process
 - df_scaled: boolean concerning whther or not the data has undergone the scaling process for continuous variables
 - df_shifted: boolean concerning whther or not the data has undergone the shifting process for categorical variables
 - field_titles: labels for all non-OHC clinical field variables
 - rsid_titles: labels for all non_OHC SNP variables
 - object_occupied: boolean concerning whether or not there is data currently in the data processor
 - df: DataFrame in which the data is stored
 - storage: where data is stored after a variable is removed or categorized
 - scaled: where data is stored when continuous variables are scaled
### Functions

#### prep()
| Contains the information about the desired fields. (MOD)
#### getInformation(types,levels,codes,fields_prep='standard',include_imputed=False,binary=False,GAD7_cutoff=8,PHQ9_cutoff=10)
| Gathers primary information (clinical fields and non-imputed SNPs) and fills out processor fields
- types: list of data types for each clinical field in order ('c'=categorical,'g'=gaussian/continuous)
- levels: list of category counts for each clinical field in order (1 for all continuous variables)
- codes: ID for encoding file to change categorical variables to different assignments (None for all other variables)
- fields_prep: 'GAD7' to calculate add GAD7 scores, 'PHQ9' for PHQ9 scores, 'both' for both, and 'standard' for neither
- binary: boolean concerning whether or not GAD7/PHQ9 scores are binary
- GAD7_cutoff: binary cuttoff (included in upper portion) for GAD7 binarization
- PHQ9_cutoff: binary cuttoff (included in upper portion) for PHQ9 binarization
#### addPairwise()
| Adds two-way interactions for currently-included SNPs
#### containsNaN(test)
| Checks vector for NaN values (DNC)
#### addImputedInformation()
| Adds imputed genetic information for all participants
#### getGeneticInformation(chroms,rsids,include_imputed)
| Retrieves genetic data for participants (DNC)
- chroms: list of chromosomes
- rsids: list of lists of rsids, each inner list corresponding to a chromosome
- include_imputed: boolean indicating whether or not to imclude imputed genetic data
#### featureUndersample(reference,n=1000)
| Undersample in proportion to a particular categorical variable
- reference: categorical variable to undersample with respect to
- n: number to undersample to
#### undersample(n=-1)
| Randomized undersampling
- n: number to undersample to
#### kmeansUndersample(num_clusters=3,n=1000)
| Randomly sample in proportion to clusters found through kmeans clustering
- num_clusters: number of clusters
- n: number to undersample to
#### calcBoth(test1,binary,GAD7_cutoff,PHQ9_cutoff)
| Calculate and add GAD7 and PHQ9 scores for all participants (DNC)
- test1: dataframe of clinical features
- binary: boolean concerning whether or not GAD7/PHQ9 scores are binary
- GAD7_cutoff: binary cuttoff (included in upper portion) for GAD7 binarization
- PHQ9_cutoff: binary cuttoff (included in upper portion) for PHQ9 binarization
#### calcPHQ9(test1,binary,binary_cutoff)
| Calculate and add PHQ9 scores for all participants (DNC)
- test1: dataframe of clinical features
- binary: boolean concerning whether or not PHQ9 scores are binary
- binary_cutoff: binary cuttoff (included in upper portion) for PHQ9 binarization
#### calcGAD7(test1,binary,binary_cutoff)
| Calculate and add GAD7 scores for all participants (DNC)
- test1: dataframe of clinical features
- binary: boolean concerning whether or not GAD7 scores are binary
- binary_cutoff: binary cuttoff (included in upper portion) for GAD7 binarization
#### split(returnFrac)
| Remove and return a fraction of the pariticipants
- returnFrac: fractions of the participants to remove and return
#### shift()
| Use encoding schemes from 'encodingSchemes.txt' to change categorical labels in a predetermined manner
#### unscale() 
| Reverse the effects of scale() (SAME)
#### scale()
| Normalize all continuous variables so that they are between 0 and 1 (REV)
#### minmax(L)
| Finds and returns the minimum and maximum values of list-like data (DNC)
- L: list-like data
#### purge()
| Remove all participants that have a NaN value in any column
#### handle_bools_None(s)
| Convert a string version of a boolean or None object to their python equivalents (DNC)
- s: string to convert
#### buildExtension(num_categories,baseline,target)
| Helper function for OHC that builds extension to a subjects data (DNC)
- num_categories: number of categories for variable
- baseline: indicate the category from which deviation is measured
- target: category of subject for who extension is being built
#### OHC(baselines=None,encode_fields=True,encode_genes=True)
| One-hot-encodes all categorical variables
- baselines: custom baselines for OHC - baselines for the categorical variables only, mainting the same order
- encode_fields: boolean concerning whether or not to OHC categorical field variables
- encode_genes: boolean concerning whether or not to OHC genetic variables
#### loadMeta(filename)
| Load metadata from file (DNC)
- filename: name for file containing metadata
#### loadFile(filename='working_df.csv')
| Load data from file
- filename: name for file containing data
#### makeMeta(filename)
| Save metadata into file (DNE)
- filename: name for file which metadata is to be put into
#### makeFile(filename='working_df.csv')
| Save data into file
- filename: name for file which data is to be put into
#### remove(feature)
| Remove a variable from the data (REV)
- feature: variable to be removed from the data
#### restore(feature)
| Undo an action taken on a feature such as categorization or removal (SAME)
- feature: variable to be restored to its previous form
#### categorize(feature,ranges)
| Categorize a continuous variable through the use of 'bins' in which ranges of values are assigned
- feature: variable to be categorized
- ranges: list-like object of list-like ranges (inclusive) with the index of each range serving as its assigned category
#### reset()
| Reset all instance variables to their base states (DNC)
















## Network Analyzer
__init__(self,dp,target=None)
- dp: data_processor object with data from which networks are to be constructed
- target: feature with which to color a subject network
### Purpose
| Create and analyze subject and feature networks
### Attributes
- dp: data_processor object with data from which networks are to be constructed
- target: feature with which to color a subject network
- dfmat: edge weight matrix created by mixedNetwork() or subjectNetwork()
- graph: NetworkX graph created from dfmat
- cat_masks: booelan array concerning whether or not each variables is categorical
### Functions

#### createNetwork(isSubject)
| Creates NetworkX graph using the edge weight matrix stored in dfmat (attribute)
- isSubject: boolean concering whether or not it is a subject Network being created (as opposed to a feature network)
#### mixedNetwork(title='mixeddfmat.csv')
| creates an edge weight matrix using mutual information analysis between the features in self.dp.df
- title: title of file into which to save the resulting edge weight matrix - in addition to it being stored in the calling network_analyzer object
#### breakdown()
| Breaks down features into categorical and continuous values for each subject (DNC)
#### subjectNetwork(title='subjectdfmat.csv')
| creates an edge weight matrix using HEOM distance metric between subjects in self.dp.df
- title: title of file into which to save the resulting edge weight matrix - in addition to it being stored in the calling network_analyzer object
#### featureAnalysis()
| Analyzes graph in self.graph according to a variety of metrics relevant to a feature network
#### subjectAnalysis()
| Analyzes graph in self.graph according to a variety of metrics relevant to a subject network





## UKBB_Analyzer.py
  - The main file of the package that will be used to run all the different analyses.
  - This should be the only file that NEED to be modified to use the package -- you're welcome to navigate the other files,
    but the main functionalities should be easily accessible from here.
  - For now, the available analysis to be conducted are Feature Selection, Classification, and Statistical Analysis. 
    Feature Selection & Classification can be run by calling NormalRun in kFoldValidationRuns class. To conduct Statistical Analysis, 
    create a Stats object and call the appropriate function with the appropriate data, detailed descriptions of that 
    can be found below description of StatisitcalAnalysis.py
    
### Function: main()
  - calls other functions to analyze the data





## kFoldValidationRuns.py
  - main getway to run the feature selection and classifications
  - depending on the parameters passed will run feature selection, classification, or both
  - The feature selection and classification runs are parelized as they take a long time. If the parallelization 
    is causing issues, it probably can be traced back to this file.
  - Note: the data that gets passed to this file to be used for classification and feature selection has to be
    one-hot-encoded with categorical outcome
 
### Function: NormalRun()
  - runs either feature selection, classification, or both
  - attributes are broken down in the function header
  
  
  
  
  
## RunFeatureSelection.py
  - the file running the feature selection with bootstrapping
  - the resutls of these runs are saved in xxxxx
  - Note: this file doesn't combine the results of the bootstrap runs -- this happens in ClassFeatureStats.py
  
### Function: fselectNew()
  - main function to run the feature selection with bootstrapping
  - the function creates a list of tuples that contains the input for the helper function used to run the feature selections
  - the tuples mainly contain the feature selection method and a number ranging from 1 to fselectRepeat passed to NormalRun, 
    which would represent the random_state used to resample the data for bootstrapping
  - attributes are broken down in the function header
  
### Function: fsNewHelper()
  - helper function for fselectNew() that takes a tuple and does data resampling based on the content of the tuple.
  - After resampling, important features are found for that run based on the feature selection method passed 
    and the output is saved in a txt file in xxxxx
  - The function that does the selection of the indices of important features is run_feature_selection, which is called by fsNewHelper
  - attributes are broken down in the function header

### Function: run_feature_selection()
  - this function receives X and y which would be predictive features and outcome variable as well as a string 
    representing the feature selection method
  - based on the string representing the feature selection method, another function would be called with X and y, 
    which will return the indices of the selected features
    
 ### Function: fisher_exact_test(), cfs(), merit_calculation(), fcbf(), reliefF(), infogain(), fcbfHelper(), jmi(), mrmr(), chisquare(), su_calculation(), mutual_information(), conditional_entropy(), entropy()
  - These are the functions doing the feature selection or helper functions for them
  - There should be no need to modify any of these functions, but they include detailed description of what they are doing 
    and the parameters needed in the python code


    
    
    
## RunClassifiers.py
  - the file running each of the specified classifiers with each of the feature selection methods to produce a heatmap
  - the resutls, which include the confusion matrices, of these runs are saved in xxxxx
  - Note: this file doesn't produce the heatmaps -- this happens in ClassFeatureStats.py

### Function: classify()
  - The main function of the file -- gets called to run the classification'
  - The function recieves a combination of classification and feature selection method as well as the data.
  - The function breaks down the data into train and testing, and then KFolds cross validation is being conducted on the train
    data, breaking it to train and validation sets. on each of the KFold runs, Baysian search is used for hyperparameter optimization
  - After finding the best parameters (depending on optimizing F1 or Accuracy which can be changed), the classifier is being run
    on the test data that was split before running KFold cross validation.
  - This whole processing of train/validation/test split is being repeated n_seed times (n_seed gets passed in UKBB_Analyzer)
  - attributes are broken down in the function header

### Function: getParameters()
  - A helper function for classify() that return a dictionary with hyperparameters and the values that Baysian search
    should look through for the optimization
    
### Sub-Class: MyClassifier()
  - A sub class that will be used to run the classification runs that gets called from the Baysian search function
  - The parameter includes an estimator, based on which an instance of the appropratie classifier will be created, and 
    a method, based on which a subset of features chosed by that method (fselect method) will be used.
  - The class has default values for all the hyperparameters used by all the classifiers, of which the appropraite ones are
    used to create instance of a new classifier.
  - After that, set_params() function is used to update these parameters during each run of the Baysian search
  - Note: what is being optimized in these runs is what's being returned by the score() function, and if a differnt scoring
    metric is desired, it should be calculated using standrded python libraries and returned (almost every scoring metric can
    be calculated by only using y and yp -- which are the predicted values for y)
  - All the functions in the class are either writing already exisiting functions of the classifiers to do the same 
    job but with some modification to fit the code -- detailed description of each of the functions is included in the code
  - Attributes are broken down in the class header





## StatisticalAnalysis.py
  - The main file for running all the stastical analysis on UKBiobank data
  - currently, the following statistics are fully implemented: Mediation Analysis, Mendelian Randomization, Odds Ratios (includes
    univariate logistic regression), Multivariate Linear Regression, one & two way ANOVA, Association Rule Learning,
    Association Analysis (G-test of independence, Fisher exact test, and Chi-Square test of independence).
    
### Sub-Class: Stats()
  - The main class in StasticalAnalysis.py underwhich all the function are written
  - To initiate an instance of the class, the directory_path as well as StasticalAnalysis path (both are created 
    in UKBB_Analyzer.py) need to be passed.
  - The class has an instance variable for each of the tests which would store the test results if they needed
    to be accessed later (note: all the results are wrtiien to the StasticalAnalysis folder, so usually theses instance variables
    won't be needed unless the results are needed to run further analysis on it)
  - The attributes needed for each of the functions under Stats() are described in details in the python file. It must be
    noted, though, that all the function recieves a data parameter which represent the data to run the analysis on. The structure
    of this data differs from one type of analysis to another. For example, outcome variable has to be continuous for linear
    regression, but it has to be categorical for logistic regression (odds ratios). There will be a discription below of the type
    of data needed to run each stastical test.
    
#### Attributes
  - path: directory path that contains the code
  - statsPath: file path of stasticalAnalysis directory
  - dpOHC: version of the data that's one hot encoded with binary/categorical outcome variable
  - dpnonOHC: version of the data that's not one hot encoded with binary/categorical outcome variable
  - dpOHCnonBin: version of the data that's one hot encoded with continuous outcome variable
  - dpnonOHCnonBin: version of the data that's not one hot encoded with continuous outcome variable
  - ARL: dictionary to store Association Rule Learning results
  - mediation: dictionary to store mediation analysis results
  - mendelian: dictionary to store mendelian randomization results
  - linearRegression: dictionary to store multivariate logistic regression results
  - logisticRegression: dictionary to store odds ratio results
  - oneANOVA: dictionary to store one-way ANOVA results
  - twoANOVA: dictionary to store two-way ANOVA results
  - AssocAnalysis: dictionary to store association analysis results
    
#### Sub-Class Function: Association_Rule_Learning()
  - Does Association Rules mining for the items within the passed dataframe. Write all the found association rules that meet the 
    specified conditions and save the produced graphs in the passed parameters to AssociationRules.txt in Apriori Folder in statsPath.
  - After finding the rules that meets the specified requirment, the code finds the odds ratios of the left-hand-side combinations
    of the rules as well as the p-value and confidence interval and write the results down to the same file.
  - Because ARL usually identifies risk factors (lhs that increase chances of observing rhs), there is a parameter -- protective --
    that if set to True, the values of the rhs would be flipped (what is 1 becomes 0 and what is 0 becomes 1), so running the 
    analysis identifies rules that increases chances of finding 0 or protective factors.
  - the ARL dictionary of the Stats object stores either the risk-rules in "ARLRisk" key or protective-rules in "ARLProtect"
  - ARL requires all the data to be in binary form. Therefore, dpOHC is used for that, and any features that aren't binary
    gets dropped before running the analysis. Note: if the outcome variable isn't binary, it'll be dropped from the dataframe and
    an error will occur
  - The function returns a dataframe containing the rules that have been identified

#### Sub-Class Function: Mediation_Analysis()
  - Does Association Rules mining for the items within the passed dataframe. Write all the found association rules that meet the 
    specified conditions and save the produced graphs in the passed parameters to AssociationRules.txt in Apriori Folder in statsPath.
  - After finding the rules that meets the specified requirment, the code finds the odds ratios of the left-hand-side combinations
    of the rules as well as the p-value and confidence interval and write the results down to the same file.
  - Because ARL usually identifies risk factors (lhs that increase chances of observing rhs), there is a parameter -- protective --
    that if set to True, the values of the rhs would be flipped (what is 1 becomes 0 and what is 0 becomes 1), so running the 
    analysis identifies rules that increases chances of finding 0 or protective factors.
  - the ARL dictionary of the Stats object stores either the risk-rules in "ARLRisk" key or protective-rules in "ARLProtect"
  - ARL requires all the data to be in binary form. Therefore, dpOHC is used for that, and any features that aren't binary
    gets dropped before running the analysis. Note: if the outcome variable isn't binary, it'll be dropped from the dataframe and
    an error will occur
  - The function returns a dataframe containing the rules that have been identified
















## Association Rule Learning: (done) (upgraded)
  - Input must be only one hot encoded and outcome variables has to be binary
  - can't include continuous variables

## Mediation Analysis: (undone)
  - Outcome preferably continuous or categorical (preferably continuous)
  - mediator can be either continuous or categorical (doesn't need to be one hot encoded)
  - independent can be either continuous or OHC

## Mendelian Randomization (undone)
  - Don't remember honestly but I think continuous and categorical non-one-hot-encoded variables are fine

## Multivariate Regression (done) (upgraded)
  - categorical variables need to be one hot encoded
  - continuous variables are okay
  - outcome variable has to be continuous

## Logistic Regression (Odds Ratios) (done) (upgraded)
  - categorical variables need to be one hot encoded
  - continuous variables are okay
  - outcome variable has to be binary or categorical

## ANOVA (done) (upgraded)
  - indep variables have to be categorical (preferably not one hot encoded)
  - outcome variable has to be continuous

## Association Analysis (G-Test) (done) (upgraded)
  - indep variables have to be categorical (preferably not one hot encoded)
  - outcome variable has to be continuous
  - Don't include interaction terms for this analysis
