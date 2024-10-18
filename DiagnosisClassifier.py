
from os.path import  isfile, isdir, join
from os import mkdir
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix,roc_auc_score,classification_report,\
    precision_recall_fscore_support, plot_roc_curve
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import sys
import pandas
from pandas import factorize
import numpy as np
import pickle

RANDFESTIMATORS=100

def print_roc(ax,fig,tprs,aucs,fpr,outdir, filename,classifier,ovar,ind_var):
    '''
    This function will save the ROC plots for model with xtest and ytest data to args.output

    :param outdir: output directory to save plots
    :param filename: filename to save plot
    :return: None
    '''

    title_font = {'fontname': 'Arial', 'size': '16', 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'bottom'}

    if not isdir(outdir):
        mkdir(outdir)

        # added to test roc plots
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_fpr = np.mean(fpr, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.6)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax.set_title(label="ROC (%s), \n model: \n %s = %s " %(classifier,ovar," + ".join(str(x) for x in ind_var)),
                 fontdict={'family': 'arial',
                  'color': 'black',
                  'weight': 'normal',
                  'size': 10,
                  }
                 )
    ax.legend(loc="lower right",prop={'size': 6})


    #fig.show()
    fig.set_size_inches(10.5, 8)
    fig.savefig(join(outdir,filename + ".png"),dpi=150)
    #plt.delete()


def write_models (dtree_model,svm_model_linear,randf_model,logreg_model,directory):

    filename = join(directory,'dtree_model.sav')
    pickle.dump(dtree_model, open(filename, 'wb'))

    filename = join(directory,'svm_model.sav')
    pickle.dump(svm_model_linear, open(filename, 'wb'))

    filename = join(directory,'randf_model.sav')
    pickle.dump(randf_model, open(filename, 'wb'))

    filename = join(directory,'logreg_model.sav')
    pickle.dump(logreg_model, open(filename, 'wb'))



def output_results(num_classes,classes,args,pchance_npa,
                   dtree_npa,dtree_precision, dtree_recall,dtree_fbeta,dtree_auc,
                   svm_npa,svm_precision, svm_recall, svm_fbeta,svm_auc,
                   randf_npa,randf_precision, randf_recall, randf_fbeta,randf_auc,
                   logreg_npa,logreg_aucroc_npa,logreg_precision, logreg_recall, logreg_fbeta,directory):


    if not isdir(directory):
        mkdir(directory)
    with open(join(directory,"ClassifierResults.txt"),'w') as f:
        f.write("Datafile: %s\n" %args.data_file)
        f.write("Classifier Results:\n")
        f.write("Cross validation number of stratified splits: %s \n" %args.cv)
        f.write("Predictor Variables Used: %s\n" % " + ".join(str(x) for x in args.ind_var))
        f.write("Outcome Variable Used: %s\n" % args.out_var)
        f.write("Classes in Outcome Variable: %s\n" % classes)
        f.write("mean chance prob=%.2f,\nstd chance prob=%.2f\n\n" %(pchance_npa.mean(),pchance_npa.std()))


        f.write("Decision Tree results:\n")
        f.write("-----------------------\n")
        f.write("accuracy=%.2f +/- %.2f \n" % (dtree_npa.mean(),dtree_npa.std()))
        f.write("precision=%.2f +/- %.2f \n" % (dtree_precision.mean(), dtree_precision.std()))
        f.write("recall=%.2f +/- %.2f \n" % (dtree_recall.mean(), dtree_recall.std()))
        f.write("fbeta=%.2f +/- %.2f \n" % (dtree_fbeta.mean(), dtree_fbeta.std()))
        #f.write("confusion matrix: %s\n\n" %dtree_cm)
        f.write("auc=%.2f +/- %.2f\n\n" % (np.mean(dtree_auc), np.std(dtree_auc)))


        f.write("SVM results:\n")
        f.write("-----------------------\n")
        f.write("accuracy=%.2f +/- %.2f \n" % (svm_npa.mean(),svm_npa.std()))
        f.write("precision=%.2f +/- %.2f \n" % (svm_precision.mean(), svm_precision.std()))
        f.write("recall=%.2f +/- %.2f \n" % (svm_recall.mean(), svm_recall.std()))
        f.write("fbeta=%.2f +/- %.2f \n" % (svm_fbeta.mean(), svm_fbeta.std()))
        #f.write("confusion matrix: %s\n\n" %svm_cm)
        f.write("auc=%.2f +/- %.2f\n\n" % (np.mean(svm_auc), np.std(svm_auc)))

        f.write("Random Forest results:\n")
        f.write("-----------------------\n")
        f.write("accuracy=%.2f +/- %.2f \n" % (randf_npa.mean(),randf_npa.std()))
        f.write("precision=%.2f +/- %.2f \n" % (randf_precision.mean(), randf_precision.std()))
        f.write("recall=%.2f +/- %.2f \n" % (randf_recall.mean(), randf_recall.std()))
        f.write("fbeta=%.2f +/- %.2f \n" % (randf_fbeta.mean(), randf_fbeta.std()))
        #f.write("confusion matrix: %s\n" %randf_cm)
        f.write("auc=%.2f +/- %.2f\n\n" % (np.mean(randf_auc), np.std(randf_auc)))


        f.write("Logistic Regression results:\n")
        f.write("-----------------------\n")
        f.write("accuracy=%.2f +/- %.2f\n" % (logreg_npa.mean(),logreg_npa.std()))
        f.write("precision=%.2f +/- %.2f \n" % (logreg_precision.mean(), logreg_precision.std()))
        f.write("recall=%.2f +/- %.2f \n" % (logreg_recall.mean(), logreg_recall.std()))
        f.write("fbeta=%.2f +/- %.2f \n" % (logreg_fbeta.mean(), logreg_fbeta.std()))
        f.write("auc=%.2f +/- %.2f\n" % (np.mean(logreg_aucroc_npa),np.std(logreg_aucroc_npa)))


def classtransform(args):
    class_map = {}


    #if args.class_trans is defined then look for class+class style transformations otherwise ignore
    if args.class_trans:
        for categories in args.class_trans:
            #check if there's a "+" in the string else remove rows in X and y that aren't represented in the args.class_trans
            plus_loc = categories.find("+")
            if plus_loc != -1:
                plus_class1 = categories[:plus_loc-1]
                plus_class2 = categories[plus_loc+1:]

    return class_map


def main(argv):

    parser = ArgumentParser(description='This software will load in the data file and perform cross-validated'
                                        'classification using the -ivars and predicting the -ovar')
    parser.add_argument('-d', dest='data_file', required=True, help="CSV data file to use for classification")
    parser.add_argument('-ivars', dest='ind_var', required=True, nargs='+', help="Space separated list of variables to"
                                                                                "use for classification")
    parser.add_argument('-ovar', dest='out_var', required=True, help="Outcome variable to predict")
    parser.add_argument('-nuisance', dest='nuisance', required=False, nargs='+', help="A space separated list of nusiance"
                            "variables.  If nuisance variables are specified then " \
                            "a simple linear regression will be done on the -ivars removing the linear effects of" \
                            "the nuisance variables and the residuals will be used for classification.")
    #parser.add_argument('-class_transform', dest='class_trans', nargs="+", required=False, help="Allows groupings of -ovar classes"
    #                        "For example, if -ovar classes are ND, MCI, and DEM the -class_transform string can be"
    #                        "used to transform to 2 classes MCI+DEM ND and the examples with diagnosis MCI will be"
    #                        "added to the class of DEM and they'll be treated as 1 class for a 2-class classification"
    #                        "problem instead of the original 3.  Another alternative is for one to limit the classifiers"
    #                        "to only 2 classes and thus -class_transform DEM ND would limit the classifier to only"
    #                        "doing a 2 class classification problem.")
    parser.add_argument('-cv', dest='cv', required=True, help="Number of cross validated splits")
    parser.add_argument('-plot', action='store_true', required=False, help="If flag set then ROC curve plots will be "
                        "added to the -output directory")
    parser.add_argument('-output', dest='output_directory', required=True, help="Path to store classification results")
    args = parser.parse_args()

    if not isfile(args.data_file):
        print("Data file not found! %s" %args.data_file)
        exit()
    # load data file into data frame
    data = pandas.read_csv(args.data_file)

    num_classes = data[args.out_var].nunique()

    #if num_classes > 2 and (not args.class_trans):
    if num_classes > 2:
        print("Only 2 class classification problems supported at this time.")
        exit(1)

    # if user added nuisance variables then do Linear Regression to remove the linear effects of the nuisance
    # variables from the args.ind_vars
    if args.nuisance:
        # check if any of the nuisance variables are string categorical.  If so factorize them
        for var in args.nuisance:
             # if var is a string then categorical so factorize for linear regression model
            if data[var].dtype == object:
                print("\n\nfactorizing nuisance categorical variable: %s" %var)
                #data[var], temp = factorize(data[var])
                #t, temp = factorize(data[var])
                le = LabelEncoder()
                data[var] = le.fit_transform(data[var])

        # for each args.ind_var, remove the linear effects of nuisance vars ind_var ~ nuisance + e
        print("running linear regression to remove nuisance variables: %s" %args.nuisance)
        for var in args.ind_var:
            # if var is a string then categorical so factorize for linear regression model
            if data[var].dtype == object:
                print("factorizing categorical variable: %s" %var)
                #data[var], temp = factorize(data[var])
                t, temp = factorize(data[var])
                le = LabelEncoder()
                data[var] = le.fit_transform(data[var])
            lm=LinearRegression()
            # fit linear reg data[var] ~ data[args.nuisance] + e
            lm.fit(data[args.nuisance],data[var])
            # replace data[var] with the residuals from the above model
            data[var] = lm.predict(data[args.nuisance]) - data[var]


    # subset data based on user's selection
    X = data[args.ind_var].to_numpy()
    y = data[args.out_var].to_numpy()


    classes = data[args.out_var].unique()
    pchance=[]
    dtree_accuracy=[]
    dtree_precision=[]
    dtree_recall=[]
    dtree_fbeta=[]
    dtree_tprs=[]
    dtree_fpr=[]
    dtree_auc=[]
    svm_accuracy=[]
    svm_precision=[]
    svm_recall=[]
    svm_fbeta=[]
    svm_tprs=[]
    svm_fpr=[]
    svm_auc=[]
    randf_accuracy=[]
    randf_precision=[]
    randf_recall=[]
    randf_fbeta=[]
    randf_models=[]
    randf_tprs=[]
    randf_fpr=[]
    randf_auc=[]
    logreg_accuracy=[]
    logreg_auc=[]
    logreg_aucroc=[]
    logreg_precision=[]
    logreg_recall=[]
    logreg_fbeta=[]
    logreg_tprs=[]
    logreg_fpr=[]

    mean_fpr = np.linspace(0, 1, 100)

    # set up plotting axes for each classifier
    classifiers=['DecisionTree','SVM','RandomForest','LogisticRegression']
    randf_fig,randf_ax = plt.subplots()
    logreg_fig, logreg_ax = plt.subplots()
    svm_fig, svm_ax = plt.subplots()
    dtree_fig, dtree_ax = plt.subplots()





    print("running classifiers with %s stratified cross-validated splits.." %args.cv)

    # set up stratified sampling
    skf = StratifiedKFold(n_splits=int(args.cv))

    fold=1
    # run cross validation training and testing
    for train_indx, test_indx in skf.split(X, y):
        X_train, X_test = X[train_indx], X[test_indx]
        y_train, y_test = y[train_indx], y[test_indx]

    #for i in range(int(args.cv)):
        # divide data into test/training data
        #if args.ptest:
            #X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=float(args.ptest))

        #else:
            #X_train, X_test, y_train, y_test = train_test_split(X, y)

        # store chance probability which is the frequency of the majority class
        num_examples={}
        for j in classes:
            #temp = y_test.apply(lambda x: True if x == j else False)
            temp = np.where(y_test == j)
            #num_examples[j] = len(temp[temp == True].index)
            num_examples[j] = len(temp[0])

        pchance.append(float(max(num_examples.values()))/float(sum(num_examples.values())))

        # encode outcome variable as factor using reversible trasform
        le = LabelEncoder()
        y_train_factors = le.fit_transform(y_train)
        y_test_factors = le.fit_transform(y_test)



        # training a DescisionTreeClassifier
        dtree_model = DecisionTreeClassifier().fit(X_train, y_train_factors)
        dtree_predictions = dtree_model.predict(X_test)
        # creating a confusion matrix
        #dtree_cm = confusion_matrix(y_test, le.inverse_transform(dtree_predictions),labels=np.unique(y))
        dtree_accuracy.append(float(dtree_model.score(X_test,y_test_factors)))

        # compute precision, recall, and fbeta using weighted averaging to insulate from possible
        # class imbalance
        prec, recall, fbeta, support = precision_recall_fscore_support(y_test,
                            le.inverse_transform(dtree_predictions).tolist(),labels=classes.tolist(),
                            average="weighted")
        dtree_precision.append(prec)
        dtree_recall.append(recall)
        dtree_fbeta.append(fbeta)


        # roc plotting
        viz = plot_roc_curve(dtree_model, X_test, y_test_factors,
                                 name='ROC fold {}'.format(fold),
                                 alpha=0.3, lw=1, ax=dtree_ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        dtree_tprs.append(interp_tpr)
        dtree_fpr.append(mean_fpr)
        dtree_auc.append(auc(mean_fpr, interp_tpr))

        # training a linear SVM classifier
        svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train_factors)
        svm_predictions = svm_model_linear.predict(X_test)

        # model accuracy for X_test
        svm_accuracy.append(float(svm_model_linear.score(X_test, y_test_factors)))

        # creating a confusion matrix
        #svm_cm = confusion_matrix(y_test, le.inverse_transform(svm_predictions),labels=np.unique(y))
        # compute precision, recall, and fbeta using weighted averaging to insulate from possible
        # class imbalance

        prec, recall, fbeta, support = precision_recall_fscore_support(y_test,
                    le.inverse_transform(svm_predictions).tolist(),labels=classes.tolist(),
                    average="weighted")
        svm_precision.append(prec)
        svm_recall.append(recall)
        svm_fbeta.append(fbeta)

        # roc plotting
        viz = plot_roc_curve(svm_model_linear, X_test, y_test_factors,
                             name='ROC fold {}'.format(fold),
                             alpha=0.3, lw=1, ax=svm_ax)

        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        svm_tprs.append(interp_tpr)
        svm_fpr.append(mean_fpr)
        svm_auc.append(auc(mean_fpr, interp_tpr))

        # Random forest classifier

        randf_model = RandomForestClassifier(n_estimators=RANDFESTIMATORS).fit(X_train, y_train_factors)
        randf_predictions = randf_model.predict(X_test)
        randf_accuracy.append(float(randf_model.score(X_test, y_test_factors)))
        #randf_cm = confusion_matrix(y_test, le.inverse_transform(randf_predictions),labels=np.unique(y))

        # compute precision, recall, and fbeta using weighted averaging to insulate from possible
        # class imbalance
        prec, recall, fbeta, support = precision_recall_fscore_support(y_test,
                        le.inverse_transform(randf_predictions).tolist(),labels=classes.tolist(),
                        average="weighted")
        randf_precision.append(prec)
        randf_recall.append(recall)
        randf_fbeta.append(fbeta)

        # roc plotting
        viz = plot_roc_curve(randf_model, X_test, y_test_factors,
                             name='ROC fold {}'.format(fold),
                             alpha=0.3, lw=1, ax=randf_ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        randf_tprs.append(interp_tpr)
        randf_fpr.append(mean_fpr)
        randf_auc.append(auc(mean_fpr, interp_tpr))

        # logistic regression classification only supported for 2 class classification problems


        logreg_model = LogisticRegression(random_state=0, solver='liblinear').fit(X_train, y_train_factors)
        logreg_predictions = logreg_model.predict(X_test)
        # get probabilities for the class with the highest label which in this binary decision is '1'
        logreg_predictions_prob = logreg_model.predict_proba(X_test)
        #logreg_cm = confusion_matrix(y_test, le.inverse_transform(logreg_predictions), labels=np.unique(y))

        # roc plotting
        viz = plot_roc_curve(logreg_model, X_test, y_test_factors,
                                 name='ROC fold {}'.format(fold),
                                 alpha=0.3, lw=1, ax=logreg_ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        logreg_tprs.append(interp_tpr)
        logreg_fpr.append(mean_fpr)
        logreg_auc.append(auc(mean_fpr, interp_tpr))


        # accuracy
        logreg_accuracy.append(float(logreg_model.score(X_test,y_test_factors)))
        # precision / recall / fbeta
        prec,recall,fbeta,support = precision_recall_fscore_support(y_test,
                                        le.inverse_transform(logreg_predictions).tolist(),labels=classes.tolist(),
                                        average="weighted")
        logreg_precision.append(prec)
        logreg_recall.append(recall)
        logreg_fbeta.append(fbeta)

        fold += 1



    # convert result lists to numpy arrays

    dtree_npa = np.asarray(dtree_accuracy, dtype=np.float32)
    dtree_precision_npa = np.asarray(dtree_precision, dtype=np.float32)
    dtree_recall_npa = np.asarray(dtree_recall, dtype=np.float32)
    dtree_fbeta_npa = np.asarray(dtree_fbeta, dtype=np.float32)
    svm_npa = np.asarray(svm_accuracy, dtype=np.float32)
    svm_precision_npa = np.asarray(svm_precision, dtype=np.float32)
    svm_recall_npa = np.asarray(svm_recall, dtype=np.float32)
    svm_fbeta_npa = np.asarray(svm_fbeta, dtype=np.float32)
    randf_npa = np.asarray(randf_accuracy, dtype=np.float32)
    randf_precision_npa = np.asarray(randf_precision, dtype=np.float32)
    randf_recall_npa = np.asarray(randf_recall, dtype=np.float32)
    randf_fbeta_npa = np.asarray(randf_fbeta, dtype=np.float32)
    logreg_npa = np.asarray(logreg_accuracy,dtype=np.float32)
    logreg_precision_npa = np.asarray(logreg_precision,dtype=np.float32)
    logreg_recall_npa = np.asarray(logreg_recall, dtype=np.float32)
    logreg_fbeta_npa = np.asarray(logreg_fbeta, dtype=np.float32)
    pchance_npa = np.asarray(pchance,dtype=np.float32)




    # plot dtree forest regression
    print_roc(ax=dtree_ax, fig=dtree_fig, aucs=dtree_auc, tprs=dtree_tprs, fpr=dtree_fpr,
                  outdir=args.output_directory,
                  filename="dtree_roc", classifier=classifiers[0], ovar=args.out_var, ind_var=args.ind_var)


    # plot svm forest regression
    print_roc(ax=svm_ax, fig=svm_fig, aucs=svm_auc, tprs=svm_tprs, fpr=svm_fpr,
                      outdir=args.output_directory,
                      filename="svm_roc", classifier=classifiers[1], ovar=args.out_var, ind_var=args.ind_var)

    # plot random forest regression
    print_roc(ax=randf_ax, fig=randf_fig,aucs=randf_auc, tprs=randf_tprs, fpr=randf_fpr, outdir=args.output_directory,
                      filename="randf_roc", classifier=classifiers[2], ovar=args.out_var, ind_var=args.ind_var)

    # plot logistic regression
    print_roc(ax=logreg_ax,fig=logreg_fig, aucs=logreg_auc, tprs=logreg_tprs, fpr=logreg_fpr, outdir=args.output_directory,
                      filename="logreg_roc", classifier=classifiers[3], ovar=args.out_var, ind_var=args.ind_var)

    output_results(num_classes=num_classes,classes=classes,args=args,pchance_npa=pchance_npa,
                       dtree_npa=dtree_npa,dtree_precision=dtree_precision_npa,dtree_fbeta = dtree_fbeta_npa,
                       dtree_recall = dtree_recall_npa, dtree_auc=dtree_auc,
                       svm_npa=svm_npa,svm_precision=svm_precision_npa,svm_recall=svm_recall_npa,
                       svm_fbeta=svm_fbeta_npa,svm_auc=svm_auc,
                       randf_npa=randf_npa,randf_precision=randf_precision_npa,
                       randf_recall = randf_recall_npa,randf_fbeta=randf_fbeta_npa, randf_auc=randf_auc,
                       logreg_npa=logreg_npa,logreg_aucroc_npa=logreg_auc, logreg_precision= logreg_precision_npa,
                       logreg_recall=logreg_recall_npa,logreg_fbeta=logreg_fbeta_npa,
                       directory=args.output_directory)

    write_models(dtree_model=dtree_model,svm_model_linear=svm_model_linear,randf_model=randf_model,
                     logreg_model=logreg_model,directory=args.output_directory)



if __name__ == "__main__":
    main(sys.argv[1:])

