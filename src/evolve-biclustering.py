# This script has been designed to perform multi-objective learning of core sets 
# by Alberto Tonda and Pietro Barbiero, 2018 <alberto.tonda@gmail.com> <pietro.barbiero@studenti.polito.it>

#basic libraries
import argparse
import copy
import datetime
import inspyred
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import sys
import logging
import time
import tensorflow as tf

# sklearn library
from sklearn import datasets
from sklearn.datasets.samples_generator import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

# pandas
from pandas import read_csv

import seaborn as sns
from matplotlib.colors import ListedColormap

import warnings
warnings.filterwarnings("ignore")

def main(selectedDataset = "digits", pop_size = 100, max_generations = 100):
	
	# a few hard-coded values
	figsize = [5, 4]
	seed = 42
	max_points_in_core_set = 99
	min_points_in_core_set = 1 # later redefined as 1 per class
#	pop_size = 300
	offspring_size = 2 * pop_size
#	max_generations = 300
	maximize = False
#	selectedDataset = "digits"
	selectedClassifiers = ["SVC"]
	n_splits = 10

	# a list of classifiers
	allClassifiers = [
			[RandomForestClassifier, "RandomForestClassifier", 1],
			[BaggingClassifier, "BaggingClassifier", 1],
			[SVC, "SVC", 1],
			[RidgeClassifier, "RidgeClassifier", 1],
#			[AdaBoostClassifier, "AdaBoostClassifier", 1],
#			[ExtraTreesClassifier, "ExtraTreesClassifier", 1],
#			[GradientBoostingClassifier, "GradientBoostingClassifier", 1],
#			[SGDClassifier, "SGDClassifier", 1],
#			[PassiveAggressiveClassifier, "PassiveAggressiveClassifier", 1],
#			[LogisticRegression, "LogisticRegression", 1],
			]
	
	selectedClassifiers = [classifier[1] for classifier in allClassifiers]
	
	folder_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + "-evocore2C-" + selectedDataset + "-" + str(pop_size)
	if not os.path.exists(folder_name) : 
		os.makedirs(folder_name)
	else :
		sys.stderr.write("Error: folder \"" + folder_name + "\" already exists. Aborting...\n")
		sys.exit(0)
	# open the logging file
	logfilename = os.path.join(folder_name, 'logfile.log')
	logger = setup_logger('logfile_' + folder_name, logfilename)
	logger.info("All results will be saved in folder \"%s\"" % folder_name)
	
	# load different datasets, prepare them for use
	logger.info("Preparing data...")
	# synthetic databases
#	centers = [[1, 1], [-1, -1], [1, -1]]
#	blobs_X, blobs_y = make_blobs(n_samples=400, centers=centers, n_features=2, cluster_std=0.6, random_state=seed)
#	circles_X, circles_y = make_circles(n_samples=400, noise=0.15, factor=0.4, random_state=seed)
#	moons_X, moons_y = make_moons(n_samples=400, noise=0.2, random_state=seed)
#	iris = datasets.load_iris()
#	digits = datasets.load_digits()
	wine = datasets.load_wine()
#	breast = datasets.load_breast_cancer()
#	pairs = datasets.fetch_lfw_pairs()
#	olivetti = datasets.fetch_olivetti_faces()
#	forest_X, forest_y = loadForestCoverageType() # local function
#	mnist_X, mnist_y = loadMNIST() # local function
	
#	plants = datasets.fetch_openml(name='one-hundred-plants-margin', cache=False)
#	isolet = datasets.fetch_openml(name='isolet', cache=False)
#	ctg = datasets.fetch_openml(name='cardiotocography', cache=False)
#	ozone = datasets.fetch_openml(name='ozone-level-8hr', cache=False)
#	ilpd = datasets.fetch_openml(name='ilpd', cache=False)
#	biodeg = datasets.fetch_openml(name='qsar-biodeg', cache=False)
#	hill = datasets.fetch_openml(name='hill-valley', cache=False)

	dataList = [
#			[blobs_X, blobs_y, 0, "blobs"],
#			[circles_X, circles_y, 0, "circles"],
#			[moons_X, moons_y, 0, "moons"],
#			[iris.data, iris.target, 0, "iris4"],
#			[iris.data[:, 2:4], iris.target, 0, "iris2"],
#			[digits.data, digits.target, 0, "digits"],
			[wine.data, wine.target, 0, "wine"],
#			[breast.data, breast.target, 0, "breast"],
#			[pairs.data, pairs.target, 0, "pairs"],
#			[olivetti.data, olivetti.target, 0, "people"],
#			[forest_X, forest_y, 0, "covtype"],
#			[mnist_X, mnist_y, 0, "mnist"],
#			[plants.data, plants.target, 0, "plants"],
#			[isolet.data, isolet.target, 0, "isolet"],
#			[ctg.data, ctg.target, 0, "ctg"],
#			[ozone.data, ozone.target, 0, "ozone"],
#			[ilpd.data, ilpd.target, 0, "ilpd"],
#			[biodeg.data, biodeg.target, 0, "biodeg"],
#			[hill.data, hill.target, 0, "hill-valley"],
			]

	# argparse; all arguments are optional
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--classifiers", "-c", nargs='+', help="Classifier(s) to be tested. Default: %s. Accepted values: %s" % (selectedClassifiers[0], [x[1] for x in allClassifiers]))
	parser.add_argument("--dataset", "-d", help="Dataset to be tested. Default: %s. Accepted values: %s" % (selectedDataset,[x[3] for x in dataList]))
	
	parser.add_argument("--pop_size", "-p", type=int, help="EA population size. Default: %d" % pop_size)
	parser.add_argument("--offspring_size", "-o", type=int, help="Ea offspring size. Default: %d" % offspring_size)
	parser.add_argument("--max_generations", "-mg", type=int, help="Maximum number of generations. Default: %d" % max_generations)
	
	parser.add_argument("--min_points", "-mip", type=int, help="Minimum number of points in the core set. Default: %d" % min_points_in_core_set)
	parser.add_argument("--max_points", "-mxp", type=int, help="Maximum number of points in the core set. Default: %d" % max_points_in_core_set)
	
	# finally, parse the arguments
	args = parser.parse_args()
	
	# a few checks on the (optional) inputs
	if args.dataset : 
		selectedDataset = args.dataset
		if selectedDataset not in [x[3] for x in dataList] :
			print("Error: dataset \"%s\" is not an accepted value. Accepted values: %s" % (selectedDataset, [x[3] for x in dataList]))
			sys.exit(0)
	
	if args.classifiers != None and len(args.classifiers) > 0 :
		selectedClassifiers = args.classifiers
		for c in selectedClassifiers :
			if c not in [x[1] for x in allClassifiers] :
				print("Error: classifier \"%s\" is not an accepted value. Accepted values: %s" % (c, [x[1] for x in allClassifiers]))
				sys.exit(0)
	
	if args.min_points : min_points_in_core_set = args.min_points
	if args.max_points : max_points_in_core_set = args.max_points
	if args.max_generations : max_generations = args.max_generations
	if args.pop_size : pop_size = args.pop_size
	if args.offspring_size : offspring_size = args.offspring_size
	
	# TODO: check that min_points < max_points and max_generations > 0
	
	
	# print out the current settings
	logger.info("Settings of the experiment...")
	logger.info("Fixed random seed:", seed)
	logger.info("Selected dataset: %s; Selected classifier(s): %s" % (selectedDataset, selectedClassifiers))
#	logger.info("Min points in candidate core set: %d; Max points in candidate core set: %d" % (min_points_in_core_set, max_points_in_core_set))
	logger.info("Population size in EA: %d; Offspring size: %d; Max generations: %d" % (pop_size, offspring_size, max_generations))

	# create the list of classifiers
	classifierList = [ x for x in allClassifiers if x[1] in selectedClassifiers ]	

	# pick the dataset 
	db_index = -1
	for i in range(0, len(dataList)) :
		if dataList[i][3] == selectedDataset :
			db_index = i

	dbname = dataList[db_index][3]
	
	X, y = dataList[db_index][0], dataList[db_index][1]
	number_classes = np.unique(y).shape[0]
	
	logger.info("Creating train/test split...")
	from sklearn.model_selection import StratifiedKFold
	skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
	listOfSplits = [split for split in skf.split(X, y)]
	trainval_index, test_index = listOfSplits[0]
	X_trainval, y_trainval = X[trainval_index], y[trainval_index]
	X_test, y_test = X[test_index], y[test_index]
	skf = StratifiedKFold(n_splits=3, shuffle=False, random_state=seed)
	listOfSplits = [split for split in skf.split(X_trainval, y_trainval)]
	train_index, val_index = listOfSplits[0]
	X_train, y_train = X_trainval[train_index], y_trainval[train_index]
	X_val, y_val = X_trainval[val_index], y_trainval[val_index]
	logger.info("Training set: %d lines (%.2f%%); test set: %d lines (%.2f%%)" % (X_trainval.shape[0], (100.0 * float(X_trainval.shape[0]/X.shape[0])), X_test.shape[0], (100.0 * float(X_test.shape[0]/X.shape[0]))))
	
	# rescale data
	scaler = StandardScaler()
	sc = scaler.fit(X_train)
	X = sc.transform(X)
	X_trainval = sc.transform(X_trainval)
	X_train = sc.transform(X_train)
	X_val = sc.transform(X_val)
	X_test = sc.transform(X_test)
	
	for classifier in classifierList:
		
		classifier_name = classifier[1]

		# start creating folder name
		experiment_name = os.path.join(folder_name, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + "-core-set-evolution-" + dbname + "-" + classifier_name
		if not os.path.exists(experiment_name) : os.makedirs(experiment_name)
	   
		logger.info("Classifier used: " + classifier_name)
		
		max_points_in_core_set = X_train.shape[0]
		start = time.time()
		min_points_in_core_set = number_classes
		solutions, trainAccuracy, testAccuracy = evolveCoreSets(X, y, X_train, y_train, X_test, y_test, classifier, pop_size, offspring_size, max_generations, min_points_in_core_set, max_points_in_core_set, number_classes, maximize, seed=seed, experiment_name=experiment_name) 
		end = time.time()
		exec_time = end - start
		
		# only candidates with all classes are considered
		final_archive = []
		for sol in solutions :
			c = sol.candidate
			individual = np.array(c, dtype=bool)
			indPoints = individual[ :X_train.shape[0] ]
			y_core = y_train[indPoints]
			if len(set(y_core)) == number_classes :
				final_archive.append(sol)
		
#		logger.info("Now saving final Pareto front in a figure...")
		pareto_front_x = [ f.fitness[0] for f in final_archive ]
		pareto_front_y = [ f.fitness[1] for f in final_archive ]
		pareto_front_z = [ f.fitness[2] for f in final_archive ]

#		figure = plt.figure(figsize=figsize)
#		ax = figure.add_subplot(111)
#		ax.plot(pareto_front_x, pareto_front_y, "bo-", label="Solutions in final archive")
#		ax.set_title("Optimal solutions")
#		ax.set_xlabel("Core set size")
#		ax.set_ylabel("Error")
#		ax.set_xlim([1, X_train.shape[0]])
#		ax.set_ylim([0, 0.4])
#		plt.tight_layout()
#		plt.savefig( os.path.join(experiment_name, "%s_EvoCore_%s_pareto.png" %(dbname, classifier_name)) )
#		plt.savefig( os.path.join(experiment_name, "%s_EvoCore_%s_pareto.pdf" %(dbname, classifier_name)) )
#		plt.close(figure)
		
		figure = plt.figure(figsize=figsize)
		ax = figure.add_subplot(111)
		ax.plot(pareto_front_x, pareto_front_y, "bo", label="Solutions in final archive")
		ax.set_title("Optimal solutions")
		ax.set_xlabel("Core set size")
		ax.set_ylabel("Core feature size")
		plt.tight_layout()
		plt.savefig( os.path.join(experiment_name, "%s_EvoCore_%s_pareto_zoom_xy.png" %(dbname, classifier_name)) )
		plt.savefig( os.path.join(experiment_name, "%s_EvoCore_%s_pareto_zoom_xy.pdf" %(dbname, classifier_name)) )
		plt.close(figure)
		
		figure = plt.figure(figsize=figsize)
		ax = figure.add_subplot(111)
		ax.plot(pareto_front_x, pareto_front_z, "bo", label="Solutions in final archive")
		ax.set_title("Optimal solutions")
		ax.set_xlabel("Core set size")
		ax.set_ylabel("Error")
		plt.tight_layout()
		plt.savefig( os.path.join(experiment_name, "%s_EvoCore_%s_pareto_zoom_xz.png" %(dbname, classifier_name)) )
		plt.savefig( os.path.join(experiment_name, "%s_EvoCore_%s_pareto_zoom_xz.pdf" %(dbname, classifier_name)) )
		plt.close(figure)
		
		figure = plt.figure(figsize=figsize)
		ax = figure.add_subplot(111)
		ax.plot(pareto_front_y, pareto_front_z, "bo", label="Solutions in final archive")
		ax.set_title("Optimal solutions")
		ax.set_xlabel("Core feature size")
		ax.set_ylabel("Error")
		plt.tight_layout()
		plt.savefig( os.path.join(experiment_name, "%s_EvoCore_%s_pareto_zoom_yz.png" %(dbname, classifier_name)) )
		plt.savefig( os.path.join(experiment_name, "%s_EvoCore_%s_pareto_zoom_yz.pdf" %(dbname, classifier_name)) )
		plt.close(figure)
		
		# initial performance
		X_err, testAccuracy, model, fail_points, y_pred = evaluate_core(X_trainval, y_trainval, X_test, y_test, classifier[0], cname=classifier_name, SEED=seed)
		X_err, trainAccuracy, model, fail_points, y_pred = evaluate_core(X_trainval, y_trainval, X_trainval, y_trainval, classifier[0], cname=classifier_name, SEED=seed)
#		logger.info("Compute performances!")
		logger.info("Problem dimensions: #samples: %d - #features: %d - #classes: %d" % (X.shape[0], X.shape[1], number_classes))
		logger.info("Initial performance with #samples: %d - #features: %d --> train=%.4f, test=%.4f" % (X_trainval.shape[0], X.shape[1], trainAccuracy, testAccuracy))
		logger.info("Elapsed time using EvoCore2C (seconds): %.4f" %(exec_time))
		
		# best solution
		accuracy = []
		for sol in final_archive :
			c = sol.candidate
			individual = np.array(c, dtype=bool)
			indPoints = individual[ :X_train.shape[0] ]
			indFeatures = individual[ X_train.shape[0]: ]
			
			X_core = X_train[indPoints]
			X_core = X_core[:, indFeatures]
			y_core = y_train[indPoints]
			
			X_trainval_t = X_trainval[:, indFeatures]
			X_train_t = X_train[:, indFeatures]
			X_val_t = X_val[:, indFeatures]
			
			X_err, accuracy_val, model, fail_points, y_pred = evaluate_core(X_core, y_core, X_val_t, y_val, classifier[0], cname=classifier_name, SEED=seed)
			X_err, accuracy_train, model, fail_points, y_pred = evaluate_core(X_core, y_core, X_train_t, y_train, classifier[0], cname=classifier_name, SEED=seed)
			accuracy.append( np.mean([accuracy_val, accuracy_train]) )
			
		
		best_ids = np.array(np.argsort(accuracy)).astype('int')[::-1]
		count = 0
		for i in best_ids:	
			
			if count > 2:
				break
			
			c = final_archive[i].candidate
			individual = np.array(c, dtype=bool)
			indPoints = individual[ :X_train.shape[0] ]
			indFeatures = individual[ X_train.shape[0]: ]
			
			X_core = X_train[indPoints]
			X_core = X_core[:, indFeatures]
			y_core = y_train[indPoints]
			
			X_t = X[:, indFeatures]
			X_trainval_t = X_trainval[:, indFeatures]
			X_train_t = X_train[:, indFeatures]
			X_val_t = X_val[:, indFeatures]
			X_test_t = X_test[:, indFeatures]
			
			X_err, accuracy_train, model, fail_points, y_pred = evaluate_core(X_core, y_core, X_train_t, y_train, classifier[0], cname=classifier_name, SEED=seed)
			X_err, accuracy_val, model, fail_points, y_pred = evaluate_core(X_core, y_core, X_val_t, y_val, classifier[0], cname=classifier_name, SEED=seed)
			X_err, accuracy, model, fail_points, y_pred = evaluate_core(X_core, y_core, X_test_t, y_test, classifier[0], cname=classifier_name, SEED=seed)
			logger.info("Solution %d: #samples: %d - #features: %d --> train: %.4f, val: %.4f, test: %.4f" %(count, X_core.shape[0], X_core.shape[1], accuracy_train, accuracy_val, accuracy))
		
			if (dbname == "mnist" or dbname == "digits") and count == 0:
				
				if dbname == "mnist":
					H, W = 28, 28
				if dbname == "digits":
					H, W = 8, 8
				
#				logger.info("Now saving figures...")
				
				core_features = np.zeros((X_train.shape[1], 1))
				core_features[indFeatures] = 1
				core_features = np.reshape(core_features, (H, W))
				
				flatui = ["#ffffff", "#e74c3c"]
				cmap = ListedColormap(sns.color_palette(flatui).as_hex())
#				sns.palplot(sns.color_palette(flatui))
				
				X_t = np.zeros((X_core.shape[0], H*W))
				k = 0
				for i in range(0, len(indFeatures)):
					if indFeatures[i] == True:
						X_t[:, i] = X_core[:, k]
						k = k + 1
				
#				# save archetypes
#				for index in range(0, len(y_core)):
#					image = np.reshape(X_t[index, :], (H, W))
#					plt.figure()
#					plt.axis('off')
#					plt.imshow(image, cmap=plt.cm.gray_r)
#					plt.imshow(core_features, cmap=cmap, alpha=0.3)
#					plt.title('Label: %d' %(y_core[index]))
#					plt.tight_layout()
#					plt.savefig( os.path.join(experiment_name, "digit_%d_idx_%d.pdf" %(y_core[index], index)) )
#					plt.savefig( os.path.join(experiment_name, "digit_%d_idx_%d.png" %(y_core[index], index)) )
#					plt.close()

				# save test errors
				e = 1
				for index in range(0, len(y_test)):
					if fail_points[index] == True:
						image = np.reshape(X_test[index, :], (H, W))
						plt.figure()
						plt.axis('off')
						plt.imshow(image, cmap=plt.cm.gray_r)
						plt.imshow(core_features, cmap=cmap, alpha=0.3)
						plt.title('Label: %d - Prediction: %d' %(y_test[index], y_pred[index]))
						plt.tight_layout()
						plt.savefig( os.path.join(experiment_name, "err_lab_%d_pred_%d_idx_%d.pdf" %(y_test[index], y_pred[index], e)) )
						plt.savefig( os.path.join(experiment_name, "err_lab_%d_pred_%d_idx_%d.png" %(y_test[index], y_pred[index], e)) )
						plt.close()
						e = e + 1
			
			# plot decision boundaries if we have only 2 dimensions!
			if X_core.shape[1] == 2:
				
				cmap = ListedColormap(sns.color_palette("bright", 3).as_hex())
				xx, yy = make_meshgrid(X_t[:, 0], X_t[:, 1])
				figure = plt.figure(figsize=figsize)
				_, Z_0 = plot_contours(model, xx, yy, colors='k', alpha=0.2)
	#			plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, marker='s', alpha=0.4, label="train")
				plt.scatter(X_test_t[:, 0], X_test_t[:, 1], c=y_test, cmap=cmap, marker='+', alpha=0.3, label="test")
				plt.scatter(X_core[:, 0], X_core[:, 1], c=y_core, cmap=cmap, marker='D', facecolors='none', edgecolors='none', alpha=1, label="core set")
				plt.scatter(X_err[:, 0], X_err[:, 1], marker='x', facecolors='k', edgecolors='k', alpha=1, label="errors")
				plt.legend()
				plt.title("%s - acc. %.4f" %(classifier_name, accuracy))
				plt.tight_layout()
				plt.savefig( os.path.join(experiment_name, "%s_EvoCore2C_%s_%d.png" %(dbname, classifier_name, count)) )
				plt.savefig( os.path.join(experiment_name, "%s_EvoCore2C_%s_%d.pdf" %(dbname, classifier_name, count)) )
				plt.close(figure)
				
				if count == 0:
					# using all samples in the training set
					X_err, accuracy, model, fail_points, y_pred = evaluate_core(X_trainval, y_trainval, X_test, y_test, classifier[0], cname=classifier_name, SEED=seed)
					X_err_t = X_err[:, indFeatures]
					X_err_train, trainAccuracy, model_train, fail_points_train, y_pred_train = evaluate_core(X_trainval, y_trainval, X_trainval, y_trainval, classifier[0], cname=classifier_name, SEED=seed)
				
					figure = plt.figure(figsize=figsize)
#					_, Z_0 = plot_contours(model, xx, yy, colors='k', alpha=0.2)
					plt.scatter(X_trainval_t[:, 0], X_trainval_t[:, 1], c=y_trainval, cmap=cmap, marker='s', alpha=0.4, label="train")
					plt.scatter(X_test_t[:, 0], X_test_t[:, 1], c=y_test, cmap=cmap, marker='+', alpha=0.4, label="test")
					plt.scatter(X_err_t[:, 0], X_err_t[:, 1], marker='x', facecolors='k', edgecolors='k', alpha=1, label="errors")
					plt.legend()
					plt.title("%s - acc. %.4f" %(classifier_name, accuracy))
					plt.tight_layout()
					plt.savefig( os.path.join(experiment_name, "%s_EvoCore2C_%s_alltrain.png" %(dbname, classifier_name)) )
					plt.savefig( os.path.join(experiment_name, "%s_EvoCore2C_%s_alltrain.pdf" %(dbname, classifier_name)) )
					plt.close(figure)
				
			count = count + 1
	
	logger.handlers.pop()

	return

# function that does most of the work
def evolveCoreSets(X, y, X_train, y_train, X_test, y_test, classifier, pop_size, offspring_size, max_generations, min_points_in_core_set, max_points_in_core_set, number_classes, maximize=True, seed=None, experiment_name=None, split="") :

	classifier_class = classifier[0]
	classifier_name = classifier[1]
	classifier_type = classifier[2]
	
	# a few checks on the arguments
	if seed == None : seed = int( time.time() )
	if experiment_name == None : 
		experiment_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + "-ea-impair-" + classifier_name  
	elif split != "" : 
		experiment_name = experiment_name + "/" + classifier_name + "-split-" + split
	
	# create filename that will be later used to store whole population
	all_population_file = os.path.join(experiment_name, "all_individuals.csv") 

	# initialize classifier; some classifiers have random elements, and
	# for our purpose, we are working with a specific instance, so we fix
	# the classifier's behavior with a random seed
	if classifier_type == 1: classifier = classifier_class(random_state=seed) 
	else : classifier = classifier_class()
	
	# initialize pseudo-random number generation
	prng = random.Random()
	prng.seed(seed)

	print("Computing initial classifier performance...")
	referenceClassifier = copy.deepcopy(classifier)
	referenceClassifier.fit(X_train, y_train)
	y_train_pred = referenceClassifier.predict(X_train)
	y_test_pred = referenceClassifier.predict(X_test)
	y_pred = referenceClassifier.predict(X)
	trainAccuracy = accuracy_score(y_train, y_train_pred)
	testAccuracy = accuracy_score(y_test, y_test_pred)
	overallAccuracy = accuracy_score(y, y_pred)
	print("Initial performance: train=%.4f, test=%.4f, overall=%.4f" % (trainAccuracy, testAccuracy, overallAccuracy))

	print("\nSetting up evolutionary algorithm...")
	ea = inspyred.ec.emo.NSGA2(prng)
	ea.variator = [ variate ]
	ea.terminator = inspyred.ec.terminators.generation_termination
	ea.observer = observeCoreSets

	final_population = ea.evolve(    
					generator = generateCoreSets,
					evaluator = evaluateCoreSets,
					pop_size = pop_size,
					num_selected = offspring_size,
					maximize = maximize, 
					max_generations = max_generations,
					
					# extra arguments here
					n_classes = number_classes,
					classifier = classifier,
					X=X,
					y=y,
					X_train = X_train,
					y_train = y_train,
					X_test = X_test,
					y_test = y_test,
					min_points_in_core_set = min_points_in_core_set, 
					max_points_in_core_set = max_points_in_core_set,
					experimentName = experiment_name,
					all_population_file = all_population_file,
					current_time = datetime.datetime.now()
					)

	final_archive = sorted(ea.archive, key = lambda x : x.fitness[1])

	return final_archive, trainAccuracy, testAccuracy

def setup_logger(name, log_file, level=logging.INFO):
	"""Function setup as many loggers as you want"""
	
	formatter = logging.Formatter('%(asctime)s %(message)s')
	handler = logging.FileHandler(log_file)        
	handler.setFormatter(formatter)
	
	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.addHandler(handler)
	
	return logger

# utility function to load the covtype dataset
def loadForestCoverageType() :

	inputFile = "../data/covtype.csv"
	#logger.info("Loading file \"" + inputFile + "\"...")
	df_covtype = read_csv(inputFile, delimiter=',', header=None)

	# class is the last column
	covtype = df_covtype.as_matrix()
	X = covtype[:,:-1]
	y = covtype[:,-1].ravel()-1

	return X, y

def loadMNIST():
	mnist = tf.keras.datasets.mnist
	(x_train, y_train),(x_test, y_test) = mnist.load_data()
	
	X = np.concatenate((x_train, x_test))
	X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[1]))
	y = np.concatenate((y_train, y_test))
	
	return X, y

def make_meshgrid(x, y, h=.02):
	x_min, x_max = x.min() - 1, x.max() + 1
	y_min, y_max = y.min() - 1, y.max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
						np.arange(y_min, y_max, h))
	return xx, yy

def plot_contours(clf, xx, yy, **params):
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	out = plt.contour(xx, yy, Z, **params)
	return out, Z

def evaluate_core(X_core, y_core, X, y, classifier, cname=None, SEED=0):
	
	if cname == "SVC":
		referenceClassifier = copy.deepcopy(classifier(random_state=SEED, probability=True))
	else:
		referenceClassifier = copy.deepcopy(classifier(random_state=SEED))
	referenceClassifier.fit(X_core, y_core)
	y_pred = referenceClassifier.predict(X)
	
	fail_points = y != y_pred
	
	X_err = X[fail_points]
	accuracy = accuracy_score( y, y_pred)
	
	return X_err, accuracy, referenceClassifier, fail_points, y_pred

# initial random generation of core sets (as binary strings)
def generateCoreSets(random, args) :

	individual_length = args["X_train"].shape[0] + args["X_train"].shape[1]
	individual = [0] * individual_length
	
	points_in_core_set = random.randint( args["min_points_in_core_set"], args["max_points_in_core_set"] )
	for i in range(points_in_core_set) :
		random_index = random.randint(0, args["X_train"].shape[0]-1)
		individual[random_index] = 1
		
	features_in_core_set = random.randint( 1, args["X_train"].shape[1] )
	for i in range(features_in_core_set) :
		random_index = random.randint(args["X_train"].shape[0], individual_length-1)
		individual[random_index] = 1
	
	return individual

# using inspyred's notation, here is a single operator that performs both
# crossover and mutation, sequentially
@inspyred.ec.variators.crossover
def variate(random, parent1, parent2, args) :
	
	# well, for starters we just crossover two individuals, then mutate
	children = [ list(parent1), list(parent2) ]
	
	# one-point crossover!
	cutPoint = random.randint(0, len(children[0])-1)
	for index in range(0, cutPoint+1) :
		temp = children[0][index]
		children[0][index] = children[1][index]
		children[1][index] = temp 
	
	# mutate!
	for child in children : 
		mutationPoint = random.randint(0, len(child)-1)
		if child[mutationPoint] == 0 :
			child[mutationPoint] = 1
		else :
			child[mutationPoint] = 0
	
	# check if individual is still valid, and (in case it isn't) repair it
	for child in children :
		
		if args.get("max_points_in_core_set", None) != None and args.get("min_points_in_core_set", None) != None :
			
			points_in_core_set = [ index for index, value in enumerate(child) if value == 1 and index < args["X_train"].shape[0] ]
			
			while len(points_in_core_set) > args["max_points_in_core_set"] :
				index = random.choice( points_in_core_set )
				child[index] = 0
				points_in_core_set = [ index for index, value in enumerate(child) if value == 1 ]
			
			if len(points_in_core_set) < args["min_points_in_core_set"] :
				index = random.choice( [ index for index, value in enumerate(child) if value == 0 ] )
				child[index] = 1
				points_in_core_set = [ index for index, value in enumerate(child) if value == 1 ]
				
			features_in_core_set = [ index for index, value in enumerate(child) if value == 1 and index > args["X_train"].shape[0] ]
			
			if len(features_in_core_set) < 1 :
				index = random.choice( [ index for index, value in enumerate(child) if value == 0 and index > args["X_train"].shape[0] ] )
				child[index] = 1
				features_in_core_set = [ index for index, value in enumerate(child) if value == 1 and index > args["X_train"].shape[0] ]
	
	return children

# function that evaluates the core sets
def evaluateCoreSets(candidates, args) :
	fitness = []

	for c in candidates :
		#print("candidate:", c)
		cAsBoolArray = np.array(c, dtype=bool)
		
		cPoints = cAsBoolArray[ :args["X_train"].shape[0] ]
		cFeatures = cAsBoolArray[ args["X_train"].shape[0]: ]
		
		X_train_reduced = args["X_train"][cPoints, :]
		X_train_reduced = X_train_reduced[:, cFeatures]
		y_train_reduced = args["y_train"][cPoints]

		#print("Reduced training set:", X_train_reduced.shape[0])
		#print("Reduced training set:", y_train_reduced.shape[0])
		
		if len(set(y_train_reduced)) == args["n_classes"] :
			classifier = copy.deepcopy( args["classifier"] )
			classifier.fit(X_train_reduced, y_train_reduced)
			
			# evaluate accuracy for every point (training, test)
			X_train = args["X_train"]
			y_pred_train = classifier.predict( X_train[:, cFeatures] )
			#y_pred_test = classifier.predict( args["X_test"] )
			#y_pred = np.concatenate((y_pred_train, y_pred_test))
			#y = np.concatenate((args["y_train"], args["y_test"]))
			#accuracy = accuracy_score(y, y_pred)
			
			accuracy = accuracy_score(args["y_train"], y_pred_train)
			error = round(1-accuracy, 4)
			
			# also store valid individual
#			all_population_file = args.get("all_population_file", None)
#			if all_population_file != None :
#				
#				# if the file does not exist, write header
#				if not os.path.exists(all_population_file) : 
#					with open(all_population_file, "w") as fp :
#						fp.write("#points,accuracy,individual\n")
#				
#				# in any case, append individual
#				with open(all_population_file, "a") as fp :
#					fp.write( str(len([ x for x in c if x == 1])) )
#					fp.write( "," + str(accuracy) )
#					
#					for g in c :
#						fp.write( "," + str(g) )
#					fp.write("\n")
		else:
			# individual gets a horrible fitness value
			maximize = args["_ec"].maximize # let's fetch the bool that tells us if we are maximizing or minimizing
			if maximize == True :
				error = -np.inf
			else :
				error = np.inf
				
		# maximizing the points removed also means minimizing the number of points taken (LOL)
		corePoints = sum(cPoints)
		coreFeatures = sum(cFeatures)
		fitness.append( inspyred.ec.emo.Pareto( [corePoints, coreFeatures, error] ) )
	
	return fitness

# the 'observer' function is called by inspyred algorithms at the end of every generation
def observeCoreSets(population, num_generations, num_evaluations, args) :
	
#	training_set_size = args["X_train"].shape[0]
	old_time = args["current_time"]
	current_time = datetime.datetime.now()
	delta_time = current_time - old_time 
	
	# I don't like the 'timedelta' string format, so here is some fancy formatting
	delta_time_string = str(delta_time)[:-7] + "s"
	
	print("[%s] Generation %d, Random individual: #samples=%d, #features=%d, error=%.2f" % (delta_time_string, num_generations, population[0].fitness[0], population[0].fitness[1], population[0].fitness[2]))
	
	args["current_time"] = current_time

	return

if __name__ == "__main__" :
	
#	dataList = [
#		["blobs", 200, 1000],
#		["circles", 200, 1000],
#		["moons", 200, 1000],
#		["iris4", 200, 1000],
#		["iris2", 500, 500],
#		["digits", 200, 1000],
#		#["covtype", 10, 5],
#		#["mnist", 10, 5],
#		]
	
	dataList = [
#		["blobs", 200, 200],
#		["circles", 200, 200],
#		["moons", 200, 200],
#		["iris4", 20, 20],
#		["iris2", 200, 200],
#		["digits", 200, 200],
#		["wine", 200, 200],
#		["breast", 200, 200],
#		["pairs", 200, 200],
#		["olivetti", 200, 200],
#		["covtype", 200, 200],
#		["mnist", 200, 200],
#		["plants", 200, 200],
#		["isolet", 200, 200],
#		["ctg", 200, 200],
#		["ozone", 200, 200],
#		["ilpd", 200, 200],
#		["biodeg", 200, 200],
#		["hill-valley", 200, 200],
		]
	
#	dataList = [
#		["blobs", 100, 100],
#		["circles", 100, 100],
#		["moons", 100, 100],
#		["iris4", 100, 100],
#		["iris2", 100, 100],
#		["digits", 100, 100],
#		["covtype", 100, 100],
#		["mnist", 100, 100],
#		]
	for dataset in dataList:
		main(dataset[0], dataset[1], dataset[2])
	sys.exit()
