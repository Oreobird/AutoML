Detailed design doc, please reference to:

![image](https://github.com/Oreobird/effect3d/blob/master/wechat.jpg)
# AutoML
Auto machine learning framework based on sklearn, mlxtend, etc.

# Usage
# Config file format explanation.
	
	#define problem: binary classify, multi-class classify, reggression and cluster.
	[basic]
	model_type = binary | multi | reg | cluster 

	#define metrics
	[binary_clf_metrics]
	accuracy = true
	precision = true

	[multi_clf_metrics]
	accuracy = true
	precision = true

	[reg_metrics]
	explained_variance = true
	neg_mean_absolute_error = true

	[cluster_metrics]
	adjusted_mutual_info_score = true
	adjusted_rand_score = true

	#define models
	[clf_models]
	LR = true
	SVM = true
	DecisionTree = false
	RandomForest = false
	xgboost = true

	[reg_models]
	RandomForest = true

	[cluster_models]
	KMeans = true

	#define meta-model used in stacking
	[meta_models]
	lgbm = true
 	
 # API.
	Step 1. define cfg_obj.
	cfg_obj = config_parser.CfgParser(os.path.join(CFG_FILE_PATH, 'binary_config.ini'))
	
	Step 2. parse metric, basic model and meta-model.
	metric_list, model_list = cfg.parse_metrics_models()
	meta_model_label = cfg.parse_meta_models()
	
	Step 3. This step is optional. Define model_util_obj for model fine-tune param set. Need to define your own model_dict and meta_model_dict first. model_dict format: {'model_label': [model_obj, {param set}]}. 
	model_dict = {'lr': [LogisticRegression(), {'C': [x / 10.0 for x in range(1, 50, 5)]}]}
	model_util_obj = model_util.ModelUtil(model_dict, meta_model_dict)

	Step 4. define automl_obj, model_util is optional, if not provided, use default model dict to fine-tune.
	automl_obj = automl_base.AutoML(model_util=model_util_obj, model_save_path=os.path.join(MODEL_FILE_PATH, 'iris_models/'))

	# Step 5. Auto train, select, fine-tune and save models.
	model = automl_obj.train(X_train, Y_train, metric_list, model_label_list, meta_model_label[0], model_save_name='iris_model.pkl', K=3)

	# Step 6. validate model.
	val_y = automl.validate(model, X_test, Y_test, metric_list)

	# Step 7. predict model.
	pred_y = automl.predict(model, X_test)
