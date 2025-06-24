In the root folder one can find:
	- data_split_metadata.csv file that provides information on the data split (dev and test) with the corresponding path to the corresponding file.
		- dev data: these data can be used for developing new methods. The dev data can be further split into training and validation by the user to create a validation dataset for hyperparameter tuning, overfitting detection, and model selection.
		- test data: these data aim to be used for the final evaluation of a method (i.e., either pre-existing or developed using the training data). These data should strictly not be used in any step of model training, and to avoid any information leak, the user should limit the number of times the test data is used.


	- one folder per collection with the following information:
		- Tree data
   		These consist in field measured tree properties. DBH will be available for all annotated trees. In addition, in some datasets other variables are available (e.g. tree species, H, V, AGB)
			Variables:
	 			plotID: unique identifier for the plots
	 			treeID: treeID matching ID numbering in annotated trees 
	 			DBH   : diameter (cm) at breast height (1.3 m)

		- Fully annotated point clouds (*.las
  		 The annotations can be accessed through the following fields:
			treeID: plotwise unique identifier annotated trees

			Classification
			0: Unclassified (scattered points that were not annotated)
	 		1: Low-vegetation (anything that is not a tree or ground)
	 		2: Terrain
	 		3: Out-points (trees outside of the measured/annotated plots)
	 		4: Stem 
	 		5: Live-branches (green crown)
	 		6: Woody-branches