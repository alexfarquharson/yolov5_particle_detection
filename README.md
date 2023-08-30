Repository for creating synthetic data for flurescent particle detection. And training and evaluating a YOLOv5 model for object detection.

* Clone the repository at: https://github.com/alexfarquharson/yolov5_particle_detection
* In the data_creation folder, create the environmnt, and create the synthetic data by running * the synthetic_data.py and the data_yaml_files_creator.py scripts.
* Create the environment in the root directory and train the model using the train.py script, * referncing the yaml training data configuration file created above. To use the n\_standard trained models' weights, use the xxx file.
* Evaluate the model using the val_extra.py script.


YOLOv5 repository off which this is modified: https://github.com/ultralytics/yolov5
