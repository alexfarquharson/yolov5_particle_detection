train_data_params = {
   'standard_train_data' : {'subdir' : 'Dataset_standard/',
               'nimages' : [3000,1500,0],
               'folders' : ['train', 'valid', 'test'],
               # image parameters
               'image_w' : 640,
               'image_h' : 640,
               'image_d' : 1,
               'label_list' : ['Spot'],
               'snr_range' : [1,25],
               'offset' : 15,
               'diameter_mean' : 12,
               'diameter_std' : 2,
               'luminosity_range' : [0.8,1],
               'density_range' : [0.01, 0.25],
               'impurity_type' : "None",
                'distance_factor' : 0.3},
                
                'big_std_train_data' : {'subdir' : 'Dataset_big_std/',
                'nimages' : [2000, 1000, 0],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : [4,25],
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 5,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.01, 0.12],
                'impurity_type' : "None",
                'distance_factor' : 0.5},

                'impurities_train_data' : {'subdir' : 'Dataset_impurities/',
                        'nimages' : [2000, 1500, 0],                        
                        'folders' : ['train', 'valid', 'test'],
                        # image parameters
                        'image_w' : 640,
                        'image_h' : 640,
                        'image_d' : 1,
                        'label_list' : ['Spot'],
                        'snr_range' : [4,25],
                        'offset' : 15,
                        'diameter_mean' : 12,
                        'diameter_std' : 2,
                        'luminosity_range' : [0.8,1],
                        'density_range' : [0.005,0.05],
                        'impurity_type' : "Circular and Rectangle",
                        'distance_factor' : 0.5,
                        
                        'circle_impurities': True,
                        'label_list_impurity_circle' : ['Spot_impurity'],
                        'density_range_impurity_circle' : [0.005,0.012],
                        'diameter_mean_impurity_circle' : 5,
                        'diameter_std_impurity_circle' : 4,
                        
                        'rectangle_impurities':True,
                        'label_list_impurity_rectangle' : ["Rectangle_impurity"],
                        'density_range_impurity_rectangle' : [0.005, 0.012],
                        'length_mean_impurity_rectangle' : 12,
                        'length_std_impurity_rectangle' : 5}
                        }


test_data_params = {
    '1. Standard' : {'subdir' : 'Dataset_standard/test/1_standard/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 2,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "None",
                'distance_factor' : 1},

                '2.1 Density 10%' : {'subdir' : 'Dataset_standard/test/2_1_density_10/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 2,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.098, 0.12],
                'impurity_type' : "None",
                'distance_factor' : 1},

                '2.2 Density 20%' : {'subdir' : 'Dataset_standard/test/2_2_density_20/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 2,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.198, 0.22],
                'impurity_type' : "None",
                'distance_factor' : 1},

                '2.3 Density 40%' : {'subdir' : 'Dataset_standard/test/2_1_density_40/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 2,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.38, 0.42],
                'impurity_type' : "None",
                'distance_factor' : 1},

                '3.1 Big particle std 4 0' : {'subdir' : 'Dataset_big_std/test/3_1_big_particle_std_4_0/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 4,
                'diameter_std' : 0,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "None",
                'distance_factor' : 1},

                '3.2 Big particle std 6 0' : {'subdir' : 'Dataset_big_std/test/3_2_big_particle_std_6_0/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 6,
                'diameter_std' : 0,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "None",
                'distance_factor' : 1},

                '3.3 Big particle std 7 0' : {'subdir' : 'Dataset_big_std/test/3_3_big_particle_std_7_0/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 7,
                'diameter_std' : 0,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "None",
                'distance_factor' : 1},

                '3.4 Big particle std 9 0' : {'subdir' : 'Dataset_big_std/test/3_4_big_particle_std_9_0/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 9,
                'diameter_std' : 0,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "None",
                'distance_factor' : 1},

                '3.5 Big particle std 12 0' : {'subdir' : 'Dataset_big_std/test/3_5_big_particle_std_12_0/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 0,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "None",
                'distance_factor' : 1},

                '3.6 Big particle std 15 0' : {'subdir' : 'Dataset_big_std/test/3_6_big_particle_std_15_0/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 15,
                'diameter_std' : 0,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "None",
                'distance_factor' : 1},

                '3.7 Big particle std 17 0' : {'subdir' : 'Dataset_big_std/test/3_7_big_particle_std_17_0/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 17,
                'diameter_std' : 0,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "None",
                'distance_factor' : 1},

                '3.8 Big particle std 18 0' : {'subdir' : 'Dataset_big_std/test/3_8_big_particle_std_18_0/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 18,
                'diameter_std' : 0,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "None",
                'distance_factor' : 1},

                '3.9 Big particle std 20 0' : {'subdir' : 'Dataset_big_std/test/3_9_big_particle_std_20_0/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 20,
                'diameter_std' : 0,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "None",
                'distance_factor' : 1},


                '4.1 SNR 20' : {'subdir' : 'Dataset_standard/test/snr_20/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 20,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 2,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "None",
                'distance_factor' : 1},

                '4.2 SNR 10' : {'subdir' : 'Dataset_standard/test/snr_10/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 2,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "None",
                'distance_factor' : 1},

                '4.3 SNR 5' : {'subdir' : 'Dataset_standard/test/snr_5/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 5,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 2,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "None",
                'distance_factor' : 1},

                '4.4 SNR 3' : {'subdir' : 'Dataset_standard/test/snr_3/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 3,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 2,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "None",
                'distance_factor' : 1},

                '4.5 SNR 2' : {'subdir' : 'Dataset_standard/test/snr_2/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 2,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 2,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "None",
                'distance_factor' : 1},

                '4.6 SNR 1.5' : {'subdir' : 'Dataset_standard/test/snr_1.5/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 1.5,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 2,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "None",
                'distance_factor' : 1},

                '5.1 Circular impurities size 4 0' : {'subdir' : 'Dataset_impurities/test/5_1_circular_impurities_size_4_0/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 2,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "Circular",
                'distance_factor' : 1,
                        'circle_impurities': True,
                        'label_list_impurity_circle' : ['Spot_impurity'],
                        'density_range_impurity_circle' : [0.008,0.012],
                        'diameter_mean_impurity_circle' : 4,
                        'diameter_std_impurity_circle' : 0,
                        },

                '5.2 Circular impurities size 6 0' : {'subdir' : 'Dataset_impurities/test/5_2_circular_impurities_size_6_0/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 2,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "Circular",
                'distance_factor' : 1,
                        'circle_impurities': True,
                        'label_list_impurity_circle' : ['Spot_impurity'],
                        'density_range_impurity_circle' : [0.008,0.012],
                        'diameter_mean_impurity_circle' : 6,
                        'diameter_std_impurity_circle' : 0,
                        },

                '5.3 Circular impurities size 8 0' : {'subdir' : 'Dataset_impurities/test/5_3_circular_impurities_size_8_0/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 2,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "Circular",
                'distance_factor' : 1,
                        'circle_impurities': True,
                        'label_list_impurity_circle' : ['Spot_impurity'],
                        'density_range_impurity_circle' : [0.008,0.012],
                        'diameter_mean_impurity_circle' : 8,
                        'diameter_std_impurity_circle' : 0,
                        },

                '5.4 Circular impurities size 9 0' : {'subdir' : 'Dataset_impurities/test/5_4_circular_impurities_size_8_0/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 2,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "Circular",
                'distance_factor' : 1,
                        'circle_impurities': True,
                        'label_list_impurity_circle' : ['Spot_impurity'],
                        'density_range_impurity_circle' : [0.008,0.012],
                        'diameter_mean_impurity_circle' : 9,
                        'diameter_std_impurity_circle' : 0,
                        },


                '6.1 Rectangle impurities size 12 5' : {'subdir' : 'Dataset_impurities/test/5_1_circular_impurities_size_12_5/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 2,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "Rectangle",
                'distance_factor' : 1,

                        'rectangle_impurities': True,
                        'label_list_impurity_rectangle' : ["Rectangle_impurity"],
                        'density_range_impurity_rectangle' : [0.008, 0.012],
                        'length_mean_impurity_rectangle' : 12,
                        'length_std_impurity_rectangle' : 5
                        },

                '7.0 Overlap 0.3_0.5' : {'subdir' : 'Dataset_standard/test/7_0_overlap_0.3_0.5/',
                                'nimages' : [0,0,500],
                                'folders' : ['train', 'valid', 'test'],
                                # image parameters
                                'image_w' : 640,
                                'image_h' : 640,
                                'image_d' : 1,
                                'label_list' : ['Spot'],
                                'snr_range' : 10,
                                'offset' : 15,
                                'diameter_mean' : 12,
                                'diameter_std' : 2,
                                'luminosity_range' : [0.8,1],
                                'density_range' : [0.002,0.0022],
                                'impurity_type' : "None",
                                'distance_factor' : 0.3,
                                "max_distance_factor": 0.5},

                '7.1 Overlap 0.5_0.7' : {'subdir' : 'Dataset_standard/test/7_1_overlap_0.5_0.7/',
                                'nimages' : [0,0,500],
                                'folders' : ['train', 'valid', 'test'],
                                # image parameters
                                'image_w' : 640,
                                'image_h' : 640,
                                'image_d' : 1,
                                'label_list' : ['Spot'],
                                'snr_range' : 10,
                                'offset' : 15,
                                'diameter_mean' : 12,
                                'diameter_std' : 2,
                                'luminosity_range' : [0.8,1],
                                'density_range' : [0.002,0.0022],
                                'impurity_type' : "None",
                                'distance_factor' : 0.5,
                                "max_distance_factor": 0.7},

                '7.2 Overlap 0.7_0.9' : {'subdir' : 'Dataset_standard/test/7_1_overlap_0.7_0.9/',
                                'nimages' : [0,0,500],
                                'folders' : ['train', 'valid', 'test'],
                                # image parameters
                                'image_w' : 640,
                                'image_h' : 640,
                                'image_d' : 1,
                                'label_list' : ['Spot'],
                                'snr_range' :10,
                                'offset' : 15,
                                'diameter_mean' : 12,
                                'diameter_std' : 2,
                                'luminosity_range' : [0.8,1],
                                'density_range' : [0.002,0.0022],
                                'impurity_type' : "None",
                                'distance_factor' : 0.7,
                                "max_distance_factor": 0.9},

                '7.3 Overlap 0.9_1.1' : {'subdir' : 'Dataset_standard/test/7_1_overlap_0.9_1.1/',
                                'nimages' : [0,0,500],
                                'folders' : ['train', 'valid', 'test'],
                                # image parameters
                                'image_w' : 640,
                                'image_h' : 640,
                                'image_d' : 1,
                                'label_list' : ['Spot'],
                                'snr_range' : 10,
                                'offset' : 15,
                                'diameter_mean' : 12,
                                'diameter_std' : 2,
                                'luminosity_range' : [0.8,1],
                                'density_range' : [0.002,0.0022],
                                'impurity_type' : "None",
                                'distance_factor' : 0.9,
                                "max_distance_factor": 1.1},

                '7.4 Overlap 1.1_1.3' : {'subdir' : 'Dataset_standard/test/7_1_overlap_1.1_1.3/',
                                'nimages' : [0,0,500],
                                'folders' : ['train', 'valid', 'test'],
                                # image parameters
                                'image_w' : 640,
                                'image_h' : 640,
                                'image_d' : 1,
                                'label_list' : ['Spot'],
                                'snr_range' : 10,
                                'offset' : 15,
                                'diameter_mean' : 12,
                                'diameter_std' : 2,
                                'luminosity_range' : [0.8,1],
                                'density_range' : [0.002,0.0022],
                                'impurity_type' : "None",
                                'distance_factor' : 1.1,
                                "max_distance_factor": 1.3},

                '7.5 Overlap 1.3_1.5' : {'subdir' : 'Dataset_standard/test/7_1_overlap_1.3_1.5/',
                                'nimages' : [0,0,500],
                                'folders' : ['train', 'valid', 'test'],
                                # image parameters
                                'image_w' : 640,
                                'image_h' : 640,
                                'image_d' : 1,
                                'label_list' : ['Spot'],
                                'snr_range' : 10,
                                'offset' : 15,
                                'diameter_mean' : 12,
                                'diameter_std' : 2,
                                'luminosity_range' : [0.8,1],
                                'density_range' : [0.002,0.0022],
                                'impurity_type' : "None",
                                'distance_factor' : 1.3,
                                "max_distance_factor": 1.5},

                '7.7 Overlap 1.7_1.9' : {'subdir' : 'Dataset_standard/test/7_1_overlap_1.7_1.9/',
                                'nimages' : [0,0,500],
                                'folders' : ['train', 'valid', 'test'],
                                # image parameters
                                'image_w' : 640,
                                'image_h' : 640,
                                'image_d' : 1,
                                'label_list' : ['Spot'],
                                'snr_range' : 10,
                                'offset' : 15,
                                'diameter_mean' : 12,
                                'diameter_std' : 2,
                                'luminosity_range' : [0.8,1],
                                'density_range' : [0.002,0.0022],
                                'impurity_type' : "None",
                                'distance_factor' : 1.7,
                                "max_distance_factor": 1.9},

}


extra_test_data_params = {
    '4.7 SNR 1' : {'subdir' : 'Dataset_standard/test/snr_1/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 1,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 2,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.018, 0.022],
                'impurity_type' : "None",
                'distance_factor' : 1},

                '2.25 Density 30%' : {'subdir' : 'Dataset_standard/test/2_25_density_30/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 12,
                'diameter_std' : 2,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.28, 0.32],
                'impurity_type' : "None",
                'distance_factor' : 1},

                '3.00 Big particle std 1 0' : {'subdir' : 'Dataset_big_std/test/3_00_big_particle_std_1_0/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 1,
                'diameter_std' : 0,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.0018, 0.0022],
                'impurity_type' : "None",
                'distance_factor' : 1},

                '3.0 Big particle std 2 0' : {'subdir' : 'Dataset_big_std/test/3_0_big_particle_std_2_0/',
                'nimages' : [0,0,500],
                'folders' : ['train', 'valid', 'test'],
                # image parameters
                'image_w' : 640,
                'image_h' : 640,
                'image_d' : 1,
                'label_list' : ['Spot'],
                'snr_range' : 10,
                'offset' : 15,
                'diameter_mean' : 2,
                'diameter_std' : 0,
                'luminosity_range' : [0.8,1],
                'density_range' : [0.0018, 0.0022],
                'impurity_type' : "None",
                'distance_factor' : 1},

                '7.000 Overlap 0.05_0.15' : {'subdir' : 'Dataset_standard/test/7_00_overlap_0.05_0.15/',
                                'nimages' : [0,0,500],
                                'folders' : ['train', 'valid', 'test'],
                                # image parameters
                                'image_w' : 640,
                                'image_h' : 640,
                                'image_d' : 1,
                                'label_list' : ['Spot'],
                                'snr_range' : 10,
                                'offset' : 15,
                                'diameter_mean' : 12,
                                'diameter_std' : 2,
                                'luminosity_range' : [0.8,1],
                                'density_range' : [0.002,0.0022],
                                'impurity_type' : "None",
                                'distance_factor' : 0.05,
                                "max_distance_factor": 0.15},

                '7.00 Overlap 0.15_0.3' : {'subdir' : 'Dataset_standard/test/7_0_overlap_0.15_0.3/',
                                'nimages' : [0,0,500],
                                'folders' : ['train', 'valid', 'test'],
                                # image parameters
                                'image_w' : 640,
                                'image_h' : 640,
                                'image_d' : 1,
                                'label_list' : ['Spot'],
                                'snr_range' : 10,
                                'offset' : 15,
                                'diameter_mean' : 12,
                                'diameter_std' : 2,
                                'luminosity_range' : [0.8,1],
                                'density_range' : [0.002,0.0022],
                                'impurity_type' : "None",
                                'distance_factor' : 0.15,
                                "max_distance_factor": 0.3}
}
