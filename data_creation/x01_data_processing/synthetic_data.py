import numpy as np
pi = np.pi
import matplotlib.pyplot as plt

import os
import matplotlib.patches as patches

from tqdm import tqdm # Progress bar

import configparser
from scipy import ndimage
from xml_writer import Writer # Writes XML files in the Pascal VOC format 
import xml.etree.ElementTree as ET
import os

from data_configs import train_data_params, test_data_params, extra_test_data_params

class SpotObject:
    '''
    Object class
    
    x: x position
    y: y position
    label: particle type
    diameter: particle diameter
    parameters: particle parameters
    '''
    
    def __init__(self, x, y, z, label, diameter, theta=None): # , theta
        self.x = x
        self.y = y
        self.z = z
        self.theta = theta
        self.label = label
        self.diameter = diameter


class RectangleObject:
    '''
    Object class
    
    x: x position
    y: y position
    label: particle type
    length: length diameter
    width: width parameters
    '''
    
    def __init__(self, x, y, label, length, width, theta=None): # , theta
        self.x = x
        self.y = y
        self.theta = theta
        self.label = label
        self.length = length
        self.width = width


def create_particles(frames, image_w, image_h, image_d, offset, label_list, diameter_mean, diameter_std, density_range, n_list = None, 
                     distance_factor = 1, luminosity_range = 1, fixed_camera_distance = 10, min_dist_list = []):
    ''' 
    input:
        frames: no. pictures
        n_list: [number of particles type 1, number of particles type 2]
        image_w: image width
        image_h: image height
        image_d: image depth (particles will be placed at different d -which will affect luminosty and size respectively).
            If set to 1, then there will be no variance in d (and luminosity) and no change to d or luminosity.
        distance: min_distance x_y euclidean between any two particles
        offset: random distance (must be < image_w/2)
        label_list: list of particle types (must be same length as n_list)
        diameter_mean: int (normal dist particle diameter mean)
        diamaeter_std: int (normal dist particle diameter std)
        density_range: list [min, max] area density of particles
        luminosity_range: int or range list of relative particle luminosity (e.g. [1] (no change), [0.9,1] will vary from 90 to 100%)
        fixed_camera_distance: ditance of image to camera
        
    output:
        array of Objects

    '''
    # Check and process inputs
    density_range.sort()
    # come up with density here and use it to set no. particles
    if n_list == None:
        n_list = []
        proxy_density = np.random.uniform(density_range[0], density_range[1])
        n_list = [int(np.floor((proxy_density * (image_w * image_h)) / (pi * diameter_mean ** 2)))]
    if not isinstance(n_list, list): 
        n_list = [n_list]
    if not isinstance(label_list, list):
        label_list = [label_list]
    if len(n_list) != len(label_list):
        raise ValueError('The lists must have equal length')
    if not isinstance(luminosity_range, list):
        luminosity_range = [luminosity_range, 1]
    if not (len(luminosity_range) == 1) | (len(luminosity_range) == 2):
        raise ValueError('luminosity_range must either be list of length 1 or 2')
    
    # set min dist as 4 std from 2 particle centroids * distance factor (=1 then 4 std)
    distance = 2*diameter_mean * distance_factor
    diameter_range = [diameter_mean, diameter_std]
    

    objects = []
    for _ in range(frames):
        
        # ensure picture within density boundaries
        density = 0
        while (density < density_range[0]) | (density > density_range[1]):
    #         get random x,y positions of next particle
            x_y_positions = np.random.random(2)*[image_w - 2*offset, image_h - 2*offset] + offset
    #         get random z position of all particles
            z_positions = np.random.randint(0, image_d, n_list[0])
    #             get fixed diameter of particle set by user (relative diameter size can be between [0] and [1])
            if len(diameter_range) == 2:
                fixed_diameters = np.random.normal(diameter_range[0], diameter_range[1], n_list[0])
            else:
                fixed_diameters = np.ones(n_list[0]) * diameter_range[0]

    #             get diameter and intensity given the z coordinate
            camera_distance = np.ones(n_list[0]) * fixed_camera_distance
            fixed_intensity = np.random.uniform(luminosity_range[0], luminosity_range[1], n_list[0])
            diameters = (fixed_diameters * camera_distance) / (camera_distance + z_positions)
            # ensure that diameter is not outside of 1 std range
            diameters = np.clip(diameters, diameter_mean - diameter_std, diameter_mean + diameter_std)
            intensities = (fixed_intensity * camera_distance**2) / (camera_distance + z_positions)**2

            # check density
            areas = 0
            for particle_diameter in diameters:
                areas += pi * (particle_diameter)**2 # say area is centroid + 2 std from centre (i.e. radius*2 = diameter)
            image_area = image_w * image_h
            density = areas / image_area
            if density < density_range[0]:
                n_list[0] +=1
            elif density > density_range[0]:
                n_list[0] -=1
                assert n_list[0] != 0, "No particles, please select a higher density or lower particle size"

        for _ in range(np.sum(n_list)-1):           
            min_distance = 0
#             change position until it is greater than distance parameter (x_y euclidean distance only)
            while min_distance < distance:
                new_pos = np.random.random(2)*[image_w - 2*offset + offset, image_h - 2*offset + offset]
                pos = x_y_positions.reshape(int(len(x_y_positions)/2), 2)
                d = pos - new_pos
                min_distance = np.sqrt(np.sum(d*d, axis=1)).min()
                if min_distance > distance:
                    min_dist_list.append(min_distance)

            x_y_positions = np.append(x_y_positions, new_pos)

#         create object instances given the x,y,z position, label, diameter and intensity
        x_y_positions = x_y_positions.reshape(np.sum(n_list), 2)
        label_list = np.repeat(label_list, n_list).tolist()
        objects.append([SpotObject(x,y,z, label, diameter, theta) for (x, y), z, label, diameter, theta in zip(x_y_positions, z_positions, label_list, diameters, intensities)])
     
    return np.array(objects), min_dist_list



def create_particles_custom_overlap(frames, image_w, image_h, image_d, offset, label_list, diameter_mean, diameter_std, density_range, n_list = None, 
                     distance_factor = 1, max_distance_factor = 10000, luminosity_range = 1, fixed_camera_distance = 10, min_dist_list = []):
    ''' 
    input:
        frames: no. pictures
        n_list: [number of particles type 1, number of particles type 2]
        image_w: image width
        image_h: image height
        image_d: image depth (particles will be placed at different d -which will affect luminosty and size respectively).
            If set to 1, then there will be no variance in d (and luminosity) and no change to d or luminosity.
        distance_factor: min no. of particle stds x_y euclidean between any two particles
        max_distance_factor: max no. of particle stds x_y euclidean between any two particles
        offset: random distance (must be < image_w/2)
        label_list: list of particle types (must be same length as n_list)
        diameter_mean: int (normal dist particle diameter mean)
        diamaeter_std: int (normal dist particle diameter std)
        density_range: list [min, max] area density of particles
        luminosity_range: int or range list of relative particle luminosity (e.g. [1] (no change), [0.9,1] will vary from 90 to 100%)
        fixed_camera_distance: ditance of image to camera
        
    output:
        array of Objects

    '''
    # Check and process inputs
    density_range.sort()
    # come up with density here and use it to set no. particles
    if n_list == None:
        n_list = []
        proxy_density = np.random.uniform(density_range[0], density_range[1])
        n_list = [int(np.floor((proxy_density * (image_w * image_h)) / (pi * diameter_mean ** 2)))]
    if not isinstance(n_list, list): 
        n_list = [n_list]
    if not isinstance(label_list, list):
        label_list = [label_list]
    if len(n_list) != len(label_list):
        raise ValueError('The lists must have equal length')
    if not isinstance(luminosity_range, list):
        luminosity_range = [luminosity_range, 1]
    if not (len(luminosity_range) == 1) | (len(luminosity_range) == 2):
        raise ValueError('luminosity_range must either be list of length 1 or 2')
    
    # set min dist as 4 std from 2 particle centroids * distance factor (=1 then 4 std)
    min_distance_param = 2*diameter_mean * distance_factor
    max_distance_param = 2*diameter_mean * max_distance_factor
    diameter_range = [diameter_mean, diameter_std]
    
    n_list = [2]

    objects = []
    for _ in range(frames):
    
#         get random x,y positions of next particle
        x_y_positions = np.random.random(2)*[image_w - 2*offset, image_h - 2*offset] + offset
#         get random z position of all particles
        z_positions = np.random.randint(0, image_d, n_list[0])
        
#             get fixed diameter of particle set by user (relative diameter size can be between [0] and [1])
        if len(diameter_range) == 2:
            fixed_diameters = np.random.normal(diameter_range[0], diameter_range[1], n_list[0])
        else:
            fixed_diameters = np.ones(n_list[0]) * diameter_range[0]

#             get diameter and intensity given the z coordinate
        camera_distance = np.ones(n_list[0]) * fixed_camera_distance
        fixed_intensity = np.random.uniform(luminosity_range[0], luminosity_range[1], n_list[0])
        diameters = (fixed_diameters * camera_distance) / (camera_distance + z_positions)
        # ensure that diameter is not outside of 1 std range
        diameters = np.clip(diameters, diameter_mean - diameter_std, diameter_mean + diameter_std)
        intensities = (fixed_intensity * camera_distance**2) / (camera_distance + z_positions)**2

        for _ in range(np.sum(n_list)-1):           
            min_distance = 0

#             change position until it is greater than distance parameter (x_y euclidean distance only)
            while (min_distance < min_distance_param) | (max_distance_param < min_distance):
                new_pos = np.random.random(2)*[image_w - 2*offset + offset, image_h - 2*offset + offset]
                pos = x_y_positions.reshape(int(len(x_y_positions)/2), 2)
                d = pos - new_pos
                min_distance = np.sqrt(np.sum(d*d, axis=1)).min()
                if (min_distance > min_distance_param) & (min_distance < max_distance_param):
                    min_dist_list.append(min_distance)

            x_y_positions = np.append(x_y_positions, new_pos)

#         create object instances given the x,y,z position, label, diameter and intensity
        x_y_positions = x_y_positions.reshape(np.sum(n_list), 2)
        label_list = np.repeat(label_list, n_list).tolist()
        objects.append([SpotObject(x,y,z, label, diameter, theta) for (x, y), z, label, diameter, theta in zip(x_y_positions, z_positions, label_list, diameters, intensities)])
     
    return np.array(objects), min_dist_list




def create_rectangles(frames, image_w, image_h, offset, label_list, length_mean, length_std, density_range, n_list, distance = None):
    ''' 
    input:
        frames: no. pictures
        n_list: [number of particles type 1, number of particles type 2]
        image_w: image width
        image_h: image height
        offset: random distance (must be < image_w/2)
        label_list: list of particle types (must be same length as n_list)
        length_mean: int (normal dist particle length mean)
        length_std: int (normal dist particle length std)
        density_range: list [min, max] area density of particles
        distance: min euc distance between particles (default is particle length)
        
    output:
        array of Objects

    '''
    # Check and process inputs
    density_range.sort()
    if n_list == None:
        n_list = []
        proxy_density = np.random.uniform(density_range[0], density_range[1])
        n_list = [int(np.floor((proxy_density * (image_w * image_h)) / (pi * length_mean ** 2)))]
    if not isinstance(n_list, list): 
        n_list = [n_list]
    if not isinstance(label_list, list):
        label_list = [label_list]
    if len(n_list) != len(label_list):
        raise ValueError('The lists must have equal length')
    
    # set min dist as length of particle
    if distance == None:
        distance = length_mean

    objects = []
    for _ in range(frames):
        
        density = 0
        while density < density_range[0] or density > density_range[1]:
    #         get random x,y positions of next particle
            x_y_positions = np.random.random(2)*[image_w - 2*offset, image_h - 2*offset] + offset
    #         get length and widths and ensure no greater than length +/-1 std
            intensities = np.random.uniform(0.9, 1, n_list[0])
            lengths = np.random.normal(length_mean, length_std, n_list[0])
            lengths = np.clip(lengths, length_mean - length_std, length_mean + length_std)
            widths = lengths * np.random.uniform(0.9, 1.1, n_list[0])
    #       check density 
            areas = 0
            for particle_length, particle_width in zip(lengths, widths):
                areas += particle_length * particle_width
            image_area = image_w * image_h
            density = areas / image_area
            if density < density_range[0]:
                n_list[0] +=1
            elif density > density_range[0]:
                n_list[0] -=1
                assert n_list[0] != 0, "No rectangle particles, please select a higher density or lower particle size"

        for _ in range(np.sum(n_list)-1):           
            min_distance = 0
#             change position until it is greater than distance parameter (x_y euclidean distance only)
            while min_distance < distance:
                new_pos = np.random.random(2)*[image_w - 2*offset + offset, image_h - 2*offset + offset]
                pos = x_y_positions.reshape(int(len(x_y_positions)/2), 2)
                d = pos - new_pos
                min_distance = np.sqrt(np.sum(d*d, axis=1)).min()
            x_y_positions = np.append(x_y_positions, new_pos)

#         create object instances given the x,y position, label, intensity, length, width
        x_y_positions = x_y_positions.reshape(np.sum(n_list), 2)
        label_list = np.repeat(label_list, n_list).tolist()
        
        objects.append([RectangleObject(x=x,y=y,label=label, theta=theta, length=length, width=width) for 
                        (x, y), label, theta, length, width in zip(x_y_positions, label_list, intensities, lengths, widths)])
     
    return np.array(objects)


def generateImage(objects, image_w, image_h, snr_range, impurity_objects = None):
    '''
    Input:
        objects: list of Object instances
        image_w: int image width
        image_h: int image height
        snr_range: list of either snr range min,max or snr fixed value
    '''
#     initiate all pixels as 0
    image = np.zeros([image_w, image_h])
    bboxes = []
    labels = []
#     X and Y are matrices of indexes
    X, Y = np.meshgrid(np.arange(0, image_w), np.arange(0, image_h))
    for obj in objects:
        x = obj.x
        y = obj.y
        i = obj.theta

        if obj.label == 'Spot':
#             superimpose objects gaussian functions on image (starting at 0)
            s = obj.diameter/2
            image = image + i*np.exp(-((X-x)**2+(Y-y)**2)/(2*s**2))
#             define bbox width/height as 2 s.d. from centre
            bx = 2*s
            by = 2*s
            bboxes.append(np.array([[x-bx,y-by],[x+bx,y+by]]))
            labels.append(obj.label)
            
#   set the max here, otherwise will be superceded due to overlap later 
    image_max = image.max()
    if impurity_objects is not None:
        for obj in impurity_objects:
            x = obj.x
            y = obj.y
            i = obj.theta

            if obj.label == 'Spot_impurity':
    #             superimpose objects gaussian functions on image (starting at 0)
                s = obj.diameter/2
                image = image + i*np.exp(-((X-x)**2+(Y-y)**2)/(2*s**2))
                
            elif obj.label == 'Rectangle_impurity':
                l = obj.length
                w = obj.width
                angle = np.random.uniform(0, 2*pi)
                im = np.zeros([image_w, image_h])
                im[int(image_w/2-w/2):int(-image_w/2+w/2), int(image_h/2-l/2):int(-image_h/2+l/2)] = 1
                im = ndimage.rotate(im, np.degrees(angle), reshape=False, mode='constant')
                im = ndimage.shift(im, (y-int(image_h/2)+0.5, x-int(image_w/2)+0.5))
                im = ndimage.gaussian_filter(im, 1)
                im /= im.max()
                image = image + i*im

    # clip overlaps to original max
    image = np.clip(image, 0, image_max)
    # set to max of 1
    image = image/image.max()
    # apply noise (ratio wrt image max)
    noise = np.abs(np.random.randn(image_w, image_h))
    if isinstance(snr_range, list):
        snr = np.random.uniform(snr_range[0], snr_range[1])             
    else:
        snr = snr_range
#     add with the snr
    image = snr*image + noise                
    return (bboxes, labels, image) 


def exportConfig(file, nimages, image_w, image_h, image_d, 
                 label_list, diameter_mean, diameter_std, density_range, min_distance, max_distance, particle_dist_4, particle_dist_3, particle_dist_2,
                 luminosity_range, snr_range, offset,
                 impurity_type, impurity_density = None, impurity_size_mean = None, impurity_size_std = None):
    config = configparser.ConfigParser()
    config.optionxform = lambda option: option  # preserve case for letters

    config.add_section('Section1')
    config.set('Section1', 'nimages', str(nimages))
    config.set('Section1', 'image_w', str(image_w))
    config.set('Section1', 'image_h', str(image_h))
    config.set('Section1', 'image_d', str(image_d))
    config.add_section('Section2')
    config.set('Section2', 'label_list', str(label_list))
    config.set('Section2', 'diameter_mean', str(diameter_mean))
    config.set('Section2', 'diameter_std', str(diameter_std))
    config.set('Section2', 'density_range', str(density_range))
    config.set('Section2', 'min_distance', str(min_distance))
    config.set('Section2', 'max_distance', str(max_distance))
    config.set('Section2', 'rate_particle_less_4_std', str(particle_dist_4))
    config.set('Section2', 'rate_particle_less_3_std', str(particle_dist_3))
    config.set('Section2', 'rate_particle_less_2_std', str(particle_dist_2))
    config.add_section('Section3')
    config.set('Section3', 'snr_range', str(snr_range))
    config.set('Section3', 'luminosity_range', str(luminosity_range))
    config.set('Section3', 'offset', str(offset))
    config.add_section('Section4')
    config.set('Section4', 'impurity_type', str(impurity_type))
    if impurity_type != "None":
        config.set('Section4', 'impurity_density', str(impurity_density))
        config.set('Section4', 'impurity_size_mean', str(impurity_size_mean))
        config.set('Section4', 'impurity_size_std', str(impurity_size_std))

    with open(file, 'w') as configfile:    # save
        config.write(configfile)


def convert_xml_to_txt(xml_file, txt_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    tree = ET.parse(xml_file)
    root = tree.getroot()

    with open(txt_file, 'w') as f:
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            width = xmax - xmin
            height = ymax - ymin
            x_centre = (xmin + width / 2) / 640
            y_centre = (ymin + height / 2) / 640

            line = f"0 {x_centre:.6f} {y_centre:.6f} {width/640:.6f} {height/640:.6f}\n"
            f.write(line)



def convert_xml_files_in_directory(xml_directory, txt_directory):
    if not os.path.exists(txt_directory):
        os.mkdir(txt_directory)
    for file_name in os.listdir(xml_directory):
        if file_name.endswith('.xml'):
            xml_file = os.path.join(xml_directory, file_name)
            base_name = os.path.splitext(file_name)[0]
            txt_file = os.path.join(txt_directory, f"{base_name}.txt")
            convert_xml_to_txt(xml_file, txt_file)


def less_than_rate(particle_distances, threshold):

    number_interactions = len(particle_distances)
    count = len([x for x in particle_distances if x < threshold])
    return count / number_interactions   

def create_images_wrapper(subdir,
                        nimages,
                        folders,
                        # image parameters
                        image_w,
                        image_h,
                        image_d,
                        label_list,
                        snr_range,
                        offset,
                        diameter_mean,
                        diameter_std,
                        luminosity_range,
                        density_range,
                        impurity_type,

                        distance_factor = 1,
                        max_distance_factor = None,

                        circle_impurities = False,
                        label_list_impurity_circle = None,
                        density_range_impurity_circle =  None,
                        diameter_mean_impurity_circle =  None,
                        diameter_std_impurity_circle =  None,
                        rectangle_impurities = False,
                        label_list_impurity_rectangle = None,
                        density_range_impurity_rectangle = None,
                        length_mean_impurity_rectangle = None,
                        length_std_impurity_rectangle = None):
    
    particle_dist_list = []
    #create dirs
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    for i, prefix in enumerate(folders):

        i_dir = subdir + 'images/' + prefix + "/"
        if not os.path.exists(i_dir):
            os.makedirs(i_dir)
        a_dir = subdir + 'temp/' + prefix + "/"
        if not os.path.exists(a_dir):
            os.makedirs(a_dir)
        t_dir = subdir + 'labels/' + prefix + "/"
        if not os.path.exists(t_dir):
            os.makedirs(t_dir)

        # for each image
        for i in tqdm(range(nimages[i])):
            # create particle objects
            if max_distance_factor is None:
                objects, particle_dist_list = create_particles(frames = 1, image_w= image_w, image_h= image_h, image_d= image_d, offset= offset,
                            label_list= label_list, diameter_mean= diameter_mean, diameter_std= diameter_std, density_range = density_range,
                            n_list = None, distance_factor = distance_factor, luminosity_range = luminosity_range, fixed_camera_distance = 10, min_dist_list = particle_dist_list)
            
            else:
                objects, particle_dist_list = create_particles_custom_overlap(frames = 1, image_w= image_w, image_h= image_h, image_d= image_d, offset= offset,
                        label_list= label_list, diameter_mean= diameter_mean, diameter_std= diameter_std, density_range = density_range,
                        n_list = None, distance_factor = distance_factor, max_distance_factor = max_distance_factor, luminosity_range = luminosity_range, fixed_camera_distance = 10, min_dist_list = particle_dist_list)
            objects = objects[0]
            print(objects)
            print(len(objects))

            objects_impurities = None
            if circle_impurities:
                objects_impurities, _ = create_particles(frames = 1, image_w= image_w, image_h= image_h, image_d= image_d, offset= offset,
                                label_list= label_list_impurity_circle, diameter_mean= diameter_mean_impurity_circle, diameter_std= diameter_std_impurity_circle, density_range = density_range_impurity_circle,
                                n_list = None, distance_factor = 1, luminosity_range = [1,1], fixed_camera_distance = 10)
                objects_impurities = objects_impurities[0]
            if rectangle_impurities:
                objects_impurities2 = create_rectangles(frames = 1, image_w= image_w, image_h= image_h, offset= offset,
                                label_list= label_list_impurity_rectangle, length_mean= length_mean_impurity_rectangle, length_std= length_std_impurity_rectangle, density_range = density_range_impurity_rectangle,
                                n_list = None, distance = None)[0]
                if not circle_impurities:
                    objects_impurities = objects_impurities2
            if circle_impurities and rectangle_impurities:
                objects_impurities = np.append(objects_impurities,  objects_impurities2)

            # create image with these objects
            bboxes, labels, image = generateImage(objects, image_w, image_h, snr_range, objects_impurities) 

            # save image 
            fname = i_dir + 'image_{:04d}.jpg'.format(i,2)
            
            if os.path.exists(fname):
                os.remove(fname)
                print(fname, 'deleted')
            plt.imsave(fname, image, cmap='gray')

            # create annotations xml
            writer = Writer(fname, image_w, image_h)
            for bbox, label in zip(bboxes, labels):
                x, y = bbox[0]
                x1, y1 = bbox[1]
                writer.addObject(label, x, y, x1, y1)
            xmlname = a_dir + 'image_{:04d}.xml'.format(i,2)    
            writer.save(xmlname)
        
        # create annotations txt
        convert_xml_files_in_directory(a_dir, t_dir)
    # save run params
    exportConfig(file = subdir + 'info.txt', nimages = nimages, image_w =image_w, image_h =image_h, image_d =image_d, 
                    label_list =label_list, diameter_mean =diameter_mean, diameter_std =diameter_std, density_range =density_range, 
                    min_distance = f'{distance_factor} particle(s)', max_distance = f'{max_distance_factor} particle(s)',
                    particle_dist_4 = less_than_rate(particle_dist_list, diameter_mean*2), particle_dist_3 = less_than_rate(particle_dist_list, diameter_mean*1.5), particle_dist_2 = less_than_rate(particle_dist_list, diameter_mean),
                    luminosity_range =luminosity_range, snr_range =snr_range, offset =offset,
                    impurity_type =impurity_type, impurity_density = {'Circle': density_range_impurity_rectangle, "Rectangle": density_range_impurity_rectangle},
                    impurity_size_mean = {'Circle': diameter_mean_impurity_circle, "Rectangle": length_mean_impurity_rectangle},
                    impurity_size_std = {'Circle': diameter_std_impurity_circle, "Rectangle": length_std_impurity_rectangle})


# def recreate_labels_wrapper(subdir,
#                         nimages,
#                         folders,
#                         # image parameters
#                         image_w,
#                         image_h,
#                         image_d,
#                         label_list,
#                         snr_range,
#                         offset,
#                         diameter_mean,
#                         diameter_std,
#                         luminosity_range,
#                         density_range,
#                         impurity_type,
                        
#                         circle_impurities = False,
#                         label_list_impurity_circle = None,
#                         density_range_impurity_circle =  None,
#                         diameter_mean_impurity_circle =  None,
#                         diameter_std_impurity_circle =  None,
#                         rectangle_impurities = False,
#                         label_list_impurity_rectangle = None,
#                         density_range_impurity_rectangle = None,
#                         length_mean_impurity_rectangle = None,
#                         length_std_impurity_rectangle = None):
    
#     #create dirs
#     if not os.path.exists(subdir):
#         os.makedirs(subdir)
#     for i, prefix in enumerate(folders):

#         i_dir = subdir + 'images/' + prefix + "/"
#         if not os.path.exists(i_dir):
#             os.makedirs(i_dir)
#         a_dir = subdir + 'temp/' + prefix + "/"
#         if not os.path.exists(a_dir):
#             os.makedirs(a_dir)
#         t_dir = subdir + 'labels/' + prefix + "/"
#         if not os.path.exists(t_dir):
#             os.makedirs(t_dir)

#         convert_xml_files_in_directory(a_dir, t_dir)


if __name__ == "__main__":

    # create data as per data_config
    for key, value in train_data_params.items():
        print(key)
        create_images_wrapper(**value)

    for key, value in test_data_params.items():
        print(key)
        create_images_wrapper(**value)

    for key, value in extra_test_data_params.items():
        print(key)
        create_images_wrapper(**value)



