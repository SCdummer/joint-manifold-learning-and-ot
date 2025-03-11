import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def save_individual_images(file_list, save_dir):
    
    
    img_array_list = [np.array(Image.open(file)) for file in file_list]
    
    for i in range(len(file_list)):
        arr = np.array(Image.open(file_list[i]))
        print(file_list[i], arr.max())
    
    img_array_all_files = np.array(img_array_list)
    max_val = img_array_all_files.max()
    
    print(max_val)
    
    for img_arr, file in zip(img_array_list, file_list):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        fig, ax = plt.subplots()
        max_val = img_arr.max() * 0.0 + 1.0 * 0.7
        plt.imshow(img_arr.squeeze().astype(np.float32) / max_val, vmin=0, vmax=1.0)
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        file_name = 'track' + os.path.basename(file).split('_')[0][5:] + '_' + str(int(os.path.basename(file).split('_')[1][1:4]))
        plt.savefig(os.path.join(save_dir, file_name + '.png'), bbox_inches='tight',
                    pad_inches=0)
        plt.close('all')
    
if __name__ == "__main__":
    save_dir = "/mnt/c/Users/DummerSC/Downloads/gt_tifs_to_pngs"
    # main_dir = "/mnt/c/Users/DummerSC/Downloads/Circle_small_gaussian_large/Circle_small_gaussian_large/images/Track36"
    # files = ["track36_t025.tif", "track36_t026.tif", "track36_t027.tif", "track36_t028.tif", "track36_t029.tif", "track36_t030.tif"]
    track_ids = ['288', '50', '284', '76', '257', '372', '177', '200', '125', '270']
    main_main_dir = "data/Fluo-N2DL-HeLa/01_processed/01_processed/"
    for track_id in track_ids:
        main_dir = os.path.join(main_main_dir, "Track" + track_id)
        files = os.listdir(main_dir)
        files.sort()
        #files = ["track" + track_id + "_t025.tif", "track" + track_id + "_t026.tif", "track" + track_id + "_t027.tif", "track" + track_id + "_t028.tif", "track" + track_id + "_t029.tif", "track" + track_id + "_t030.tif"]
        file_list = [os.path.join(main_dir, file) for file in files]
        save_individual_images(file_list, os.path.join(save_dir, "Track" + track_id))
    # main_dir = "data/Fluo-N2DL-HeLa/01_processed/01_processed/Track270"
    # files = ["track270_t025.tif", "track270_t026.tif", "track270_t027.tif", "track270_t028.tif", "track270_t029.tif", "track270_t030.tif"]
    # file_list = [os.path.join(main_dir, file) for file in files]
    # for file in file_list:
    #     save_individual_images(file, save_dir)