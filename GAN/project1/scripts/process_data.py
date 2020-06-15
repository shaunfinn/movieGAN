 
# arguements [ colour (int) ; delta_t (float) ]

#takes images from root directory, 
#center crop to shortest dimension,
#resizes to img_dim
#and saves to savedir

from config import rawdata_dir, dataset_dir

from PIL import Image  
import os
import cv2
import sys


colour =int(sys.argv[1])    # use 0 for B&W; 1 for RGB
delta_t = float(sys.argv[2])    #time between frame grab for veideo SECONDS

#defaults
img_dim = 128      # desired output size 128x128 pixels
fps = 24        # frames per second
delta_fr = int(delta_t*fps)  
valid_ext= [".jpg", ".png", ".jpeg"]

 #create save_dir. dataset_dir needs a subfolder for imagefolder fuctionality
save_dir= dataset_dir + "images/"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    print("created images subfolder")


def process_img(im,img_dim=img_dim, colour=colour):
           # Opens a image in RGB mode
    width, height = im.size   # Get dimensions
    
    #get shortest length
    new_size = min(width, height) 
    
    #centercrop to square
    left = (width - new_size)/2
    top = (height - new_size)/2
    right = (width + new_size)/2
    bottom = (height + new_size)/2
    im = im.crop((left, top, right, bottom)) 
    
    #greyscale/colour
    if colour == 0:
        im = im.convert("L")   # greyscale
    
    #resize
    im = im.resize((img_dim,img_dim)) 
    
    return im   # PIL Image
    
    #im.save(savedir+ "/"+str(i)+ ext)
  
def process_video(pathStr, video_name, delta_fr=delta_fr, save_dir=save_dir, i_start=0):
    
    curr_frame = 0
    imgID = i_start    # the number of images already produced in "processed" folder
    error_cnt = 0

    cap = cv2.VideoCapture(pathStr)
    #get first frame
    success, frame = cap.read()
    while (success == True):
        try:
            # Capture frame-by-frame
            
            # change from BGR to RGB for PIL image conversion
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(frame)
            
            im = process_img(im)    # returns PIL image
            
            im.save(save_dir + str(imgID) + '.jpg')
            #print("success video ", imgID)
            imgID+=1
            
        except:
            #print("error on frame")
            error_cnt +=1
            
        #advance frames
        curr_frame += delta_fr
        cap.set(1, curr_frame)
        success, frame = cap.read()

    print("video ", video_name, " processed; ",(imgID-i_start), " frames captured, ",error_cnt, " errors" )

    # When everything done, release the capture
    cap.release()
    #cv2.destroyAllWindows()
    
    # pass last imgID so main loop can continue naming convention
    return imgID   

#checks for corrupt files after processing comlplete
def check_for_corrupt_files(dir_path):
    print("Now checking for corrupt files...")
    error_cnt =0
    for subdir, dirs, files in os.walk(dir_path):
        for file in files:
            pathStr = os.path.join(subdir, file)
            fn = os.path.splitext(file)
            try:
                im = Image.open(pathStr)        # Opens a image in RGB mode

            except:
                error_cnt+=1
                os.remove(pathStr)
                print("couldn't open ..", fn, "  ..so removed")

    print("check complete, ", error_cnt, " corrupt files removed")

#counts the files in a directory
def file_count(dir_path):
    return len([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))])


#---------------------------------------------------

#MAIN- loop through all files in rawdata_dir

#initialise counters
existing_files = file_count(dataset_dir)
i= existing_files               # img counter- total number of images processed
                                #used to name processed images "i.jpg"
                                # set to existing number of files in "processed" folder so no overwriting (naming starts with 0.jpg)


error_cnt =0
invalid_cnt=0


print(existing_files, " files already in folder data/processed ")
print("processing started...")
for subdir, dirs, files in os.walk(rawdata_dir):
    for file in files:
        pathStr = os.path.join(subdir, file)
        fn = os.path.splitext(file)[0]
        ext = os.path.splitext(file)[1]
        if (ext in valid_ext):
            try:
                im = Image.open(pathStr) 
                im = process_img(im)    # returns PIL image
                im.save(save_dir + str(i)+ ext)
                #print("success img", i)
                i=i+1
            except:
                print("error on img: ", fn) 
                error_cnt+=1
        elif (ext==".mp4"):
            
            try:
                # new i defined so further images don't overwrite those from the video
                i= process_video(pathStr, video_name= fn, i_start=i)
    
            except:
                print("error on video: ", fn) 
            
        else:
            invalid_cnt+=1
        
print("proccesing complete; ",i-existing_files, " images added ; errors: ", error_cnt, ", invalid files: ", invalid_cnt)       
        
check_for_corrupt_files(save_dir)

