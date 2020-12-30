# Script for running a single sample

##############################################################
# HOW TO DOWNLOAD DATA::::
# Go into the data folder: ./nyplstereo.py shows how to 
# download the data, preprocess, and split into images
##############################################################

##############################################################
# RUN CODE
# Some pointers: 
#   - we can use adversarial training as opposed to just l1 (it's slower
#           but probably a bit better). To do this: there's a flag in the
#           train_boundless.py with use_adv.
#   - The more iterations, the better (but slower)
##############################################################

export BASE_PATH='/scratch/shared/nfs1/ow/datasets/stereo/nypl_large_preprocessed/'

# Set up conda environment and python environment
# This environment is actually saved out
conda activate pytorch3d1.4

im_name=000010

# 1. Do the Preprocessing and DTW
python dynamic_time_warping.py --comparison 'grad' --im_name $im --normalise

# 2. Use the DTW in order to generate images/videos
rm -r temp$im_name/
mkdir ./temp$im_name/
python generate_video.py --im_name $im_name  --normalise
cd ./temp$im_name/$im_name
ffmpeg -r 3 -i im-%03d.png -r 25 warp_$im_name.mp4
cd ../../

# 3. And now train the synthesis part

# The more iterations the better but longer...
N_ITERS_V=8000
N_ITERS_T=8010

# Where to save the model and temporary files
base_file=./temp/
model_name=./checkpoints/

cp $BASE_PATH/imR/$im_name.jpg $base_file/$im_name/orig-0.jpg

python train_boundless.py --n_iters $N_ITERS_T \
                            --image $base_file/$im_name/orig-0.jpg \
                            --mask "$base_file/$im_name/mask-*" \
                            --model_path $model_name/ \
                            --training --batch_size 8

# Then generate images
echo $model_name/checkpoints/model_iter$N_ITERS_V.pth
python train_boundless.py --mask "$base_file/$im_name/mask-*" \
                            --image "$base_file/$im_name/im-*" \
                            --val_path ./temp$im_name/boundless/ \
                            --model_path $model_name/checkpoints/model_iter$N_ITERS_V.pth

# Then generate ffmpeg video of generated videos
cd ./temp$im_name/boundless/
ffmpeg -r 2 -i pred%03d.png -r 25 full_$im_name.mp4
