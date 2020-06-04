

##to run the traing model you can use the following two commands.
## Firstly, you should move to the directory with ; cd ImageClassifier

## For train.py run the first command that contains the data directory with flowers.
## For predict.py run the second command that contains the checkpoint file and image tha you want to make prediction

# you can also use them without any parameter.

#python train.py data_directory
python train.py --data_dir ./flowers/

#python predict.py /path/to/image checkpoint
python predict.py --input_img /home/workspace/ImageClassifier/flowers/test/1/image_06752.jpg --checkpoint /home/workspace/ImageClassifier/checkpoint.pth

python train.py --data_dir ./flowers/ --learning_rate 0.01 --hidden_units 512 --epochs 20