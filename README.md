
# using https://github.com/Naman-ntc/Pytorch-Human-Pose-Estimation as base

# description
I have extended the stacked hourglass model
The code for this can be found at poseProject/Pytorch-Human-Pose-Estimation-master/models/StackedHourGlass.py and 
at poseProject/Pytorch-Human-Pose-Estimation-master/models/modules/StackedHourGlass.py

I modified the builder to be able to use Adam and AdamW. This can be found in poseProject/Pytorch-Human-Pose-Estimation-master/builder.py on lines 52 to 57

If you would like to see my best performing pre-trained model, it can be found here: https://drive.google.com/file/d/1-0auFlkX29ODevSFlB1NOEc9EmgmGlpV/view?usp=sharing

If you would like to see my google colab notebook I used to train the model it can be found here: https://colab.research.google.com/drive/1Db8i-hnOtSmmwY3ZH7knsOWoN52Ip6PM?usp=sharing

if you would like to train/test the model in colab, please make sure the google colab file structure looks like the screenshot called Networks.PNG
