# Install requirements
# pip3 install -r requirements.txt

# Download Pre-Trained Model
mkdir -p ./data/pretrained_models

wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
tar -xzf resnet_v1_50_2016_08_28.tar.gz
mv resnet_v1_50.ckpt ./data/pretrained_models

wget http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
tar -xzf resnet_v1_101_2016_08_28.tar.gz
mv resnet_v1_101.ckpt ./data/pretrained_models

wget http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz
tar -xzf resnet_v1_152_2016_08_28.tar.gz
mv resnet_v1_152.ckpt ./data/pretrained_models

rm *.tar.gz