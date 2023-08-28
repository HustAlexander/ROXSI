class Defaultconfig(object):
    # class variable,every instance will share it
    data_root1 = '/storage/homefs/zk22e821/Dataset/BratsTrain_2D_2019/HGG'
    data_root2 = '/storage/homefs/zk22e821/Dataset/BratsTrain_2D_2019/LGG'
    data_root20 = '/storage/homefs/zk22e821/Dataset/Brats18_nn_training/'
    batch_size = 16
    use_gpu = True
    max_epoch = 50
    lr = 0.0001
    lr_decay = 0.95
    weight_decay = 0.0001

opt=Defaultconfig