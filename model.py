import paddle
from paddle.nn import Conv2D, MaxPool2D, BatchNorm, Linear,Dropout
import paddle.nn.functional as F
class ecg_net(paddle.nn.Layer):
    def __init__(self , num_classes=8):
        super(ecg_net,self).__init__()
        self.conv1 = Conv2D(in_channels = 1,out_channels = 64,kernel_size =3,stride = 1 ,padding =1 )
        self.bn1 = BatchNorm(64)
        self.conv2 = Conv2D(in_channels= 64,out_channels = 64,kernel_size = 3,stride = 1,padding = 1)
        self.bn2 = BatchNorm(64)
        self.pool1 = MaxPool2D(kernel_size=2, stride=2)
        self.conv3 = Conv2D(in_channels = 64,out_channels = 128,kernel_size =3,stride = 1,padding=1)
        self.bn3 = BatchNorm(128 )
        self.conv4 = Conv2D(in_channels = 128, out_channels =128,kernel_size =3,stride = 1,padding =1 )
        self.bn4 = BatchNorm(128)
        self.pool2 = MaxPool2D(kernel_size=2, stride=2)
        self.conv5 =  Conv2D(in_channels = 128, out_channels =256,kernel_size =3,stride = 1,padding =1 )
        self.bn5 = BatchNorm(256)
        self.conv6 =  Conv2D(in_channels = 256, out_channels =256,kernel_size =3,stride = 1,padding =1 )
        self.bn6 = BatchNorm(256)
        self.pool3 = MaxPool2D(kernel_size=2, stride=2)
        self.fc1 = Linear(in_features= 16*16*256 ,out_features = 2048)
        self.bn7 = BatchNorm(2048)
        self.drop_ratio = 0.5
        self.dropout1 = Dropout(self.drop_ratio)
        self.fc2 = Linear(in_features= 2048 ,out_features= num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x= F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x= F.relu(x)
        x = self.bn2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x= F.relu(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x= F.relu(x)
        x = self.bn4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x= F.relu(x)
        x = self.bn5(x)
        x = self.conv6(x)
        x= F.relu(x)
        x = self.bn6(x)
        x = self.pool3(x)
        x = paddle.reshape(x,[-1,16 * 16 * 256])
        x = self.fc1(x)
        x= F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
# net = ecg_net()
# paddle.summary(net,(20,1,128,128))