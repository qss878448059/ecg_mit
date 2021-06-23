#定义训练过程
import paddle.optimizer as op
import time
import paddle
import paddle.nn.functional as f
import logging
import data_load as da
dir_data = 'Data/'
_size = (128,128)
_n_classes = 8
logging.basicConfig(filename='ecg_train.log',format='%(asctime)s - %(message)s',level=logging.INFO)
logger = logging.getLogger(__name__)
def train(model,epoch_num):
    start = time.time()
    print('数据加载...')
    train1,validation = da.load_files(dir_data)
    train_data = da.load_data(dir_data, train1,_size,batch_size=128)
    val_data = da.load_data(dir_data, validation,_size,batch_size=128)
    print('数据加载完成，用时%d秒'%(time.time()-start))
    use_gpu = True
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
    # loss_func = paddle.nn.functional.cross_entropy()
    print('模型开始训练...')
    model.train()
    opt = op.Adam(learning_rate = 0.001, parameters = model.parameters())
    for epoch in range(epoch_num):
        for batch_id ,data in enumerate(train_data()):
            x_data,y_data = data
            imgs = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            logits = model(imgs)
            # loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, y_data)
            # print(logits)
            loss = paddle.nn.functional.cross_entropy(logits,label)
            avg_loss = paddle.mean(loss)
            if batch_id % 100 == 0:
                logger.info('epch:{},batch_id:{},loss:{}'.format(epoch,batch_id,float(avg_loss.numpy())))
                print('epch:{},batch_id:{},loss:{}'.format(epoch,batch_id,float(avg_loss.numpy())))
            avg_loss.backward()
            opt.step()
            opt.clear_grad()
        model.eval()
        accuracies = []
        losses = []
        for batch_id,data in enumerate(val_data()):
            x_data,y_data = data
            y_data1 = y_data.reshape(-1,1)
            imgs = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            label1 = paddle.to_tensor(y_data1)
            logits = model(imgs)
            pred = f.softmax(logits)
            loss = f.cross_entropy(logits,label)
            acc = paddle.metric.accuracy(pred,label1)
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())
        logger.info("[validation] accuracy/loss: {}/{}".format(np.mean(accuracies), np.mean(losses)))
        print("[validation] accuracy/loss: {}/{}".format(np.mean(accuracies), np.mean(losses)))
        model.train()
    #保存模型
    paddle.save(model.state_dict(), 'ecg.pdparams')