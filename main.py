import model as mm
import  train as tr
if __name__ == '__main__':
    model =mm. ecg_net()
    tr.train(model,10)
