import cv2
import argparse
from utils import *
from  NetWorks import PReNet
import time
import paddle
from cacs_lwfx import compute

parser = argparse.ArgumentParser(description="PReNet_Test")
parser.add_argument("--logdir", type=str, default="", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="datasets/Rain100H/", help='path to training data')
parser.add_argument("--save_path", type=str, default="results/PReNet", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
parser.add_argument("--pretrained", type=str, default='')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def main():

    os.makedirs(opt.save_path, exist_ok=True)

    # Build model
    print('Loading model ...\n')
    model_paddle = PReNet(opt.recurrent_iter, opt.use_GPU)
    print_network(model_paddle)

    model_paddle.load_dict(paddle.load(opt.logdir))
    model_paddle.eval()

    time_test = 0
    count = 0

    for img_name in os.listdir(opt.data_path+"/rainy"):
        if is_image(img_name):
            img_path = os.path.join(opt.data_path+"/rainy", img_name)

            # input image
            y = cv2.imread(img_path)
            b, g, r = cv2.split(y)
            y = cv2.merge([r, g, b])

            y = normalize(np.float32(y))
            y = np.expand_dims(y.transpose(2, 0, 1), 0)

            y  = paddle.to_tensor(y)
            with paddle.no_grad(): #
                start_time = time.time()

                out, _ = model_paddle(y)
                out= paddle.clip(out, 0., 1.)

                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time

                print(img_name, ': ', dur_time)

            if opt.use_GPU:
                save_out = np.uint8(255 * out.detach().cpu().numpy().squeeze())   #back to cpu
            else:
                save_out = np.uint8(255 * out.data.numpy().squeeze())

            save_out = save_out.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out= cv2.merge([r, g, b])

            cv2.imwrite(os.path.join(opt.save_path, img_name), save_out)
            count += 1
    print('Avg. time:', time_test/count)
    compute(opt.save_path,opt.data_path)
if __name__ == "__main__":
    main()



