import struct
import torch
import torchvision


def save_to_weights(model, weights):
    net = torch.load(model).eval()
    net = net.cuda()
    # save alexnet.pth to alexnet.weights
    # weights's format
    # line1: num_layers
    # line2: layer1_name num_weights weight1 weight2 weight3 ...
    # line3: layer2_name num_weights weight1 weight2 weight3 ...
    # ...
    f = open(weights, 'w')
    print("Layers: ", len(net.state_dict().keys()))
    f.write("{}\n".format(len(net.state_dict().keys())))  # 层数量
    for k,v in net.state_dict().items():
        print('key: ', k)
        print('value: ', v.shape)
        vals = v.reshape(-1).cpu().numpy()  # 将参数拉平
        print('reshape: ', vals.shape)
        f.write("{} {}".format(k, len(vals)))  # 记录参数名字以及长度，在rt中用于读取固定长度
        for val in vals:
            f.write(" ")  # 空格作为cpp中读取文件时的截断标志
            f.write(struct.pack(">f", float(val)).hex())
        f.write("\n")
    print("{} to {} done!".format(model, weights))


def main():
    model = "weights/alexnet.pth"
    weights = "weights/alexnet.weights"
    # download and save alexnet.pth
    net = torchvision.models.alexnet(pretrained=True)
    net.eval()
    net = net.cuda()
    print(net)
    tmp = torch.ones(2, 3, 224, 224).float().cuda()
    out = net(tmp)
    print('alexnet out:', out.shape)
    print(out)
    torch.save(net, model)
    # convert to weights
    save_to_weights(model, weights)


if __name__ == "__main__":
    main()
