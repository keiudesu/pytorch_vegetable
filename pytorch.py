import streamlit as st
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision.models import resnet50
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.drop = nn.Dropout2d()

        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.conv3 = nn.Conv2d(16,32,4)

        self.fc1 = nn.Linear(32 * 50 * 50, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.drop(x)
        x = x.view(-1,32*50*50)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def detect(image, model):
    detect_face = cv2.resize(image, (224,224))
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    detect_face = transform(detect_face)

    output = model(detect_face)

    name_label = output.argmax(dim=1, keepdim=True)
    name = label_to_name(name_label)

    return name,output

#ラベルから対応する野菜の名前を返す関数
def label_to_name(name_label):

    if name_label == 0:
        name = "Broccoli"
    elif name_label == 1:
        name = "Cabbage"
    elif name_label == 2:
        name = "Carrot"
    elif name_label == 3:
        name = "Radish"
    elif name_label == 4:
        name = "Tomato"

    return name

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    st.sidebar.write(device)
    st.sidebar.title("野菜の画像認識アプリ")
    st.sidebar.write("画像認識モデルを使って野菜の種類を判定します。")
    st.sidebar.write("（注：ブロッコリー、人参、キャベツ、大根、トマト）")
    st.sidebar.write("判別が可能な種類は以下の通りです。")

    st.sidebar.write("")

    st.sidebar.write("画像のソースを選択してください")
    #サイドバーの表示
    image = st.sidebar.file_uploader("画像をアップロードしてください", type=['jpg','jpeg', 'png'])

    #保存済みのモデルをロード
    model = CNN()
    model.load_state_dict(torch.load("cnn-99.model", map_location=device))
    model.eval()

    #画像ファイルが読み込まれた後，顔認識を実行
    if image != None:
        st.image(image)
        #画像の読み込み
        image = np.array(Image.open(image))
        #画像から検出を行う
        name,output = detect(image, model)
        #検出を行った結果を表示
        st.write(name)

        m = nn.Softmax(dim=1)
        st.write('確率：', torch.max(m(output)).item() * 100)


if __name__ == "__main__":
    main()
