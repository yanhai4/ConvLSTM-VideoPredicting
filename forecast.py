import cv2
import torch
import numpy as np
from model import MyModel
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt

"""
This is a demo of convLSTM + CBAM to do a simple video predicting.
Use the train function and then evaluate to see the result.

用 convLSTM+CBAM 做视频预测的小demo
训练用train，然后预测用evaluate
因为这是个黑白的视频所以我直接用的F.binary_cross_entropy_with_logits，如果是其他任务这里需要改，model那里也可以加sigmoid
"""

video_path="video.mp4"
loss_path = "loss_curve.png"
pthpath="trained_model.pth"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
batchSize = 8
epochs = 10000 
learning_rate=0.001 
length = 1 # only predict one frame after


def load_video_frames(video_path, num_input_frames=3):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))  # the size of image can be changed hereee 
        gray = gray.astype(np.float32) / 255.0
        frames.append(gray)
    cap.release()

    # (T, 1, H, W)
    frames_tensor = torch.stack([ToTensor()(f) for f in frames], dim=0)
    return frames_tensor  # shape: (T, 1, H, W)


@torch.no_grad()
def predict_future_frames(model, input_seq, num_future=10):

    model.eval()
    device = next(model.parameters()).device
    predicted = []

    current_input = input_seq.unsqueeze(0).to(device)  # shape: (1, T, 1, H, W)

    for _ in range(num_future):
        output = model(current_input)             # shape: (1, T, 1, H, W)
        next_frame = output[:, -1:]               # shape: (1, 1, 1, H, W)
        predicted.append(next_frame.squeeze(0))   # shape: (1, H, W)
        current_input = torch.cat([current_input[:, 1:], next_frame], dim=1)

    return torch.cat(predicted, dim=0)  # shape: (T_future, 1, H, W)


def save_frames_to_video(frames, output_path, fps=10):

    frames = frames.cpu().numpy()
    frames = np.clip(frames * 255, 0, 255).astype(np.uint8)

    h, w = frames.shape[2], frames.shape[3]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), isColor=False)

    for f in frames:
        frame = f[0]  # shape: (H, W)
        out.write(frame)

    out.release()
    print(f"Saved prediction video to {output_path}")

def extract_training_samples(frames, input_len=3):

    samples = []
    for i in range(len(frames) - input_len):
        input_seq = frames[i:i + input_len]      # (input_len, 1, H, W)
        target = frames[i + input_len]           # (1, H, W)
        samples.append((input_seq, target))
    return samples


def loadDataAndTrain(all_frames, loss_path, pthpath):

    samples = extract_training_samples(all_frames, input_len=3)
    inputs = torch.stack([s[0] for s in samples])       # (N, T, 1, H, W)
    targets = torch.stack([s[1] for s in samples])      # (N, 1, H, W)

    train_loader = DataLoader(TensorDataset(inputs, targets), batch_size=batchSize, shuffle=True)

    model = MyModel(hidden_dim=[8, 16, 64], length=length).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.8)
    #criterion = F.MSELoss().to(device)

    loss_list=[]
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device) # (B, 1, H, W) B is 1 in  here

            output = model(x_batch)             # (B, 3, 1, H, W)
            pred_next = output[:, -1]           # take the last frame
            loss = F.binary_cross_entropy_with_logits(pred_next, y_batch)#F.mse_loss(pred_next, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            loss_list.append(avg_loss)  
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")



    plt.plot(loss_list, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_path)
    # save
    torch.save(model.state_dict(), pthpath)

def train(video_path,loss_path,pthpath):
    all_frames = load_video_frames(video_path)  # shape: (T, 1, H, W)
    loadDataAndTrain(all_frames,loss_path,pthpath=pthpath)

def evaluate(video_path,pthpath):
    model = MyModel(hidden_dim=[8, 16, 64], length=length)
    model.load_state_dict(torch.load(pthpath, map_location="cpu")) 
    model.eval()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    frames = load_video_frames(video_path)

    input_seq = frames[:10]  # (3, 1, H, W)
    predicted_frames = predict_future_frames(model, input_seq, num_future=10)

    save_frames_to_video(predicted_frames, "predicted_output.mp4")

if __name__ == "__main__":
    train(video_path=video_path,loss_path=loss_path,pthpath=pthpath)
    evaluate(video_path=video_path,pthpath=pthpath)



    
