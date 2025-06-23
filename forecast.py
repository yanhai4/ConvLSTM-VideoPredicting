import cv2
import torch
import numpy as np
from model import MyModel
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

def load_video_frames(video_path, num_input_frames=3):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))  # 可根据模型输入调整
        frames.append(gray)
    cap.release()

    # 转换为张量序列: (T, 1, H, W)
    frames_tensor = torch.stack([ToTensor()(f).unsqueeze(0) for f in frames], dim=0)
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
    frames = frames.cpu().numpy() * 255
    frames = frames.astype(np.uint8)

    h, w = frames.shape[2], frames.shape[3]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), isColor=False)

    for f in frames:
        frame = f[0]  # shape: (H, W)
        out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

    out.release()
    print(f"Saved prediction video to {output_path}")

def extract_training_samples(frames, input_len=3):
    """
    从连续视频帧中生成训练样本：
    输入：frames[t:t+input_len]，标签：frames[t+input_len]
    """
    samples = []
    for i in range(len(frames) - input_len):
        input_seq = frames[i:i + input_len]      # (input_len, 1, H, W)
        target = frames[i + input_len]           # (1, H, W)
        samples.append((torch.stack(input_seq, dim=0), target))
    return samples


def loadDataAndTrain(video_path, pthpath):
    # 假设你用单个视频训练（可扩展为多个视频）
    all_frames = load_video_frames(video_path)  # shape: (T, 1, H, W)

    samples = extract_training_samples(all_frames, input_len=3)
    inputs = torch.stack([s[0] for s in samples])       # (N, T, 1, H, W)
    targets = torch.stack([s[1] for s in samples])      # (N, 1, H, W)

    train_loader = DataLoader(TensorDataset(inputs, targets), batch_size=8, shuffle=True)

    model = MyModel(hidden_dim=[8, 16, 64], length=3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)  # (B, 1, H, W)

            output = model(x_batch)             # (B, 3, 1, H, W)
            pred_next = output[:, -1]           # take the last frame
            loss = F.mse_loss(pred_next, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")

    # save
    torch.save(model.state_dict(), pthpath)


if __name__ == "__main__":
    # change the path here
    video_path="video.mp4"
    pthpath="trained_model.pth"
    
    loadDataAndTrain(video_path=video_path,pthpath=pthpath)

    model = MyModel(hidden_dim=[8, 16, 64], length=3)
    model.load_state_dict(torch.load(pthpath, map_location="cpu")) 
    model.eval()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # 加载视频帧（灰度）
    video_path = "input_video.mp4"  # ← 输入视频路径
    frames = load_video_frames(video_path)

    # 用前3帧预测后10帧
    input_seq = frames[:10]  # (3, 1, H, W)
    predicted_frames = predict_future_frames(model, input_seq, num_future=10)

    # 保存结果视频
    save_frames_to_video(predicted_frames, "predicted_output.mp4")
