import os
import time
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import numpy as np
import netCDF4 as nc

from torch.utils.data import DataLoader
from progress.spinner import MoonSpinner
from loguru import logger

from data.preprocess import get_scaler
from data.dataset import TrainingSet, ForecastingSet
from data.postprocess import get_climatology, get_land, plot_helper, index_helper
from model import MyModel

os.environ["LOGURU_INFO_COLOR"] = "<green>"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

device = "cuda:0" if torch.cuda.is_available() else "cpu"
logger.add(f"./train.log", enqueue=True)
raw_path = os.path.join(r'./file/', 'data1.nc')
save_path = os.path.join(r'./file/', 'model.pth')
data_path = os.path.join(r'./file/', 'data.npz')
ncfile = nc.Dataset(raw_path, mode='r')
lons = np.array(ncfile.variables['longitude'][:])
lats = np.array(ncfile.variables['latitude'][:])
climatology_scope=[1440, 1800]
epoch_num = 10000
batch_size = 8
length = 3
scaler = get_scaler(raw_path)
climatology = get_climatology(raw_path, scope=climatology_scope)
land = get_land(raw_path)

if __name__=='__main__':
    train_loader = torch.utils.data.DataLoader(dataset=TrainingSet(data_path, length), batch_size=batch_size, shuffle=True)
    forecast_loader = torch.utils.data.DataLoader(dataset=ForecastingSet(data_path, length), batch_size=1, shuffle=False)
    model = MyModel(hidden_dim=[8, 16, 64], length=length).to(device)

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        template = ("load model weights from: {}.")
        logger.info(template.format(save_path))

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.8)
    criterion = nn.MSELoss().to(device)

    for e in range(epoch_num):
        for step, (input, output) in enumerate(train_loader):

            optimizer.zero_grad()

            input = input.to(device).to(torch.float32).unsqueeze(2)
            output = output.to(device).to(torch.float32)

            model.train()
            pred = model(input).squeeze(2)
            loss = criterion(pred, output)

            template = ("epoch {} - step {}: loss is {:1.5f}.")
            logger.info(template.format(e, step, loss))

            loss.backward()
            optimizer.step()
            scheduler.step()
            del pred, loss
            torch.cuda.empty_cache()

            if e == 0:
                break

        spinner = MoonSpinner('Testing ')
        # test....
        for step, (input, output) in enumerate(forecast_loader):
            input = input.to(device).to(torch.float32).unsqueeze(2)

            model.eval()
            pred = model(input).squeeze(2).detach().numpy()
            shape = input[0, 0, 0, ...].shape
            for i in range(length):
                month = (ncfile['sst'].shape[0] - length + i) % 12
                result = np.reshape(scaler.inverse_transform(np.reshape(pred[0, i], (1, -1))), shape)
                plot_helper(result, lons, lats, climatology=climatology[month], land=land, save=True, filename=f'epoch{e}_pred{month+1}.png')
                nino3, nino4, nino34 = index_helper(result, climatology=climatology[month])
                template = ("epoch {} - forecast month {}: nino3 index is {:1.5f}, nino4 index is {:1.5f}, nino3.4 index is {:1.5f}.")
                logger.info(template.format(e, month + 1, nino3, nino4, nino34))
            spinner.next()
        spinner.finish()

        template = ("-----------epoch {} finish!-----------")
        logger.info(template.format(e))

        torch.save(model.state_dict(), save_path)
        print('Model saved successfully:', save_path)
