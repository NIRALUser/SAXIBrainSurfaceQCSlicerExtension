import argparse
import math
import os
import pandas as pd
import numpy as np 
import torch
from torch import nn
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import nrrd
import monai

from shapeaxi import saxi_nets, utils
from shapeaxi.saxi_dataset import SaxiFreesurferDataset_Slicer
from shapeaxi.saxi_transforms import EvalTransform, UnitSurfTransform, TrainTransform, RandomRemoveTeethTransform, RandomRotationTransform,ApplyRotationTransform, GaussianNoisePointTransform, NormalizePointTransform, CenterTransform
from shapeaxi.dental_model_seg import segmentation_crown, post_processing
from shapeaxi.colors import bcolors
torch.backends.cudnn.benchmark = False

# This file proposes a prediction with the test data. It calls SaxiDataset which is a custom class from PyTorch that inherits from torch.utils.data.Datset.
# It calls also EvalTransform 


def SaxiFreesurfer_predict(args, mount_point, df, fname, ext, model): 
    model.eval()
    test_ds = SaxiFreesurferDataset_Slicer(df,transform=UnitSurfTransform(),name_class=args.class_column,freesurfer_path=args.fs_path)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers, pin_memory=True)

    with torch.no_grad():
        # The prediction is performed on the test data
        probs = []
        predictions = []
        softmax = nn.Softmax(dim=1)

        for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            # The generated CAM is processed and added to the input surface mesh (surf) as a point data array
            VL, FL, VFL, FFL, VR, FR, VFR, FFR, Y = batch 
            VL = VL.cuda(non_blocking=True,device=args.device)
            FL = FL.cuda(non_blocking=True,device=args.device)
            VFL = VFL.cuda(non_blocking=True,device=args.device)
            FFL = FFL.cuda(non_blocking=True,device=args.device)
            VR = VR.cuda(non_blocking=True,device=args.device)
            FR = FR.cuda(non_blocking=True,device=args.device)
            VFR = VFR.cuda(non_blocking=True,device=args.device)
            FFR = FFR.cuda(non_blocking=True,device=args.device)
            FFL = FFL.squeeze(0)
            FFR = FFR.squeeze(0)

            X = (VL, FL, VFL, FFL, VR, FR, VFR, FFR)
            x = model(X)

            x = softmax(x).detach()
            predictions.append(torch.argmax(x, dim=1, keepdim=True))
            with open(args.log_path,'r+') as log_f :
                    log_f.write(str(idx))

        predictions = torch.cat(predictions).cpu().numpy().squeeze()

        out_dir = os.path.join(args.out)

        df['pred'] = predictions
        out_name = os.path.join(out_dir, fname.replace(ext, "_prediction.csv"))
        df.to_csv(out_name, index=False)

        print(bcolors.SUCCESS, f"Saving results to {out_name}", bcolors.ENDC)



def main(args):
    # Read of the test data from a CSV or Parquet file
    mount_point = args.mount_point
    fname = os.path.basename(args.csv)    
    ext = os.path.splitext(fname)[1]
    path_to_csv = os.path.join(mount_point, args.csv)
    
    if ext == ".csv":
        df = pd.read_csv(path_to_csv)
    else:
        df = pd.read_parquet(path_to_csv)
    
    SAXINETS = getattr(saxi_nets, args.nn)
    model = SAXINETS.load_from_checkpoint(args.model)
    model.to(torch.device(args.device))
    
    if args.nn == "SaxiRing_QC":
        SaxiFreesurfer_predict(args, mount_point, df, fname, ext, model)

    else:
        raise NotImplementedError(f"Neural network {args.nn} is not implemented")             


def get_argparse():
    # This function defines the arguments for the prediction
    parser = argparse.ArgumentParser(description='Saxi prediction')    

    ##Trained
    model_group = parser.add_argument_group('Trained')
    model_group.add_argument('--model', type=str, help='Model for prediction', required=True)
    model_group.add_argument('--nn', type=str, help='Neural network name : SaxiClassification, SaxiRegression, SaxiSegmentation, SaxiIcoClassification, SaxiRing, SaxiRingMT, SaxiRingClassification', required=True, choices=["SaxiClassification", "SaxiRegression", "SaxiSegmentation", "SaxiIcoClassification", "SaxiIcoClassification_fs", 'SaxiRing', 'SaxiRingClassification', 'SaxiRingMT', 'SaxiMHA', 'SaxiRing_QC'])

    ##Input
    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--csv', type=str, help='CSV with column surf', required=True)   
    input_group.add_argument('--surf_column', type=str, help='Surface column name', default='surf')
    input_group.add_argument('--class_column', type=str, help='Class column name', default='class')
    input_group.add_argument('--mount_point', type=str, help='Dataset mount directory', default='./')
    input_group.add_argument('--num_workers', type=int, help='Number of workers for loading', default=4)
    input_group.add_argument('--array_name', type=str, help = 'Predicted ID array name for output vtk', default='PredictedID')
    input_group.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
    input_group.add_argument('--crown_segmentation', type=bool, help='Isolation of each different tooth in a specific vtk file', default=False)
    input_group.add_argument('--fdi', type=int, help='numbering system. 0: universal numbering; 1: FDI world dental Federation notation', default=0)
    input_group.add_argument('--path_ico_left', type=str, default='./3DObject/sphere_f327680_v163842.vtk', help='Path to ico left (default: ../3DObject/sphere_f327680_v163842.vtk)')
    input_group.add_argument('--path_ico_right', type=str, default='./3DObject/sphere_f327680_v163842.vtk', help='Path to ico right (default: ../3DObject/sphere_f327680_v163842.vtk)')
    input_group.add_argument('--fs_path', type=str, help='Path to freesurfer folder', default=None)
    input_group.add_argument('--device', type=str, help='Device for prediction', default='cuda:0')
    input_group.add_argument('--log_path', type=str, help='Path to log file', default='./log.txt')

    ##Hyperparameters
    hyper_group = parser.add_argument_group('Hyperparameters')
    hyper_group.add_argument('--radius', type=float, help='Radius of icosphere', default=1.35)    
    hyper_group.add_argument('--subdivision_level', type=int, help='Subdivision level for icosahedron', default=2)

    ##Gaussian Filter
    gaussian_group = parser.add_argument_group('Gaussian filter')
    gaussian_group.add_argument('--mean', type=float, help='Mean (default: 0)',default=0,)
    gaussian_group.add_argument('--std', type=float, help='Standard deviation (default: 0.005)', default=0.005)

    ##Output
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', type=str, help='Output directory', default="./")

    return parser


if __name__ == '__main__':

    parser = get_argparse()
    args = parser.parse_args()

    main(args)
