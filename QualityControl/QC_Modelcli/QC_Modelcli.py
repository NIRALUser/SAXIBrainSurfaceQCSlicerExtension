#!/usr/bin/env python-real
import argparse
import os
import json
import urllib.request
import subprocess
import sys
import pandas as pd

surfaces = ['white.vtk', 'sphere.reg.vtk', 'thickness', 'curv', 'sulc', 'area']
hemisphere = ['lh', 'rh']
input_csv = os.path.join(os.path.dirname(__file__), "input.csv")
out_model_path = os.path.join(os.path.dirname(__file__), "QC_MODEL.ckpt")

def check_surfaces(args):
    """
    Check if the surfaces files are present in the input directory

    Args:
        args: Arguments from the command line
    """
    subjects_dir = args.input_dir

    for subject in os.listdir(subjects_dir):
        subject_path = os.path.join(subjects_dir, subject)
        if not os.path.isdir(subject_path) or len(os.listdir(subject_path)) == 0:
            continue

        for session in os.listdir(subject_path):
            missing_files = []
            session_path = os.path.join(subject_path, session)
            if not os.path.isdir(session_path) or len(os.listdir(session_path)) == 0:
                continue

            for hemi in hemisphere:
                for surf in surfaces:
                    surf_file = f"{hemi}.{surf}"
                    surf_path = os.path.join(session_path, 'surf', surf_file)
                    if not os.path.exists(surf_path):
                        missing_files.append(surf_path)

            if missing_files:
                print(f"Missing files: {subject} - {session}")
            else:
                csv_edit(args, subject, session)


def csv_edit(args, subject, session):
    """
    Edit the csv file with the subject and session

    Args:
        args: Arguments from the command line
        subject: Subject ID
        session: Session ID
    """
    session_path = os.path.join(args.input_dir, subject, session)
    with open(input_csv, 'a') as f:
        f.write(f"{session_path},1.0\n")

def download_model(args, output_path):
    json_path = os.path.join(os.path.dirname(__file__), "QC_model_path.json")
    with open(json_path, 'r') as file:
        model_info = json.load(file)
    model_url = model_info["model"]["url"]
    urllib.request.urlretrieve(model_url, output_path)

def run_prediction(args):
    python_path = sys.executable
    os.chdir(os.path.join(os.path.dirname(__file__), 'ShapeAXI'))
    subprocess.run([python_path, '-m', 'shapeaxi.saxi_predict', '--nn', 'SaxiRing_QC', '--class_column', 'class', '--csv', input_csv, '--model', out_model_path, '--fs_path', args.input_dir, '--log_path', args.log_path, '--out', args.out])


def main(args):
    with open(args.log_path,'w') as log_f:
        # clear log file
        log_f.truncate(0)
    # Check if the input csv file exists
    if not os.path.exists(input_csv):
        print("Creating input csv file...")
        with open(input_csv, 'w') as f:
            f.write("path,class\n")
        check_surfaces(args)

    if not os.path.exists(out_model_path):
        print("Downloading model...")
        download_model(args, out_model_path)

    run_prediction(args)

    df = pd.read_csv(os.path.join(args.out, "input_prediction.csv"))
    df = df.drop(columns=['class'])
    df.to_csv(os.path.join(args.out, 'input_prediction.csv'), index=False)


def get_argparse():
    parser = argparse.ArgumentParser(description='Quality Control Model')
    parser.add_argument('input_dir', type=str, help='Path to the input directory with all the subjects')
    parser.add_argument('out', type=str, help='Path to the output directory')
    parser.add_argument('log_path',type=str)
    return parser

if __name__ == '__main__':
    parser = get_argparse()
    args = parser.parse_args()
    main(args)
