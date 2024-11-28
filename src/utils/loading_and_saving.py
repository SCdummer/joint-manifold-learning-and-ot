import os
import json
import torch
import warnings

def create_code_snapshot(root, dst_path, extensions=(".py", ".json"), exclude=()):
    """Creates tarball with the source code"""
    import tarfile
    from pathlib import Path

    with tarfile.open(str(dst_path), "w:gz") as tar:
        for path in Path(root).rglob("*"):
            if '.git' in path.parts:
                continue
            exclude_flag = False
            if len(exclude) > 0:
                for k in exclude:
                    if k in path.parts:
                        exclude_flag = True
            if exclude_flag:
                continue
            if path.suffix.lower() in extensions:
                tar.add(path.as_posix(), arcname=path.relative_to(root).as_posix(), recursive=True)

def load_experiment_specifications(experiment_directory):

    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    filename = os.path.join(base_dir, experiment_directory, 'specs.json')

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            + '"specs.json"'.format(experiment_directory)
        )

    return json.load(open(filename))


def get_model_params_dir(experiment_dir, model_name, static_or_dynamic, create_if_nonexistent=False):

    subdir = "static" if static_or_dynamic == 'static' else "dynamic"
    dir = os.path.join(experiment_dir, "ModelParameters", model_name, subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir

def get_optimizer_params_dir(experiment_dir, static_or_dynamic='static', create_if_nonexistent=False):

    subdir = "static" if static_or_dynamic == 'static' else "dynamic"
    dir = os.path.join(experiment_dir, "OptimizerParameters", subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir

def save_model(experiment_directory, filename, model, model_name, static_or_dynamic, epoch):

    model_params_dir = get_model_params_dir(experiment_directory, model_name, static_or_dynamic,True)

    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict()},
        os.path.join(model_params_dir, filename),
    )

def save_optimizer(experiment_directory, filename, optimizer, static_or_dynamic, epoch):

    optimizer_params_dir = get_optimizer_params_dir(experiment_directory, static_or_dynamic,True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )

def save_model_and_optimizer(experiment_directory, static_or_dynamic, epoch, encoder, decoder, time_warper, optimizer, filename='latest.pth'):
    save_model(experiment_directory, filename, decoder, 'decoder', static_or_dynamic, epoch)
    save_model(experiment_directory, filename, encoder, 'encoder', static_or_dynamic, epoch)
    if static_or_dynamic == 'dynamic':
        save_model(experiment_directory, filename, time_warper, 'time_warper', 'dynamic', epoch)
    save_optimizer(experiment_directory, filename, optimizer, static_or_dynamic, epoch)

def load_model(experiment_directory, model, model_name, static_or_dynamic, checkpoint_filename):

    # Get the base directory
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")

    # Get the name of the file which is used to load the parameters
    substring = "static" if static_or_dynamic == 'static' else "dynamic"
    filename = os.path.join(base_dir, experiment_directory, "ModelParameters", model_name, substring, checkpoint_filename)

    if not os.path.isfile(filename):
        warnings.warn('model state dict "{}" does not exist'.format(filename))
        return 0

    # Load the parameters
    data = torch.load(filename)

    # Put the parameters into the model
    model.load_state_dict(data["model_state_dict"])

    # Return the epoch at which the model was saved
    return data["epoch"]

def load_optimizer(experiment_directory, optimizer, static_or_dynamic, checkpoint_filename, device):

    # Get the base directory
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")

    # Get the filename of the file containing the optimizer parameters
    substring = "static" if static_or_dynamic == 'static' else "dynamic"
    filename = os.path.join(base_dir, experiment_directory, "OptimizerParameters", substring, checkpoint_filename)

    if not os.path.isfile(filename):
        warnings.warn('model state dict "{}" does not exist'.format(filename))
        return 0

    # Load the parameters
    data = torch.load(filename, map_location=device, weights_only=True)

    # Put the parameters into the optimizer
    optimizer.load_state_dict(data["optimizer_state_dict"])

    # Return the epoch at which the optimizer was saved
    return data["epoch"]
