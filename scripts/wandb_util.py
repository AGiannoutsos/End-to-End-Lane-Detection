import wandb
import os
import glob

# Log an artifact, which is a dataset or a model
def wandb_log_artifact(run=None, artifact_name=None, file=None, directory=None, type_=None):
    # Initialize a new W&B run to track this job
    if run is None:
        run = wandb.init(job_type="artifact-creation")

    
    if file is not None:
        artifact = wandb.Artifact(artifact_name, type=type_)
        artifact.add_file(file)
        wandb.log_artifact(artifact)
        if run is None:
            wandb.finish()
        return

    if directory is not None:
        artifact = wandb.Artifact(artifact_name, type=type_)
        dataset_paths = glob.glob("%s/*"%(directory), recursive=False)
        for path in dataset_paths: 
            artifact.add_file(path)
        wandb.log_artifact(artifact)
        if run is None:
            wandb.finish()
        pass


def wandb_load_artifact(run=None, artifact_name=None, version="latest"):
    # Initialize a new W&B run to track this job
    if run is None:
        run = wandb.init(job_type="artifact-loading")

    # Pull down that dataset you logged
    artifact = wandb.use_artifact("%s:%s"%(artifact_name, version))
    artifact_dir = artifact.download()
    if run is None:
        wandb.finish()
    return artifact_dir