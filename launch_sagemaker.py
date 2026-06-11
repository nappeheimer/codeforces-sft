import sagemaker
import boto3
from sagemaker.session import Session
from sagemaker.huggingface import HuggingFace

def launch_training():
    # ==========================================
    # USER CONFIGURATION - EDIT THESE VALUES
    # ==========================================
    role = "<role>"
    bucket = "cf-generation-us-east-1"


    # Assuming your dataset files are inside a specific prefix
    s3_training_data_uri = f"s3://{bucket}/olympic_coder/data/"

    checkpoint_s3_uri = f"s3://{bucket}/olympic_coder/checkpoints_cosine_test_e2_v3/"

    # Final model will be saved here
    output_path = f"s3://{bucket}/olympic_coder/output_cosine_test_e2_v3/"

    # Toggle this to True for rapid debugging
    DRY_RUN = False
    # ==========================================

    print("Setting up SageMaker HuggingFace Estimator...")

    # Set up hyper-parameters
    hyperparameters = {
        "curriculum_file": "train_curriculum.jsonl",
        "curriculum_epochs": 1,
        "total_epochs": 10,
        "learning_rate": 4e-5,
        "warmup_ratio": 0.03,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "max_length": 24576,
        "lr_scheduler_type": "cosine",
        "logging_steps": 1,
        "deepspeed": "ds_config_zero2.json",
        "dry_run": DRY_RUN,
    }


    # Force the region to us-east-1 since the user's quota is there
    boto_session = boto3.Session(region_name="us-east-1")
    sagemaker_session = Session(boto_session=boto_session)

    # Create the HuggingFace estimator
    huggingface_estimator = HuggingFace(
        entry_point="train.py",
        source_dir=".",  # Uploads current directory
        role=role,
        image_uri="<path>",
        instance_type="ml.p5en.48xlarge",  # 8x H200 (141GB)
        instance_count=1,
        output_path=output_path,
        base_job_name="qwen-cosine-test-e2",
        checkpoint_s3_uri=checkpoint_s3_uri,
        hyperparameters=hyperparameters,
        py_version="py311",
        pytorch_version="2.1.0",
        transformers_version="4.36.0",
        # 'torch_distributed' enables torchrun across all GPUs automatically
        distribution={"torch_distributed": {"enabled": True}},
        # SageMaker volume sizes: since 32k context and 7B model require heavy I/O
        volume_size=500,  # GB
        sagemaker_session=sagemaker_session,
        
        # Reverted back to Spot instances because On-Demand quota is pending
        use_spot_instances=True,
        max_run=60000,
        max_wait=72000,  # Failsafe: Wait up to 20 hours for a new spot instance if interrupted
    )

    print("Launching SageMaker Training Job...")
    print(f"Data Source: {s3_training_data_uri}")
    print(f"Dry Run: {DRY_RUN}")

    # Start the training job
    # This automatically downloads the data from S3 to /opt/ml/input/data/train
    huggingface_estimator.fit(
        inputs={"train": s3_training_data_uri},
        wait=False,  # Set to True to block and stream logs to terminal, False returns immediately
    )

    print("\nJob launched successfully!")
    print(f"You can monitor the training progress in the AWS SageMaker Console.")
    print(f"Job Name: {huggingface_estimator.latest_training_job.job_name}")


if __name__ == "__main__":
    launch_training()
