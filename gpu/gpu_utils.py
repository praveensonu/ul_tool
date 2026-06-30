import subprocess


def get_available_gpu_ids() -> list[int]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index",
        "--format=csv,noheader,nounits",
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    return [int(x.strip()) for x in result.stdout.strip().splitlines()]


def validate_gpu_id(gpu_id: int) -> None:
    gpu_ids = get_available_gpu_ids()

    if gpu_id not in gpu_ids:
        raise ValueError(
            f"Invalid gpu_id={gpu_id}. Available GPU ids are: {gpu_ids}"
        )