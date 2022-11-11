from typing import Text

from absl import logging
from tfx.orchestration import metadata, pipeline


def init_pipeline(pipeline_root: Text, pipeline_name, metadata_path, components):
    logging.info(f"Pipeline root set to: {pipeline_root}")

    beam_args = [
        "--direct_running_mode=multi_processing",
        "----direct_num_workers=0",
    ]

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path,
        ),
        eam_pipeline_args=beam_args,
    )
