import argparse

import logging
logger = logging.getLogger("logger")
logging.basicConfig(level=logging.INFO)

def build_pipeline(args):
    import os
    import kfp
    from kfp.v2 import compiler, dsl
    import kfp.dsl as dsl
    from kfp.v2.dsl import pipeline
    from google.cloud import aiplatform
    import google_cloud_pipeline_components as gcpc
    from google_cloud_pipeline_components import aiplatform as gcc_aip
    from google_cloud_pipeline_components.experimental import vertex_notification_email
    from datetime import datetime

    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
    bucket = args.bucket
    bucket_name = bucket[5:]

    job_id = f"{args.job_id}-{TIMESTAMP}"
    service_account = args.service_account

    pipeline_root = f"{bucket}/{args.pipeline_root}"

    @pipeline(name=args.pipeline_name, pipeline_root=pipeline_root)
    def pipeline(
        gcs_source: str = args.gcs_source,
        bucket: str = args.bucket,
        project: str = args.project_id,
        job_id: str = job_id,
        region: str = args.region
    ):
        notify_email_task = vertex_notification_email.VertexNotificationEmailOp(
            recipients=args.recipients
        )

        with dsl.ExitHandler(notify_email_task):
            dataset_create_op = gcc_aip.ImageDatasetCreateOp(
                display_name="damaged-car-parts-dataset",
                project=project,
                location=region,
                gcs_source=gcs_source,
                import_schema_uri=aiplatform.schema.dataset.ioformat.image.single_label_classification
            )

            training_op = gcc_aip.AutoMLImageTrainingJobRunOp(
                project=project,
                display_name=f'automl-images-{TIMESTAMP}',
                dataset=dataset_create_op.outputs["dataset"],
                prediction_type="classification",
                model_display_name="automl-car-images-damage-detection",
                budget_milli_node_hours=8000,
            )

            endpoint_op = gcc_aip.EndpointCreateOp(
                project=project,
                location=region,
                display_name="car-images-damage-detection-endpoint",
            )

            gcc_aip.ModelDeployOp(
                model=training_op.outputs["model"],
                endpoint=endpoint_op.outputs["endpoint"],
                automatic_resources_min_replica_count=1,
                automatic_resources_max_replica_count=1,
            )
    
    compiler.Compiler().compile(pipeline_func = pipeline, package_path="automl_vision_pipeline.json")

    pipeline_job = aiplatform.PipelineJob(
        display_name="custom-train-pipeline",
        template_path="automl_vision_pipeline.json",
        job_id=job_id,
        enable_caching=True
    )
    pipeline_job.submit(service_account=service_account)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket',
                        required=True,
                        help='gcs bucket formatted as gs://my-bucket')
    parser.add_argument('--pipeline-root',
                        required=True,
                        help='name of pipeline')
    parser.add_argument('--pipeline-name',
                        required=True,
                        help="name of pipeline")
    parser.add_argument('--project-id',
                        required=True,
                        help="project id")
    parser.add_argument('--region',
                        required=True,
                        help="region. Ex: us-central1")
    parser.add_argument("--gcs-source",
                        required=True,
                        help="source data")
    parser.add_argument("--recipients",nargs='+',
                       required=True,
                       help="email recipients when pipeline exists")
    parser.add_argument("--job-id",
                        required=True,
                        help="job id of your pipeline")
    parser.add_argument("--service-account",
                        required=True,
                        help="service account email to run this pipeline")
    args = parser.parse_args()
    build_pipeline(args)