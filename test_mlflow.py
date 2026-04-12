"""Quick smoke test — uploads a run to MLflow and prints the result."""

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://localhost:5001"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

print(f"Connecting to {TRACKING_URI} ...")

# Create or reuse experiment
exp_name = "adk/smoke-test"
exp = client.get_experiment_by_name(exp_name)
if exp is None:
    exp_id = client.create_experiment(exp_name)
    print(f"Created experiment: {exp_name} (id={exp_id})")
else:
    exp_id = exp.experiment_id
    print(f"Using existing experiment: {exp_name} (id={exp_id})")

# Create a run and log something
run = client.create_run(experiment_id=exp_id, run_name="smoke-test-run")
run_id = run.info.run_id
print(f"Created run: {run_id}")

client.log_param(run_id, "agent_name", "smoke_test")
client.log_metric(run_id, "latency_ms", 123.4)
client.set_tag(run_id, "status", "success")
client.set_terminated(run_id, status="FINISHED")

print(f"Done — check http://localhost:5000/#/experiments/{exp_id}/runs/{run_id}")
