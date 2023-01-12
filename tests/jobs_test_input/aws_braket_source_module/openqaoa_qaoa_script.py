import json
from braket.jobs import save_job_result
from openqaoa.workflows.managed_jobs import AWSJobs


def main():
    """
    The entry point is kept clean and simple and all the load statements are hidden in the `aws_jobs_load` function (which will become part of the OpenQAOA library)
    """

    job = AWSJobs(algorithm="QAOA")
    job.load_input_data()
    job.set_up()
    job.run_workflow()

    save_job_result({"result": json.loads(job.workflow.results.dumps())})


if __name__ == "__main__":
    main()
