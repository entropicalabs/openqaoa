from braket.jobs import save_job_result

import openqaoa
from openqaoa_braket.algorithms import AWSJobs


def main():
    """
    The entry point is kept clean and simple and all the load statements are hidden in the `aws_jobs_load` function (which will become part of the OpenQAOA library)
    """

    job = AWSJobs(algorithm="QAOA")
    job.load_compile_data()
    job.run_workflow()

    save_job_result({"result": job.workflow.asdict()})


if __name__ == "__main__":
    main()
