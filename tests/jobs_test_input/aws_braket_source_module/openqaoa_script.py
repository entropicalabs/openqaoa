from openqaoa.workflows.managed_jobs import Aws_job
from braket.jobs import save_job_result

def main():
    """
    The entry point is kept clean and simple and all the load statements are hidden in the `aws_jobs_load` function (which will become part of the OpenQAOA library)
    """
   
    job = Aws_job(algorithm='QAOA')
    job.load_input_data()
    job.set_up()
    job.run_workflow()


    save_job_result({"problem_qubo": job.qubo.asdict(),
                     "qaoa_result_evals": job.workflow.results.evals,
                     "qaoa_result_intermediate": job.workflow.results.intermediate,
                     "qaoa_result_optimized": job.workflow.results.optimized,
                     "qaoa_result_most_probable_states": job.workflow.results.most_probable_states})
    

if __name__ == "__main__":
    main()