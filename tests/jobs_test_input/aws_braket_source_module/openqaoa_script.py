from openqaoa.workflows.managed_jobs import Aws_job

def main():
    """
    The entry point is kept clean and simple and all the load statements are hidden in the `aws_jobs_load` function (which will become part of the OpenQAOA library)
    """
   
    job = Aws_job(algorithm='QAOA')
    job.load_hyperparams()

    job.set_up()
    job.run_workflow()
    

if __name__ == "__main__":
    main()