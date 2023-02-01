#   Copyright 2022 Entropica Labs
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import os
import pytest
import unittest
import networkx as nw
from braket.jobs.local import LocalQuantumJob

from openqaoa.problems import MinimumVertexCover
from openqaoa.workflows.managed_jobs import AWSJobs
from openqaoa.workflows.optimizer import QAOA, RQAOA
from openqaoa.devices import create_device


class TestingAwsJobs(unittest.TestCase):
    def setUp(self):
        # the input data directory opt/braket/input/data
        os.environ["AMZN_BRAKET_INPUT_DIR"] = "./tests/jobs_test_input/"
        # the output directory opt/braket/model to write ob results to
        os.environ["AMZN_BRAKET_JOB_RESULTS_DIR"] = "/oq_release_tests/testing_jobs/"
        # the name of the job
        os.environ["AMZN_BRAKET_JOB_NAME"] = "oq_release_test"
        # the checkpoint directory
        os.environ["AMZN_BRAKET_CHECKPOINT_DIR"] = "oq_test_suite_checkpoint"
        # the hyperparameter
        os.environ["AMZN_BRAKET_HP_FILE"] = ""
        # the device ARN (AWS Resource Number)
        os.environ[
            "AMZN_BRAKET_DEVICE_ARN"
        ] = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
        # the output S3 bucket, as specified in the CreateJob request’s OutputDataConfig
        os.environ["AMZN_BRAKET_OUT_S3_BUCKET"] = "amazon-braket-us-east-1-oq-testing"
        # the entry point as specified in the CreateJob request’s ScriptModeConfig
        os.environ["AMZN_BRAKET_SCRIPT_ENTRY_POINT"] = ""
        # the compression type as specified in the CreateJob request’s ScriptModeConfig
        os.environ["AMZN_BRAKET_SCRIPT_COMPRESSION_TYPE"] = ""
        # the S3 location of the user’s script as specified in the CreateJob request’s ScriptModeConfig
        os.environ["AMZN_BRAKET_SCRIPT_S3_URI"] = ""
        # the S3 location where the SDK would store the task results by default for the job
        os.environ["AMZN_BRAKET_TASK_RESULTS_S3_URI"] = ""
        # the S3 location where the job results would be stored, as specified in CreateJob request’s OutputDataConfig
        os.environ["AMZN_BRAKET_JOB_RESULTS_S3_PATH"] = ""
        # the string that should be passed to CreateQuantumTask’s jobToken parameter for quantum tasks created in the job container
        # os.environ["AMZN_BRAKET_JOB_TOKEN"] = ''

        self.vc = MinimumVertexCover(
            nw.circulant_graph(10, [1]), field=1.0, penalty=10
        ).get_qubo_problem()

    def testOsEnvironAssignment(self):

        qaoa_workflow = AWSJobs(algorithm="QaoA")
        assert qaoa_workflow.algorithm == "qaoa"
        assert qaoa_workflow.input_dir == "./tests/jobs_test_input/"
        assert qaoa_workflow.device.device_name == os.environ["AMZN_BRAKET_DEVICE_ARN"]

        rqaoa_workflow = AWSJobs(algorithm="rqAoa")
        assert rqaoa_workflow.algorithm == "rqaoa"
        assert rqaoa_workflow.device.device_name == os.environ["AMZN_BRAKET_DEVICE_ARN"]

    @pytest.mark.api
    def testCreateAwsQAOA(self):
        """
        Test Creation and Loading of input_data
        """

        input_data_path = os.path.join(
            os.environ["AMZN_BRAKET_INPUT_DIR"], "input_data/"
        )

        # Create the qubo and the qaoa
        q = QAOA()
        q.set_device(create_device('aws', 'arn:aws:braket:::device/quantum-simulator/amazon/sv1'))
        q.compile(self.vc)
        q.dump(file_name='openqaoa_params.json', file_path=input_data_path, prepend_id=False, overwrite=True)

        # Create an aws workflow and try check that loading the json gives the same params
        job = AWSJobs(algorithm="QAOA")
        job.load_compile_data()

        assert job.workflow.asdict() == q.asdict()

    @pytest.mark.api
    def testCreateAwsRecursiveQAOA(self):
        """
        Test Creation and Loading of input_data
        """

        input_data_path = os.path.join(
            os.environ["AMZN_BRAKET_INPUT_DIR"], "input_data/"
        )

        # Create the qubo and the qaoa
        r = RQAOA()
        r.set_device(create_device('aws', 'arn:aws:braket:::device/quantum-simulator/amazon/sv1'))
        r.set_classical_optimizer(maxiter=3, save_intermediate=False)
        r.compile(self.vc)
        r.dump(file_name='openqaoa_params.json', file_path=input_data_path, prepend_id=False, overwrite=True)

        # Create an aws workflow and try check that loading the json gives the same params
        job = AWSJobs(algorithm="RQAOA")
        job.load_compile_data()

        assert job.workflow.asdict() == r.asdict()


    @pytest.mark.api
    def testEndToEndLocalQAOA(self):
        """
        Test Creation and Loading of input_data
        """

        input_data_path = os.path.join(
            os.environ["AMZN_BRAKET_INPUT_DIR"], "input_data/"
        )
        os.environ[
            "AMZN_BRAKET_JOB_RESULTS_DIR"
        ] = "/oq_release_tests/testing_jobs/EndToEnd"

        # Create the qubo and the qaoa
        q = QAOA()
        q.set_classical_optimizer(maxiter=2)
        q.compile(self.vc)
        q.dump(file_name='openqaoa_params.json', file_path=input_data_path, prepend_id=False, overwrite=True)

        # Create an aws workflow and try check that loading the json gives the same params
        job = AWSJobs(algorithm="QAOA")
        job.load_compile_data()
        job.run_workflow()


    @pytest.mark.api
    def testEndToEndLocalRQAOA(self):
        """
        Test Creation and Loading of input_data
        """

        input_data_path = os.path.join(
            os.environ["AMZN_BRAKET_INPUT_DIR"], "input_data/"
        )
        os.environ[
            "AMZN_BRAKET_JOB_RESULTS_DIR"
        ] = "/oq_release_tests/testing_jobs/EndToEnd"

        # Create the qubo and the qaoa
        r = RQAOA()
        r.set_rqaoa_parameters(steps = 2, n_cutoff = 3)
        r.set_classical_optimizer(maxiter=3, save_intermediate=False)
        r.compile(self.vc)
        r.dump(file_name='openqaoa_params.json', file_path=input_data_path, prepend_id=False, overwrite=True)

        # Create an aws workflow and try check that loading the json gives the same params
        job = AWSJobs(algorithm="RQAOA")
        job.load_compile_data()
        job.run_workflow()

        assert job.completed == True

    @pytest.mark.docker_aws
    def testLocalJob(self):
        """Test an end-to-end qaoa running on a local docker instance"""

        input_data_path = os.path.join(
            os.environ["AMZN_BRAKET_INPUT_DIR"], "input_data/"
        )

        # Create the qubo and the qaoa
        q = QAOA()
        q.set_device(create_device('aws', 'arn:aws:braket:::device/quantum-simulator/amazon/sv1'))
        q.compile(self.vc)
        q.dump(file_name='openqaoa_params.json', file_path=input_data_path, prepend_id=False, overwrite=True)
        
        job = LocalQuantumJob.create(
            device="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            source_module="./tests/jobs_test_input/aws_braket_source_module/openqaoa_qaoa_script.py",
            image_uri="amazon-braket-oq-dev",
            input_data={"input_data": input_data_path},
        )

        assert job.state() == 'COMPLETED'

    # @pytest.mark.docker_aws
    # def testLocalJobRQAOA(self):
    #     """Test an end-to-end rqaoa running on a local docker instance"""

    #     input_data_path = os.path.join(
    #         os.environ["AMZN_BRAKET_INPUT_DIR"], "input_data/")

    #     # Create the rqaoa
    #     r = RQAOA()
    #     r.set_rqaoa_parameters(n_cutoff=6)
    #     r.set_classical_optimizer(maxiter=3, save_intermediate=False)
    #     r.set_device(create_device('aws', 'arn:aws:braket:::device/quantum-simulator/amazon/sv1'))
    #     r.compile(self.vc)
    #     r.dump(file_name='openqaoa_params.json', file_path=input_data_path, prepend_id=False, overwrite=True)

    #     job = LocalQuantumJob.create(
    #         device="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
    #         source_module="./tests/jobs_test_input/aws_braket_source_module/openqaoa_rqaoa_script.py",
    #         image_uri="amazon-braket-oq-dev",
    #         input_data={"input_data": input_data_path},
    #     )

    #     assert job.state() == 'COMPLETED'


if __name__ == "__main__":
    unittest.main()
