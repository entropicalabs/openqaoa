import numpy as np
from typing import List
from boto3.session import Session
from botocore.exceptions import NoRegionError
from braket.aws import AwsDevice
from braket.aws.aws_session import AwsSession

from openqaoa.backends.devices_core import DeviceBase


class DeviceAWS(DeviceBase):

    """
    Contains the required information and methods needed to access QPUs hosted
    on AWS Braket.

    Attributes
    ----------
    available_qpus: `list`
        When connection to AWS is established, this attribute contains a list
        of device names which can be used to access the selected device by
        reinitialising the Access Object with the name of the available device
        as input to the device_name parameter.
    n_qubits: `int`
        The maximum number of qubits available for the selected backend. Only
        available if check_connection method is executed and a connection to the
        qpu and provider is established.
    """

    def __init__(
        self,
        device_name: str,
        s3_bucket_name: str = None,
        aws_region: str = None,
        folder_name: str = "openqaoa",
    ):
        """Input the device arn and the name of the folder in which all the
        results for the QPU runs would be saved on the pre-defined s3 bucket.
        Note that the user is required to authenticate through the AWS CLI
        before being able to use this Device object.

        See: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html for further details.

        Parameters
        ----------
        device_name: `str`
                The ARN string of the braket QPU/simulator to be used
        s3_bucket_name: `str`
            The name of S3 Bucket where the Braket run results will be saved.
        aws_region: `str`
            The aws region in which the QPU/simulator is located. Defaults to the
            region set in aws cli config.
        folder_name: `str`
            The name of the folder in the s3 bucket that will contain the results
            from the tasks performed in this run.
        """

        self.device_name = device_name
        self.device_location = "aws"
        self.s3_bucket_name = s3_bucket_name
        self.aws_region = aws_region
        self.folder_name = folder_name

        self.provider_connected = None
        self.qpu_connected = None

    def check_connection(self) -> bool:
        self.provider_connected = self._check_provider_connection()

        if self.provider_connected == False:
            return self.provider_connected

        # Only QPUs that are available for the specified aws region on Braket
        # will be shown. We filter out QPUs that do not work with the circuit model
        sess_devices = self.aws_session.search_devices()

        device_filter = np.multiply(
            [each_dict["deviceStatus"] == "ONLINE" for each_dict in sess_devices],
            [
                each_dict["providerName"] != "D-Wave Systems"
                for each_dict in sess_devices
            ],
        )
        active_devices = np.array(sess_devices)[device_filter].tolist()

        self.available_qpus = [
            backend_dict["deviceArn"] for backend_dict in active_devices
        ]

        if self.device_name == "":
            return self.provider_connected

        self.qpu_connected = self._check_backend_connection()

        if self.provider_connected and self.qpu_connected:
            return True
        else:
            return False

    def _check_backend_connection(self) -> bool:
        if self.device_name in self.available_qpus:
            self.backend_device = AwsDevice(self.device_name, self.aws_session)
        else:
            print(
                """
                These are the only available devices for this aws region: 
                {}. Try a different aws region if the device you are looking 
                for is not in the list.'
                """.format(
                    self.available_qpus
                )
            )
            return False

        # Get the maximum number of qubits for that particular AWS Backend
        try:
            self.n_qubits = self.backend_device.properties.paradigm.qubitCount
        except AttributeError:
            print(
                "OpenQAOA is unable to retrieve the number of qubits available in the selected QPU."
            )
            return False
        else:
            return True

    def _check_provider_connection(self) -> bool:
        try:
            sess = Session(region_name=self.aws_region)
            self.aws_session = AwsSession(sess, default_bucket=self.s3_bucket_name)
            self.aws_region = self.aws_session.region
            self.s3_bucket_name = self.aws_session.default_bucket()
            return True
        except NoRegionError:
            self.aws_session = None
            return True
        except Exception as e:
            print(
                "An Exception has occured when trying to connect with the provider."
                "You are required to authenticate through the AWS CLI in order to connect to the Braket QPUs."
                "Please check if you have properly set it up."
                "See: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html for further details. : {}".format(
                    e
                )
            )
            return False

    def connectivity(self) -> List[List[int]]:
        aws_connectivity = self.backend_device.topology_graph.edges
        connectivity_list = [list(edge) for edge in aws_connectivity]
        return connectivity_list
