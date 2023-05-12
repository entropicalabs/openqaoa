import unittest

from openqaoa.optimizers.logger_vqa import (
    Logger,
    LoggerVariable,
    LoggerVariableFactory,
    EmptyValue,
    AppendValue,
    ReplaceValue,
    IfLowerDo,
    IfHigherDo,
)


class TestingLoggerClass(unittest.TestCase):
    def test_logger_var_setting_1(self):
        test_var = LoggerVariable("test_var", EmptyValue, EmptyValue)

        test_var.update(10)

        self.assertEqual(test_var.history, [])
        self.assertEqual(test_var.best, [])

    def test_logger_var_setting_2(self):
        test_var = LoggerVariable("test_var", AppendValue, AppendValue)

        test_var.update(5)
        test_var.update(10)

        self.assertEqual(test_var.history, [5, 10])
        self.assertEqual(test_var.best, [5, 10])

    def test_logger_var_setting_3(self):
        test_var = LoggerVariable("test_var", ReplaceValue, ReplaceValue)

        test_var.update(5)
        test_var.update(10)

        self.assertEqual(test_var.history, [10])
        self.assertEqual(test_var.best, [10])

    def test_logger_var_setting_4(self):
        test_var = LoggerVariable(
            "test_var", IfLowerDo(EmptyValue), IfLowerDo(EmptyValue)
        )

        test_var.update(5)  # First value gets appended regardless
        test_var.update(10)

        self.assertEqual(test_var.history, [5])
        self.assertEqual(test_var.best, [5])

        test_var.update(1)

        self.assertEqual(test_var.history, [])
        self.assertEqual(test_var.best, [])

    def test_logger_var_setting_5(self):
        test_var = LoggerVariable(
            "test_var", IfLowerDo(AppendValue), IfLowerDo(AppendValue)
        )

        test_var.update(5)  # First value gets appended regardless
        test_var.update(10)

        self.assertEqual(test_var.history, [5])
        self.assertEqual(test_var.best, [5])

        test_var.update(1)

        self.assertEqual(test_var.history, [5, 1])
        self.assertEqual(test_var.best, [5, 1])

    def test_logger_var_setting_6(self):
        test_var = LoggerVariable(
            "test_var", IfLowerDo(ReplaceValue), IfLowerDo(ReplaceValue)
        )

        test_var.update(5)  # First value gets appended regardless
        test_var.update(10)

        self.assertEqual(test_var.history, [5])
        self.assertEqual(test_var.best, [5])

        test_var.update(1)

        self.assertEqual(test_var.history, [1])
        self.assertEqual(test_var.best, [1])

    def test_logger_var_setting_7(self):
        test_var = LoggerVariable(
            "test_var", IfHigherDo(EmptyValue), IfHigherDo(EmptyValue)
        )

        test_var.update(5)  # First value gets appended regardless
        test_var.update(1)

        self.assertEqual(test_var.history, [5])
        self.assertEqual(test_var.best, [5])

        test_var.update(10)

        self.assertEqual(test_var.history, [])
        self.assertEqual(test_var.best, [])

    def test_logger_var_setting_8(self):
        test_var = LoggerVariable(
            "test_var", IfHigherDo(AppendValue), IfHigherDo(AppendValue)
        )

        test_var.update(5)  # First value gets appended regardless
        test_var.update(1)

        self.assertEqual(test_var.history, [5])
        self.assertEqual(test_var.best, [5])

        test_var.update(10)

        self.assertEqual(test_var.history, [5, 10])
        self.assertEqual(test_var.best, [5, 10])

    def test_logger_var_setting_9(self):
        test_var = LoggerVariable(
            "test_var", IfHigherDo(ReplaceValue), IfHigherDo(ReplaceValue)
        )

        test_var.update(5)  # First value gets appended regardless
        test_var.update(1)

        self.assertEqual(test_var.history, [5])
        self.assertEqual(test_var.best, [5])

        test_var.update(10)

        self.assertEqual(test_var.history, [10])
        self.assertEqual(test_var.best, [10])

    def test_logger_var_methods(self):
        test_var = LoggerVariable("test_var", ReplaceValue, ReplaceValue)
        test_var_2 = LoggerVariable("test_var", ReplaceValue, ReplaceValue)

        test_var.update(1)

        test_var_2.update_history(1)
        test_var_2.update_best(1)

        self.assertEqual(test_var.history, test_var_2.history)
        self.assertEqual(test_var.best, test_var_2.best)

    def test_logger_var_fact_hist_bool(self):
        var_1 = LoggerVariableFactory.create_logger_variable(
            "new_attribute", True, "Replace"
        )
        var_2 = LoggerVariableFactory.create_logger_variable(
            "new_attribute", False, "Replace"
        )

        var_1.update(1)
        var_2.update(1)

        var_1.update(2)
        var_2.update(2)

        self.assertEqual(var_1.name, "new_attribute")
        self.assertEqual(var_2.name, "new_attribute")
        self.assertEqual(var_1.history, [1, 2])
        self.assertEqual(var_2.history, [])

    def test_logger_var_fact_best_str(self):
        """
        Testing all possible settings of a variable created through the Factory
        Class
        """

        var_1 = LoggerVariableFactory.create_logger_variable(
            "new_attribute", True, "Append"
        )
        var_2 = LoggerVariableFactory.create_logger_variable(
            "new_attribute", True, "Replace"
        )
        var_3 = LoggerVariableFactory.create_logger_variable(
            "new_attribute", True, "LowestSoFar"
        )
        var_4 = LoggerVariableFactory.create_logger_variable(
            "new_attribute", True, "LowestOnly"
        )
        var_5 = LoggerVariableFactory.create_logger_variable(
            "new_attribute", True, "HighestSoFar"
        )
        var_6 = LoggerVariableFactory.create_logger_variable(
            "new_attribute", True, "HighestOnly"
        )

        values_list = [5, 7, 10, 1]

        for each_value in values_list:
            var_1.update(each_value)
            var_2.update(each_value)
            var_3.update(each_value)
            var_4.update(each_value)
            var_5.update(each_value)
            var_6.update(each_value)

        self.assertEqual(var_1.best, [5, 7, 10, 1])
        self.assertEqual(var_2.best, [1])
        self.assertEqual(var_3.best, [5, 1])
        self.assertEqual(var_4.best, [1])
        self.assertEqual(var_5.best, [5, 7, 10])
        self.assertEqual(var_6.best, [10])

    def test_logger_obj_update_struct_1(self):
        """
        The best update structure ensures that the best value for that particular
        attribute respects a relation with another attribute.

        In the code below, the following can be observed:
        Even though the lowest in attribute 2 is 10. The best of attribute 2 is
        15 instead. This is because the best value of attribute 2 depends on whether
        attribute 1 for that set of update values is the highest. If it is not the
        highest value, attribute 2's best value is not updated. This way updating
        the best value of attribute 2 respects the update on the best value of
        attribute 1.

        Histories are always updated Indepedent of the best update structure provided.
        """

        logger_obj = Logger(
            {
                "attribute_1": {
                    "history_update_bool": True,
                    "best_update_string": "HighestOnly",
                },
                "attribute_2": {
                    "history_update_bool": True,
                    "best_update_string": "LowestSoFar",
                },
                "attribute_3": {
                    "history_update_bool": False,
                    "best_update_string": "Replace",
                },
            },
            {
                "root_nodes": ["attribute_1"],
                "best_update_structure": (
                    ["attribute_1", "attribute_2"],
                    ["attribute_1", "attribute_3"],
                ),
            },
        )

        logger_obj.log_variables(
            {"attribute_1": 10, "attribute_2": 3, "attribute_3": "string 1"}
        )

        self.assertEqual(logger_obj.attribute_1.history, [10])
        self.assertEqual(logger_obj.attribute_2.history, [3])
        self.assertEqual(logger_obj.attribute_3.history, [])

        self.assertEqual(logger_obj.attribute_1.best, [10])
        self.assertEqual(logger_obj.attribute_2.best, [3])
        self.assertEqual(logger_obj.attribute_3.best, ["string 1"])

        logger_obj.log_variables(
            {"attribute_1": 5, "attribute_2": 1, "attribute_3": "string 2"}
        )

        self.assertEqual(logger_obj.attribute_1.history, [10, 5])
        self.assertEqual(logger_obj.attribute_2.history, [3, 1])
        self.assertEqual(logger_obj.attribute_3.history, [])

        self.assertEqual(logger_obj.attribute_1.best, [10])
        self.assertEqual(logger_obj.attribute_2.best, [3])
        self.assertEqual(logger_obj.attribute_3.best, ["string 1"])

        logger_obj.log_variables(
            {"attribute_1": 15, "attribute_2": 4, "attribute_3": "string 3"}
        )

        self.assertEqual(logger_obj.attribute_1.history, [10, 5, 15])
        self.assertEqual(logger_obj.attribute_2.history, [3, 1, 4])
        self.assertEqual(logger_obj.attribute_3.history, [])

        self.assertEqual(logger_obj.attribute_1.best, [15])
        self.assertEqual(logger_obj.attribute_2.best, [3])
        self.assertEqual(logger_obj.attribute_3.best, ["string 3"])

        logger_obj.log_variables(
            {"attribute_1": 20, "attribute_2": 2, "attribute_3": "string 4"}
        )

        self.assertEqual(logger_obj.attribute_1.history, [10, 5, 15, 20])
        self.assertEqual(logger_obj.attribute_2.history, [3, 1, 4, 2])
        self.assertEqual(logger_obj.attribute_3.history, [])

        self.assertEqual(logger_obj.attribute_1.best, [20])
        self.assertEqual(logger_obj.attribute_2.best, [3, 2])
        self.assertEqual(logger_obj.attribute_3.best, ["string 4"])

    def test_logger_obj_update_struct_2(self):
        """
        Single layer test
        """

        logger_obj = Logger(
            {
                "attribute_1": {
                    "history_update_bool": True,
                    "best_update_string": "HighestOnly",
                },
                "attribute_2": {
                    "history_update_bool": True,
                    "best_update_string": "LowestSoFar",
                },
                "attribute_3": {
                    "history_update_bool": False,
                    "best_update_string": "Replace",
                },
            },
            {
                "root_nodes": ["attribute_1", "attribute_2", "attribute_3"],
                "best_update_structure": ([]),
            },
        )

        logger_obj.log_variables(
            {"attribute_1": 10, "attribute_2": 3, "attribute_3": "string 1"}
        )

        self.assertEqual(logger_obj.attribute_1.history, [10])
        self.assertEqual(logger_obj.attribute_2.history, [3])
        self.assertEqual(logger_obj.attribute_3.history, [])

        self.assertEqual(logger_obj.attribute_1.best, [10])
        self.assertEqual(logger_obj.attribute_2.best, [3])
        self.assertEqual(logger_obj.attribute_3.best, ["string 1"])

        logger_obj.log_variables(
            {"attribute_1": 5, "attribute_2": 1, "attribute_3": "string 2"}
        )

        self.assertEqual(logger_obj.attribute_1.history, [10, 5])
        self.assertEqual(logger_obj.attribute_2.history, [3, 1])
        self.assertEqual(logger_obj.attribute_3.history, [])

        self.assertEqual(logger_obj.attribute_1.best, [10])
        self.assertEqual(logger_obj.attribute_2.best, [3, 1])
        self.assertEqual(logger_obj.attribute_3.best, ["string 2"])

        logger_obj.log_variables(
            {"attribute_1": 15, "attribute_2": 4, "attribute_3": "string 3"}
        )

        self.assertEqual(logger_obj.attribute_1.history, [10, 5, 15])
        self.assertEqual(logger_obj.attribute_2.history, [3, 1, 4])
        self.assertEqual(logger_obj.attribute_3.history, [])

        self.assertEqual(logger_obj.attribute_1.best, [15])
        self.assertEqual(logger_obj.attribute_2.best, [3, 1])
        self.assertEqual(logger_obj.attribute_3.best, ["string 3"])

        logger_obj.log_variables(
            {"attribute_1": 20, "attribute_2": 2, "attribute_3": "string 4"}
        )

        self.assertEqual(logger_obj.attribute_1.history, [10, 5, 15, 20])
        self.assertEqual(logger_obj.attribute_2.history, [3, 1, 4, 2])
        self.assertEqual(logger_obj.attribute_3.history, [])

        self.assertEqual(logger_obj.attribute_1.best, [20])
        self.assertEqual(logger_obj.attribute_2.best, [3, 1])
        self.assertEqual(logger_obj.attribute_3.best, ["string 4"])

    def test_logger_obj_update_struct_3(self):
        """
        Multi layer test
        """

        logger_obj = Logger(
            {
                "attribute_1": {
                    "history_update_bool": True,
                    "best_update_string": "HighestOnly",
                },
                "attribute_2": {
                    "history_update_bool": True,
                    "best_update_string": "LowestSoFar",
                },
                "attribute_3": {
                    "history_update_bool": False,
                    "best_update_string": "Replace",
                },
            },
            {
                "root_nodes": ["attribute_1"],
                "best_update_structure": (
                    ["attribute_1", "attribute_2"],
                    ["attribute_2", "attribute_3"],
                ),
            },
        )

        logger_obj.log_variables(
            {"attribute_1": 10, "attribute_2": 3, "attribute_3": "string 1"}
        )

        self.assertEqual(logger_obj.attribute_1.history, [10])
        self.assertEqual(logger_obj.attribute_2.history, [3])
        self.assertEqual(logger_obj.attribute_3.history, [])

        self.assertEqual(logger_obj.attribute_1.best, [10])
        self.assertEqual(logger_obj.attribute_2.best, [3])
        self.assertEqual(logger_obj.attribute_3.best, ["string 1"])

        logger_obj.log_variables(
            {"attribute_1": 5, "attribute_2": 1, "attribute_3": "string 2"}
        )

        self.assertEqual(logger_obj.attribute_1.history, [10, 5])
        self.assertEqual(logger_obj.attribute_2.history, [3, 1])
        self.assertEqual(logger_obj.attribute_3.history, [])

        self.assertEqual(logger_obj.attribute_1.best, [10])
        self.assertEqual(logger_obj.attribute_2.best, [3])
        self.assertEqual(logger_obj.attribute_3.best, ["string 1"])

        logger_obj.log_variables(
            {"attribute_1": 15, "attribute_2": 4, "attribute_3": "string 3"}
        )

        self.assertEqual(logger_obj.attribute_1.history, [10, 5, 15])
        self.assertEqual(logger_obj.attribute_2.history, [3, 1, 4])
        self.assertEqual(logger_obj.attribute_3.history, [])

        self.assertEqual(logger_obj.attribute_1.best, [15])
        self.assertEqual(logger_obj.attribute_2.best, [3])
        self.assertEqual(logger_obj.attribute_3.best, ["string 1"])

        logger_obj.log_variables(
            {"attribute_1": 20, "attribute_2": 2, "attribute_3": "string 4"}
        )

        self.assertEqual(logger_obj.attribute_1.history, [10, 5, 15, 20])
        self.assertEqual(logger_obj.attribute_2.history, [3, 1, 4, 2])
        self.assertEqual(logger_obj.attribute_3.history, [])

        self.assertEqual(logger_obj.attribute_1.best, [20])
        self.assertEqual(logger_obj.attribute_2.best, [3, 2])
        self.assertEqual(logger_obj.attribute_3.best, ["string 4"])


if __name__ == "__main__":
    unittest.main()
